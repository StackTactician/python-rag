import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from src import ingest
from src.ingest import _chunk_id


class FakeChroma:
    instances = []

    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.deleted = False
        self.add_calls = []
        self.__class__.instances.append(self)

    def delete_collection(self):
        self.deleted = True

    def add_documents(self, batch, ids):
        self.add_calls.append((list(batch), list(ids)))


class IngestHelpersTest(unittest.TestCase):
    def setUp(self):
        self.original_vectorstores = sys.modules.get("langchain_community.vectorstores")
        self.original_tqdm = sys.modules.get("tqdm")

        fake_vectorstores = types.ModuleType("langchain_community.vectorstores")
        fake_vectorstores.Chroma = FakeChroma
        sys.modules["langchain_community.vectorstores"] = fake_vectorstores

        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
        sys.modules["tqdm"] = fake_tqdm

        FakeChroma.instances = []

    def tearDown(self):
        if self.original_vectorstores is None:
            sys.modules.pop("langchain_community.vectorstores", None)
        else:
            sys.modules["langchain_community.vectorstores"] = self.original_vectorstores

        if self.original_tqdm is None:
            sys.modules.pop("tqdm", None)
        else:
            sys.modules["tqdm"] = self.original_tqdm

    def test_chunk_id_is_stable_and_uses_metadata(self):
        chunk = SimpleNamespace(
            page_content="abc",
            metadata={"source": "data/file.md", "start_index": 12},
        )

        self.assertEqual(_chunk_id(chunk), "data/file.md:12:a9993e364706")

    def test_ingest_skips_when_no_docs(self):
        with patch.object(ingest, "load_documents", return_value=[]), patch.object(
            ingest, "split_documents"
        ) as split_mock, patch.object(ingest, "save_to_chroma") as save_mock:
            ingest.ingest("data")

        split_mock.assert_not_called()
        save_mock.assert_not_called()

    def test_ingest_calls_split_and_save(self):
        docs = [SimpleNamespace(page_content="doc", metadata={})]
        chunks = [SimpleNamespace(page_content="chunk", metadata={"source": "a.md", "start_index": 0})]

        with patch.object(ingest, "load_documents", return_value=docs), patch.object(
            ingest, "split_documents", return_value=chunks
        ) as split_mock, patch.object(ingest, "save_to_chroma") as save_mock:
            ingest.ingest("data", clear_existing=True)

        split_mock.assert_called_once_with(docs)
        save_mock.assert_called_once_with(chunks, clear_existing=True)

    def test_save_to_chroma_batches_and_clears(self):
        chunks = [
            SimpleNamespace(page_content=f"chunk{i}", metadata={"source": "a.md", "start_index": i})
            for i in range(3)
        ]

        with patch.object(ingest, "BATCH_SIZE", 2), patch("src.rag.get_embedding_function", return_value=object()):
            ingest.save_to_chroma(chunks, clear_existing=True)

        self.assertEqual(len(FakeChroma.instances), 2)  # one initial + one after clear
        self.assertTrue(FakeChroma.instances[0].deleted)
        add_calls = FakeChroma.instances[1].add_calls
        self.assertEqual(len(add_calls), 2)
        self.assertEqual(len(add_calls[0][0]), 2)
        self.assertEqual(len(add_calls[1][0]), 1)


if __name__ == "__main__":
    unittest.main()
