import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Lightweight stubs so tests can run without installing project deps.
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("requests", types.SimpleNamespace(RequestException=Exception, post=lambda **kwargs: None))

from src import rag
from src.rag import _extract_response_text, _parse_stream_lines, query_rag


class FakeChroma:
    """Configurable fake vector store for query_rag tests."""

    results = []

    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function

    def similarity_search_with_score(self, query_text, k=5):
        return list(self.results)


class FakeResponse:
    def __init__(self, status_code=200, json_payload=None, text="", lines=None):
        self.status_code = status_code
        self._json_payload = json_payload or {}
        self.text = text
        self._lines = lines or []

    def json(self):
        return self._json_payload

    def iter_lines(self):
        return iter(self._lines)


class RagHelpersTest(unittest.TestCase):
    def setUp(self):
        self.original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

        # Inject fake langchain module imported dynamically inside query_rag.
        self.original_langchain_module = sys.modules.get("langchain_community.vectorstores")
        fake_langchain = types.ModuleType("langchain_community.vectorstores")
        fake_langchain.Chroma = FakeChroma
        sys.modules["langchain_community.vectorstores"] = fake_langchain

    def tearDown(self):
        if self.original_api_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.original_api_key

        if self.original_langchain_module is None:
            sys.modules.pop("langchain_community.vectorstores", None)
        else:
            sys.modules["langchain_community.vectorstores"] = self.original_langchain_module

    def test_parse_stream_lines_yields_only_content_tokens(self):
        lines = [
            b"event: message",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            b"data: {\"choices\":[{\"delta\":{}}]}",
            b"data: malformed-json",
            b"data: [DONE]",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}",
        ]

        self.assertEqual(list(_parse_stream_lines(lines)), ["Hello"])

    def test_extract_response_text_reads_content(self):
        payload = {"choices": [{"message": {"content": "answer"}}]}
        self.assertEqual(_extract_response_text(payload), "answer")

    def test_extract_response_text_raises_for_bad_payload(self):
        with self.assertRaises(ValueError):
            _extract_response_text({"oops": []})

    def test_query_rag_rejects_empty_input(self):
        self.assertEqual(list(query_rag("   ")), [("Please provide a non-empty query.", [])])

    def test_query_rag_requires_api_key(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        self.assertEqual(list(query_rag("hi")), [("Error: OPENROUTER_API_KEY is not set.", [])])

    def test_query_rag_handles_no_search_results(self):
        FakeChroma.results = []
        with patch.object(rag, "get_embedding_function", return_value=object()):
            self.assertEqual(
                list(query_rag("question")),
                [("I couldn't find any relevant context to answer your question.", [])],
            )

    def test_query_rag_non_stream_success(self):
        doc = SimpleNamespace(page_content="ctx", metadata={"source": "a.md"})
        FakeChroma.results = [(doc, 0.2)]
        response = FakeResponse(json_payload={"choices": [{"message": {"content": "final answer"}}]})

        with patch.object(rag, "get_embedding_function", return_value=object()), patch.object(
            rag.requests, "post", return_value=response
        ) as mock_post:
            items = list(query_rag("What?", model_name="model/x", stream=False))

        self.assertEqual(items, [("final answer", [(doc, 0.2)])])
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "model/x")

    def test_query_rag_stream_success(self):
        doc = SimpleNamespace(page_content="ctx", metadata={"source": "a.md"})
        FakeChroma.results = [(doc, 0.2)]
        response = FakeResponse(lines=[
            b"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}",
            b"data: [DONE]",
        ])

        with patch.object(rag, "get_embedding_function", return_value=object()), patch.object(
            rag.requests, "post", return_value=response
        ):
            items = list(query_rag("What?", stream=True))

        self.assertEqual(items, [("A", [(doc, 0.2)]), ("B", [(doc, 0.2)])])


if __name__ == "__main__":
    unittest.main()
