import argparse
import hashlib
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
BATCH_SIZE = 50


def load_documents(source_dir: str):
    """Load markdown files from the source directory recursively."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return []

    loader = DirectoryLoader(
        source_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(documents):
    """Split documents into smaller chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def _chunk_id(chunk) -> str:
    source = chunk.metadata.get("source", "unknown")
    start_index = chunk.metadata.get("start_index", 0)
    content_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{start_index}:{content_hash}"


def save_to_chroma(chunks, clear_existing: bool = False):
    """Save chunks to ChromaDB with deterministic IDs to avoid duplicate rows."""
    from langchain_community.vectorstores import Chroma
    from tqdm import tqdm

    if not chunks:
        print("No chunks to save.")
        return

    from src.rag import get_embedding_function

    embeddings = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    if clear_existing:
        print("Clearing existing vector data before ingestion...")
        db.delete_collection()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    ids = [_chunk_id(chunk) for chunk in chunks]
    total_chunks = len(chunks)
    print(f"Saving {total_chunks} chunks to {CHROMA_PATH} in batches of {BATCH_SIZE}...")

    try:
        for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Ingesting Batches", unit="batch"):
            batch = chunks[i : i + BATCH_SIZE]
            batch_ids = ids[i : i + BATCH_SIZE]
            db.add_documents(batch, ids=batch_ids)

        print(f"Successfully saved {total_chunks} chunks.")
    except Exception as exc:
        print(f"Error saving to ChromaDB: {exc}")
        print(
            "Hint: If you have the CLI or API running, close them first to release the database lock."
        )


def ingest(source_dir: str, clear_existing: bool = False):
    print(f"Loading documents from: {source_dir}")
    documents = load_documents(source_dir)
    if not documents:
        print(f"No documents found in {source_dir}")
        return

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Saving to Vector DB...")
    save_to_chroma(chunks, clear_existing=clear_existing)
    print("Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DATA_PATH,
        help="Directory containing markdown files to ingest (default: data)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing vectors before writing the new ingest.",
    )
    args = parser.parse_args()

    ingest(args.directory, clear_existing=args.clear)
