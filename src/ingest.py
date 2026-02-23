import os
import argparse
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents(source_dir: str):
    """Load markdown files from the source directory recursively."""
    from langchain_community.document_loaders import TextLoader, DirectoryLoader
    
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return []
    
    loader = DirectoryLoader(source_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    return documents

def split_documents(documents):
    """Split documents into smaller chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    """Save chunks to ChromaDB."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from tqdm import tqdm
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    BATCH_SIZE = 50
    total_chunks = len(chunks)
    
    print(f"Saving {total_chunks} chunks to {CHROMA_PATH} in batches of {BATCH_SIZE}...")
    
    try:
        for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Ingesting Batches", unit="batch"):
            batch = chunks[i : i + BATCH_SIZE]
            db.add_documents(batch)
            
        print(f"Successfully saved {total_chunks} chunks.")
    except Exception as e:
        print(f"Error saving to ChromaDB: {e}")
        print("Hint: If you have the CLI or App running, close them first to release the database lock.")

def ingest(source_dir: str):
    print(f"Loading documents from: {source_dir}")
    documents = load_documents(source_dir)
    if not documents:
        print(f"No documents found in {source_dir}")
        return
    
    print("Splitting documents...")
    chunks = split_documents(documents)
    
    print("Saving to Vector DB...")
    save_to_chroma(chunks)
    print("Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument("directory", nargs="?", default=DEFAULT_DATA_PATH, help="Directory containing markdown files to ingest (default: data)")
    args = parser.parse_args()
    
    ingest(args.directory)
