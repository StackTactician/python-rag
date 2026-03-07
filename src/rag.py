"""NanoRAG retrieval and generation helpers for OpenRouter-backed QA."""

import json
import logging
import os
import warnings
from collections.abc import Generator, Iterable
from typing import Any

import requests
from dotenv import load_dotenv

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

load_dotenv()

CHROMA_PATH = "chroma_db"
DEFAULT_MODEL = "openrouter/aurora-alpha"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60
_embedding_model = None


def get_embedding_function():
    """Lazily initialize and cache the embeddings model."""
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model


def _build_messages(query_text: str, context_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided context to answer the user's question.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query_text}",
        },
    ]


def _parse_stream_lines(lines: Iterable[bytes]) -> Generator[str, None, None]:
    """Parse OpenRouter server-sent event chunks and yield content tokens."""
    for line in lines:
        if not line:
            continue

        decoded_line = line.decode("utf-8", errors="ignore").strip()
        if not decoded_line.startswith("data: "):
            continue

        data_str = decoded_line[6:]
        if data_str == "[DONE]":
            break

        try:
            data_json: dict[str, Any] = json.loads(data_str)
            content = data_json["choices"][0].get("delta", {}).get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue


def _extract_response_text(response_json: dict[str, Any]) -> str:
    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter response did not contain a valid assistant message") from exc


def query_rag(
    query_text: str,
    model_name: str = DEFAULT_MODEL,
    stream: bool = False,
) -> Generator[tuple[str, list[Any]], None, None]:
    """Run a similarity search and send context to OpenRouter for completion."""
    if not query_text or not query_text.strip():
        yield "Please provide a non-empty query.", []
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        yield "Error: OPENROUTER_API_KEY is not set.", []
        return

    from langchain_community.vectorstores import Chroma

    embeddings = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        yield "I couldn't find any relevant context to answer your question.", []
        return

    context_text = "\n\n---\n\n".join(doc.page_content for doc, _score in results)
    payload = {
        "model": model_name,
        "messages": _build_messages(query_text=query_text, context_text=context_text),
        "stream": stream,
    }

    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "NanoRAG",
            },
            json=payload,
            stream=stream,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            yield f"Error: OpenRouter API returned {response.status_code}: {response.text}", []
            return

        if stream:
            for token in _parse_stream_lines(response.iter_lines()):
                yield token, results
            return

        response_text = _extract_response_text(response.json())
        yield response_text, results
    except requests.RequestException as exc:
        yield f"Error calling OpenRouter: {exc}", []
    except ValueError as exc:
        yield f"Error parsing OpenRouter response: {exc}", []
