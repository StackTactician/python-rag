# NanoRAG

NanoRAG is a lightweight Retrieval-Augmented Generation (RAG) project for querying local Markdown files with an OpenRouter-hosted LLM.

It includes:
- **Document ingestion** into a local Chroma vector database.
- **Interactive and one-shot CLI querying**.
- **FastAPI service** for programmatic querying and ingestion.

## Architecture Overview

NanoRAG follows a simple pipeline:

1. **Load Markdown files** from a directory (default: `data/`).
2. **Split documents** into overlapping chunks.
3. **Embed and store chunks** in a local Chroma DB (`chroma_db/`).
4. **Retrieve top matches** for each query.
5. **Send retrieved context + question** to OpenRouter for final answer generation.

Core modules:
- `src/ingest.py`: loading, splitting, and persisting chunks.
- `src/rag.py`: retrieval + OpenRouter request/stream handling.
- `src/cli.py`: terminal interface.
- `src/api.py`: REST API interface.

## Requirements

- Python 3.10+
- An OpenRouter API key

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env  # if you create one
# then set:
OPENROUTER_API_KEY=your_key_here
```

At minimum, NanoRAG requires `OPENROUTER_API_KEY` in your environment or `.env` file.

## Ingestion

Place your `.md` files inside `data/` (or any directory of your choice), then run:

```bash
python src/ingest.py
```

Use a custom source directory:

```bash
python src/ingest.py path/to/notes
```

Clear and rebuild vectors:

```bash
python src/ingest.py data --clear
```

### Ingestion Notes

- Files are discovered recursively with `**/*.md`.
- Chunks use deterministic IDs (`source:start_index:hash`) to reduce duplication.
- If ingestion fails with a DB lock, stop running CLI/API processes and retry.

## CLI Usage

Interactive mode:

```bash
python src/cli.py
```

Single query mode:

```bash
python src/cli.py "What is retrieval-augmented generation?"
```

Specify a model:

```bash
python src/cli.py -m openrouter/aurora-alpha "Summarize the architecture"
python src/cli.py -i -m openrouter/aurora-alpha
```

## API Usage

Start the server:

```bash
python src/api.py
```

The API runs on `http://localhost:8000` by default.

### Endpoints

#### `POST /query`

Request body:

```json
{
  "query": "What does this project do?",
  "model": "openrouter/aurora-alpha"
}
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "data/example.md",
      "content_preview": "First 200 chars...",
      "score": 0.1234
    }
  ]
}
```

#### `POST /ingest?clear_existing=true`

Triggers ingestion from the default data path and optionally clears existing vectors first.

### API Documentation

OpenAPI/Swagger UI is available at:

- `http://localhost:8000/docs`

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

## Troubleshooting

- **"OPENROUTER_API_KEY is not set"**
  - Add the key to `.env` or export it in your shell.
- **No relevant context found**
  - Ensure documents were ingested and query terms appear in your Markdown files.
- **Database lock errors during ingest**
  - Stop concurrent processes using `chroma_db/` and rerun ingestion.

## License

Add your preferred license in this repository (for example MIT, Apache-2.0, or proprietary).
