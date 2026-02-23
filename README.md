# RAG with OpenRouter

A small RAG pipeline for querying markdown documents using an LLM via OpenRouter.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your `OPENROUTER_API_KEY` to a `.env` file in the project root.

## Ingesting Documents

Place your `.md` files in the `data/` folder, then run:
```bash
python src/ingest.py
```

You can also pass a different directory:
```bash
python src/ingest.py path/to/your/docs
```

## CLI

```bash
# Interactive mode
python src/cli.py

# Single query
python src/cli.py "What is RAG?"
```

## API

Start the server:
```bash
python src/api.py
```

Endpoints:
- `POST /query` with `{"query": "your question"}`
- `POST /ingest` to trigger ingestion

Swagger docs at `http://localhost:8000/docs`.

## Customisation

To use a different model, change the default in `src/rag.py`.
