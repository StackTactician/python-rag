# RAG with OpenRouter

This is a Retrieval-Augmented Generation (RAG) application that allows you to query Markdown documents using an OpenRouter LLM.

## Setup

1.  **Install Dependencies**:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
    *(Note: dependencies are already installed if you followed the agent's steps)*

2.  **Environment Variables**:
    Ensure `.env` contains your `OPENROUTER_API_KEY`.

## Running the App



2.  **Upload Data**:
    - Use the sidebar to upload `.md` files.
    - Click "Ingest Uploaded Files".
    - Or manually place files in `data/` and click "Ingest Existing Data Folder".

3.  **Chat**:
    - Ask questions in the chat interface.

## customization

- **Model**: Change the `model` parameter in `src/rag.py` to use a different OpenRouter model.
## CLI Usage

Run the command-line interface:
```bash
# Interactive mode
python src/cli.py

# Single query
python src/cli.py "What is RAG?"
```

## API Usage

Start the API server:
```bash
python src/api.py
```

Endpoints:
- `POST /query`: `{"query": "your question"}`
- `POST /ingest`: Trigger ingestion.

Swagger UI available at `http://localhost:8000/docs`.
