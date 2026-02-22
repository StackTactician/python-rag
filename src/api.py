
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag import query_rag
from src.ingest import ingest
from typing import List, Optional

app = FastAPI(title="RAG API", description="API for RAG System")

class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    source: str
    content_preview: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        response_text, sources = query_rag(request.query)
        
        formatted_sources = []
        for doc, score in sources:
            formatted_sources.append(Source(
                source=doc.metadata.get('source', 'Unknown'),
                content_preview=doc.page_content[:200],
                score=score
            ))
            
        return QueryResponse(answer=response_text, sources=formatted_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_endpoint():
    try:
        ingest()
        return {"status": "success", "message": "Ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
