import os
import logging
import warnings
from dotenv import load_dotenv

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

load_dotenv()

CHROMA_PATH = "chroma_db"
_embedding_model = None

def get_embedding_function():
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model

def query_rag(query_text: str, model_name: str = "openrouter/aurora-alpha", stream: bool = False):
    from langchain_community.vectorstores import Chroma
    embeddings = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        yield "I couldn't find any relevant context to answer your question.", []
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query_text}"}
    ]

    import requests
    import json

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501", 
                "X-Title": "Local RAG App",
            },
            data=json.dumps({
                "model": model_name,
                "messages": messages,
                "stream": stream
            }),
            stream=stream
        )
        
        if response.status_code == 200:
            if stream:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                content = data_json['choices'][0]['delta'].get('content', '')
                                if content:
                                    yield content, results
                            except:
                                pass
            else:
                response_json = response.json()
                response_text = response_json['choices'][0]['message']['content']
                yield response_text, results
        else:
            yield f"Error: OpenRouter API returned {response.status_code}: {response.text}", []
            
    except Exception as e:
        yield f"Error calling OpenRouter: {str(e)}", []
