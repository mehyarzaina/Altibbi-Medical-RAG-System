# RAG/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection
from mcp.server.fastmcp import FastMCP
import requests
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_URL     = os.getenv("EMBEDDING_URL")
ZILLIZ_URI        = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN      = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
API_TOKEN         = os.getenv("API_TOKEN", "")  

connections.connect(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
col = Collection(name=COLLECTION_NAME)
col.load()

# ── Auth middleware ────────────────────────────────────────────────────────

UNPROTECTED_PATHS = {"/docs", "/openapi.json", "/redoc"}
UNPROTECTED_HOSTS = {"localhost", "127.0.0.1"}

class StaticTokenMiddleware:
    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.token:
            path = scope.get("path", "")
            headers = dict(scope.get("headers", []))
            host = headers.get(b"host", b"").decode().split(":")[0]

            # Skip auth for local requests and unprotected paths
            if path not in UNPROTECTED_PATHS and host not in UNPROTECTED_HOSTS:
                auth = headers.get(b"authorization", b"").decode()
                api_key = headers.get(b"x-api-key", b"").decode().strip()

                token_valid = (
                    (auth.lower().startswith("bearer ") and auth[7:].strip() == self.token)
                    or api_key == self.token
                )

                if not token_valid:
                    body = b'{"detail": "Unauthorized. Provide Authorization: Bearer <token> or X-API-Key: <token>."}'
                    await send({
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"www-authenticate", b'Bearer error="invalid_token"'),
                        ],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    })
                    return

        await self.app(scope, receive, send)

# ── App setup ──────────────────────────────────────────────────────────────

mcp = FastMCP("Altibbi RAG")
app = FastAPI(title="Altibbi RAG")
app.mount("/mcp", mcp.sse_app())
app.add_middleware(StaticTokenMiddleware, token=API_TOKEN)

# ── Request / Response models for Fast API ──────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

# ── Shared helpers ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": EMBEDDING_API_KEY, "Content-Type": "application/json"}
    payload = {
        "docs": [text],
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
        "convert_to_float32": True,
        "normalize_vectors": False,
        "batch_size": 1,
    }
    resp = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def search_zilliz(query: str, top_k: int) -> list[dict]:
    query_vec = get_embedding(query)
    results = col.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "title", "author", "category"]
    )
    return [
        {
            "title":    hit.entity.get("title", ""),
            "category": hit.entity.get("category", ""),
            "author":   hit.entity.get("author", ""),
            "text":     hit.entity.get("text", ""),
            "score":    hit.distance,
        }
        for hit in results[0]
    ]

# ── FastAPI endpoints ──────────────────────────────────────────────────────

@app.post("/embed")
def embed(body: EmbedRequest):
    try:
        return {"embedding": get_embedding(body.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search(body: SearchRequest):
    try:
        results = search_zilliz(body.query, body.top_k)
        return {"query": body.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── MCP tools ─────────────────────────────────────────────────────────────
@mcp.tool()
def search_medical_articles(query: str) -> str:  
    """
    Search Altibbi Arabic medical articles for a given query.
    It only answer using the content returned here.
    Do NOT use any outside knowledge. If the answer is not in the results, say so.
    Always end your response with a 'Sources' section listing all articles used.
    """
    results = search_zilliz(query, top_k=3) 
    if not results:
        return "No relevant articles found in the Altibbi database. Do not use outside knowledge."

    output = "IMPORTANT: Answer using ONLY the information below. Do not use outside knowledge.\n\n"
    output += f"Query: '{query}'\n"
    output += "=" * 50 + "\n\n"

    for i, r in enumerate(results, 1):
        output += f"[Source {i}] {r['title']}\n"
        output += f"Category: {r['category']} | Author: {r['author']} | Score: {r['score']:.3f}\n"
        output += f"{r['text']}\n"
        output += "-" * 40 + "\n\n"

    output += "=" * 50 + "\n"
    output += "End of sources. Base your answer strictly on the above content only.\n\n"
    output += "REQUIRED: After your answer, you MUST include this exact section:\n"
    output += "---\n"
    output += "📚 Sources from Altibbi database:\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. {r['title']} (Category: {r['category']} | Relevance: {r['score']:.0%})\n"

    return output

@mcp.tool()
def embed_text(text: str) -> str:
    """Get the embedding vector for a text."""
    embedding = get_embedding(text)
    return f"Embedding dim: {len(embedding)}\nFirst 5 values: {embedding[:5]}"

# ── Run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)