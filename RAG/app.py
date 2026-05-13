from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection
from mcp.server.fastmcp import FastMCP
from sqlmodel import Session, select
from typing import Optional
import requests
import os
from dotenv import load_dotenv

from database.database import engine
from database.models import Article

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

# ── MySQL Config ────────────────────────────────────────────────────────
DEFAULT_TOP_K = 3          # default number of results returned
MAX_TOP_K     = 100        # hard upper limit the caller may request
 
 # ── Auth middleware ────────────────────────────────────────────────────────
UNPROTECTED_PATHS = {"/docs", "/openapi.json", "/redoc"} # for swager
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
app.add_middleware(StaticTokenMiddleware, token=API_TOKEN) # authentication layer


# ── Request / Response models for Fast API ──────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text: str
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

# ── Helpers ─────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": EMBEDDING_API_KEY, "Content-Type": "application/json"}
    payload = {
        "docs": [text],
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
        "convert_to_float32": True,
        "normalize_vectors": False,
        "batch_size": 1, # take docs one by one
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
        output_fields=["text", "title", "author", "category", "url"] 
    )
    return [
        {
            "title":    hit.entity.get("title", ""),
            "category": hit.entity.get("category", ""),
            "author":   hit.entity.get("author", ""),
            "url":      hit.entity.get("url", ""),   
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
def search_medical_articles(query: str, top_k: int = 3) -> dict:
    """
    Search Altibbi Arabic medical articles for a given query.
    Returns relevant articles from the database.
    Only answer using the content returned. Do not use outside knowledge.
    If the answer is not in the results, say so.
    Always include the article URL when available.
    """
    results = search_zilliz(query, top_k=top_k)
    if not results:
        return {
            "success": False,
            "message": "No relevant articles found in the Altibbi database.",
            "articles": []
        }
    return {
        "success": True,
        "count": len(results),
        "articles": [
            {
                "title":           r["title"],
                "category":        r["category"],
                "author":          r["author"],
                "url":             r["url"],        
                "relevance_score": round(r["score"], 3),
                "content":         r["text"]
            }
            for r in results
        ]
    }


@mcp.tool()
def embed_text(text: str) -> str:
    """Get the embedding vector for a text."""
    embedding = get_embedding(text)
    return f"Embedding dim: {len(embedding)}\nFirst 5 values: {embedding[:5]}"

 
@mcp.tool()
def search_articles(
    title:       Optional[str] = None,
    author_name: Optional[str] = None,
    category:    Optional[str] = None,
    article_id:  Optional[int] = None,
    top_k:       Optional[int] = None,
) -> dict:
    """
    Search the `articles` MySQL table and return matching records.
 
    Parameters
    ----------
    title       : Partial / full title to search (case-insensitive LIKE match).
    author_name : Filter by author name (case-insensitive LIKE match).
    category    : Filter by category (case-insensitive LIKE match).
    article_id  : Look up a specific article by its primary key.
    top_k       : Number of results to return (default: DEFAULT_TOP_K, max: MAX_TOP_K).
    """
 
    # ── top_k ──────────────────────────────
    k = DEFAULT_TOP_K if top_k is None else int(top_k)
    k = max(1, min(k, MAX_TOP_K))
 
    # ── build query ──────────────────────────────
    with Session(engine) as session:
        statement = select(Article)

        if article_id is not None:
            statement = statement.where(Article.article_id == article_id)
        if title:
            statement = statement.where(Article.title.ilike(f"%{title}%"))
        if author_name:
            statement = statement.where(Article.author_name.ilike(f"%{author_name}%"))
        if category:
            statement = statement.where(Article.category.ilike(f"%{category}%"))

        rows = session.exec(statement.limit(k)).all()
 
    if not rows:
        return {"success": False, "count": 0, "articles": []}
 
    # ── build result dicts ────────────────────────
    articles = []
    for row in rows:
        articles.append({
            "article_id":  row.article_id,
            "title":       row.title,
            "author_name": row.author_name,
            "category":    row.category,
            "url":         row.url,
            "body":        row.body,
        })

    return {"success": True, "count": len(articles), "articles": articles}
  

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)