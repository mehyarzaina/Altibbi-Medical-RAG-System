from pymilvus import connections, Collection
import os
from dotenv import load_dotenv
import requests
from groq import Groq


load_dotenv()


# ── Load env variables ─────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_URL = os.getenv("EMBEDDING_URL")

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ── Setup ──────────────────────────────────────────
connections.connect(
    uri=ZILLIZ_URI,
    token=ZILLIZ_TOKEN
)

col = Collection(name=COLLECTION_NAME)
col.load()

# ── Step 1: Embed the user query (same API you used before) ────────────────
def embed_query(text):
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
    print("✅ Query embedded")
    return resp.json()["embeddings"][0]

# ── Step 2: Search Zilliz for relevant chunks ──────────────────────────────
def retrieve(query, top_k=5):
    query_vec = embed_query(query)
    print(f"🔍 Searching Zilliz for top {top_k} chunks...")

    results = col.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "title", "author", "category", "pub_date"]
    )
    print(f"✅ Retrieved {len(results[0])} chunks")


    chunks = []
    for hit in results[0]:
        chunks.append({
            "text":     hit.entity.get("text"),
            "title":    hit.entity.get("title"),
            "author":   hit.entity.get("author"),
            "category": hit.entity.get("category"),
            "score":    hit.distance,
        })
    return chunks

# ── Step 3: Build prompt + call Gemini ────────────────────────────────────
def rag_query(user_question):
    chunks = retrieve(user_question, top_k=5)

    # Format context with source info
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}] Title: {c['title']} | Category: {c['category']}\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful medical assistant for Altibbi.
Answer the user's question using ONLY the information provided in the context below.
If the answer is not in the context, say: "I don't have enough information to answer this question."
Do NOT use any outside knowledge.

Context:
{context}

User Question: {user_question}

Answer:"""

   
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks  
    }

# ── Usage ──────────────────────────────────────────────────────────────────
result = rag_query("ما هي أعراض السكري؟")  # Arabic works fine

print(result["answer"])
print("\n--- Sources Used ---")
for s in result["sources"]:
    print(f"  • {s['title']} (score: {s['score']:.3f})")