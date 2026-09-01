# Altibbi Medical RAG System

A Retrieval-Augmented Generation (RAG) system built on top of Arabic-language medical articles scraped from [Altibbi](https://altibbi.com/). Articles are stored in MySQL, chunked and embedded into a Zilliz (Milvus) vector database, and exposed through a FastAPI service — including MCP tools — so an LLM can answer medical questions grounded only in the scraped content.

## Pipeline overview

1. **Scrape** (`articles_scraping/`) — Crawls every page of the Altibbi articles listing, extracts title, body, author, category, URL, and Arabic publish date for each article, and saves new articles to MySQL as it goes (skipping ones already stored).
2. **Chunk** (`RAG/chunking.py`, `RAG/export_chunks.py`) — Pulls articles from MySQL in batches, normalizes the Arabic text (strips tashkeel/diacritics, normalizes alef variants), splits each article into ~400-character chunks with LangChain's `RecursiveCharacterTextSplitter`, and exports the result to `chunks.json`.
3. **Embed & index** (`RAG/embedding.ipynb`) — Loads `chunks.json`, calls an embedding API for each chunk, and upserts the resulting vectors (plus metadata: title, author, category, url, pub_date) into a Zilliz Cloud collection with a cosine-similarity index.
4. **Retrieve & answer** (`RAG/retrival.py`) — Embeds a user's question, searches Zilliz for the top-k most similar chunks, builds a context-grounded prompt, and calls Groq's `llama-3.3-70b-versatile` to answer using only the retrieved context.
5. **Serve** (`RAG/app.py`) — A FastAPI app exposing `/embed` and `/search` HTTP endpoints, plus an MCP server (mounted at `/mcp`) with three tools — `search_medical_articles`, `embed_text`, and `search_articles` (structured MySQL lookups by title/author/category/id) — so an LLM client can query the system directly. Requests are protected by a static bearer-token / `X-API-Key` middleware.

## Project structure

```
Altibbi-Medical-RAG-System/
├── articles_scraping/
│   ├── main.py            # Entry point: scrapes all pages, saves new articles to MySQL
│   ├── scraping.py         # Pagination, per-page scraping, resume/skip logic
│   └── helper.py           # Article detail parsing, Arabic date parsing, DB writes
├── database/
│   ├── database.py         # SQLModel engine setup (MySQL, via .env)
│   └── models.py           # Article table definition
├── RAG/
│   ├── chunking.py         # DB → cleaned, chunked LangChain documents
│   ├── export_chunks.py    # Runs chunking.py and writes chunks.json
│   ├── embedding.ipynb     # Embeds chunks.json and loads them into Zilliz
│   ├── retrival.py         # Standalone query → retrieve → Groq-answer script
│   └── app.py               # FastAPI + MCP server: /embed, /search, MCP tools
├── requirements.txt
└── .gitignore
```

## Data model

**MySQL — `articles` table**
- `article_id` (PK, from the article URL), `title`, `body`, `pub_date`, `author_name`, `category`, `url`

**Zilliz/Milvus collection** (defined in `embedding.ipynb`)
- `id` (auto), `article_id`, `chunk_index`, `text`, `title`, `author`, `category`, `pub_date`, `url`, `embedding` (768-dim float vector, cosine index)

## Tech stack

- **Python** — FastAPI, Uvicorn
- **Requests** + **BeautifulSoup4** — scraping
- **SQLModel** / **SQLAlchemy** + **PyMySQL** — MySQL storage
- **LangChain** (`langchain_text_splitters`) — Arabic-aware text chunking
- **PyMilvus** + **Zilliz Cloud** — vector storage and similarity search
- **Groq** (`llama-3.3-70b-versatile`) — answer generation over retrieved context
- **MCP** (`mcp.server.fastmcp`) — exposes search tools to MCP-compatible LLM clients
- **python-dotenv** — environment configuration

> **Note on `requirements.txt`:** it currently doesn't list `fastapi`, `uvicorn`, `langchain-text-splitters`, `pymilvus`, `groq`, or `mcp`, even though `RAG/app.py` and `RAG/retrival.py` import them. You'll need to install these separately (see Setup below). It also lists both `psycopg[binary]` (PostgreSQL) and `mysql-connector-python` even though the code connects via `pymysql`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mehyarzaina/Altibbi-Medical-RAG-System.git
cd Altibbi-Medical-RAG-System
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn langchain-text-splitters pymilvus groq "mcp[cli]" jupyter
```

### 3. Configure environment variables

Create a `.env` file in the project root (already excluded via `.gitignore`):

```
# MySQL
DB_USER=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_NAME=your_db_name

# Embedding API
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_URL=your_embedding_api_url

# Zilliz Cloud
ZILLIZ_URI=your_zilliz_cluster_uri
ZILLIZ_TOKEN=your_zilliz_token
COLLECTION_NAME=your_collection_name

# Groq
GROQ_API_KEY=your_groq_api_key

# API auth for RAG/app.py (optional but recommended)
API_TOKEN=a_shared_secret_for_your_api
```

### 4. Run the pipeline, in order

**a. Scrape articles into MySQL:**
```bash
python -m articles_scraping.main
```

**b. Export chunked text from MySQL:**
```bash
python -m RAG.export_chunks
```
This produces `chunks.json` in the project root.

**c. Embed chunks and load them into Zilliz:**
Open `RAG/embedding.ipynb` (locally with Jupyter, or in Google Colab) and run all cells. It creates the Zilliz collection, embeds `chunks.json` in batches, and upserts the vectors.

**d. Query directly (script):**
```bash
python -m RAG.retrival
```
Runs a sample Arabic question end-to-end (embed → search Zilliz → answer with Groq) and prints the answer plus sources.

**e. Serve the API + MCP tools:**
```bash
python RAG/app.py
```
Starts a FastAPI server on `http://0.0.0.0:8000` with:
- `POST /embed` — get an embedding vector for arbitrary text
- `POST /search` — semantic search over the Zilliz collection
- `/mcp` — MCP server exposing `search_medical_articles`, `embed_text`, and `search_articles` as tools for MCP-compatible clients

If `API_TOKEN` is set, non-local requests must include `Authorization: Bearer <token>` or `X-API-Key: <token>`.


** How to run this code ** 

python -m RAG.app.py

## Notes

- The scraper and chunker both normalize/clean Arabic text (removing tashkeel, unifying alef forms) since the source content is entirely Arabic.
- `search_articles` (an MCP tool in `app.py`) queries MySQL directly for structured lookups (by title/author/category/id), while `search_medical_articles` and `/search` perform semantic vector search over Zilliz — the two are complementary retrieval paths.
- The old top-level `README.md` only documents `python -m RAG.app.py` (note the typo — it should be `python -m RAG.app` or `python RAG/app.py`); this README fills in the rest of the pipeline.

