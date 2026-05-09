import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel import Session, select
from database.models import Article
from database.database import engine


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.compile(r'[\u0610-\u061A\u064B-\u065F]') # tashkeel
    text = text.sub('', text) # removes the above
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا') # normalize
    return text.strip() # remove white space

def get_splitter() -> RecursiveCharacterTextSplitter:
    """
    It prefers splitting at:
    paragraphs \n\n, \n
    sentences (including Arabic punctuation like ؟ and ،)
    words " "
    characters (last fallback) ""
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "؟", "،", " ", ""],
        length_function=len, # split based on number of characters 
    ) 


# ── Main function (used by export_chunks.py) ────────────────────────────────
def get_all_chunks():
    """
    This function is the core data pipeline that converts database 
    articles into LangChain-ready chunks for embeddings.

    - Reads articles from your database (in batches)
    - Cleans the text (Arabic normalization)
    - Splits each article into chunks (using RecursiveCharacterTextSplitter)
    - Attaches metadata to each chunk
    - Returns a big list of LangChain documents

    """
    splitter      = get_splitter()
    langchain_docs = []
    skipped        = 0
    BATCH_SIZE     = 500 # process 500 articles at a time 

    with Session(engine) as session:
        total_count = len(session.exec(select(Article)).all())
        print(f"Total articles: {total_count}")

        offset = 0
        while offset < total_count:
            batch = session.exec(
                select(Article).offset(offset).limit(BATCH_SIZE)
            ).all()

            if not batch:
                break

            for article in batch:
                try:
                    title = article.title or ""
                    body  = article.body  or ""

                    if not body.strip():
                        skipped += 1
                        continue

                    full_text = f"{title}\n\n{body}"
                    cleaned   = clean_arabic(full_text)

                    # langchain
                    chunks = splitter.create_documents(
                        texts=[cleaned],
                        metadatas=[{
                            "article_id": article.article_id,
                            "title":      article.title,
                            "author":     article.author_name,
                            "category":   article.category,
                            "pub_date":   str(article.pub_date),
                        }]
                    )
                    langchain_docs.extend(chunks)

                except Exception as e:
                    print(f"  Error on article {article.article_id}: {e}")
                    skipped += 1
                    continue

            offset += BATCH_SIZE
            print(f"  Processed {min(offset, total_count)}/{total_count}"
                  f" → {len(langchain_docs)} chunks so far")

    print(f"\nDone.")
    print(f"  Articles processed : {total_count - skipped}")
    print(f"  Articles skipped   : {skipped}")
    print(f"  Total chunks       : {len(langchain_docs)}")
    print(f"  Avg chunks/article : {len(langchain_docs) / max(total_count - skipped, 1):.1f}")

    oversized = [d for d in langchain_docs if len(d.page_content) > 400]
    empty     = [d for d in langchain_docs if not d.page_content.strip()]
    print(f"  Oversized chunks   : {len(oversized)}")
    print(f"  Empty chunks       : {len(empty)}")

    return langchain_docs


# ── Run directly for testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    docs = get_all_chunks()
    if docs:
        print(f"\n--- First chunk preview ---")
        print(docs[0].page_content[:300])
        print(docs[0].metadata)