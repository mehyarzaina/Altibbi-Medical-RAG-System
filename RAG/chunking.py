"""
Takes Arabic articles from a database → cleans the text 
→ splits them into small chunks → attaches metadata 
→ returns them ready for embeddings
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel import Session, select
from database.models import Article
from database.database import engine


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_arabic(text: str) -> str:
    """
    clean arabic text
    1 - remove taskel
    2- Normailze أ 
    3- remove white spaces
    """
    if not text:
        return ""
    pattern = re.compile(r'[\u0610-\u061A\u064B-\u065F]') # tashkeel
    text = pattern.sub('', text) # removes the above
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا') # normalize
    return text.strip() # remove spaces at start and end

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
        length_function=len, # measures chunk size by number of characters
    ) 


# ── Main function (used by export_chunks.py) ────────────────────────────────
def get_all_chunks():
    """
    This function converts database 
    articles into LangChain-ready chunks for embeddings.

    - Reads articles from database (in batches)
    - Cleans the text (Arabic normalization)
    - Splits each article into chunks (using RecursiveCharacterTextSplitter)
    - Attaches metadata to each chunk
    - Returns a big list of LangChain documents

    """
    splitter      = get_splitter() # creates object
    langchain_docs = []
    skipped        = 0
    BATCH_SIZE     = 500 # process 500 articles at a time 

    with Session(engine) as session:
        total_count = len(session.exec(select(Article)).all()) # counts all articles in DB
        print(f"Total articles: {total_count}")

        # loop through DB in batches 
        start = 0
        while start < total_count:
            # fetching BATCH_SIZE at once not all DB 
            batch = session.exec(
                select(Article).offset(start).limit(BATCH_SIZE)
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

                    full_text = f"{title}\n\n{body}" # joins title + body 
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
                            "url":        article.url or "",  
                        }]
                    )
                    langchain_docs.extend(chunks)

                except Exception as e:
                    print(f"  Error on article {article.article_id}: {e}")
                    skipped += 1
                    continue

            start += BATCH_SIZE
            print(f"  Processed {min(start, total_count)}/{total_count}"
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
# if __name__ == "__main__":
#     docs = get_all_chunks()
#     if docs:
#         print(f"\n--- First chunk preview ---")
#         print(docs[0].page_content[:300])
#         print(docs[0].metadata)