from .helper import save_article_to_db
from .scraping import scrape_all
from database.database import create_db, engine
from sqlmodel import Session, select
from database.models import Article

if __name__ == "__main__":
    create_db()

    with Session(engine) as session:
        existing_ids = set(session.exec(select(Article.article_id)).all()) # fetch all article IDs from database and store in set
        print(f"Skipping {len(existing_ids)} already-scraped articles.")

        total = scrape_all(
            session=session,
            save_fn=save_article_to_db,
            delay=1.5,
            skip_ids=existing_ids
        )
        print(f"Done. {total} new articles saved.")