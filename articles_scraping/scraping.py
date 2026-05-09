# imports
import requests
from bs4 import BeautifulSoup
import time
import re
from .helper import BASE_URL, LIST_URL, HEADERS, get_article_id, get_article_details
from typing import Union, List


def _scrape_page(page_num: int, session, save_fn, delay: float, skip_ids: set) -> int:
    """
    Scrape a single listing page, save each article immediately via save_fn.
    Returns the count of articles saved.
    """
    print(f"--- Scraping Page {page_num} ---")
    response = requests.get(LIST_URL, params={"page": page_num}, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.select("article.article-item-container")

    article_links = []
    for article in articles:
        link_tag = article.find("a")
        if link_tag and link_tag.get("href"):
            href = link_tag.get("href")
            if not href.startswith("http"):
                href = BASE_URL + href
            article_links.append(href)

    print(f"Found {len(article_links)} article links on page {page_num}")

    saved_count = 0
    seen_ids = set()

    for href in article_links:
        article_id = get_article_id(href)

        if article_id in seen_ids:
            continue
        seen_ids.add(article_id)

        if article_id in skip_ids:
            print(f"  Skipping already-saved article {article_id}")
            continue

        details = get_article_details(href)
        if details:
            try:
                save_fn(session, details)       # save immediately
                session.commit()
                saved_count += 1
                print(f"  Saved article {article_id}")
            except Exception as e:
                session.rollback()
                print(f"  Failed to save article {article_id}: {e}")

        time.sleep(delay)

    return saved_count


def scrape(
    target: Union[int, List[int]],
    session,
    save_fn,
    delay: float = 1,
    skip_ids=None,
) -> int:
    """
    Scrape one page (int) or a list of pages.

    Args:
        target:   A single page number or a list of page numbers.
        session:  SQLAlchemy DB session.
        save_fn:  Callable(session, article_dict) that inserts one article.
        delay:    Seconds to wait between article requests.
        skip_ids: Set of article IDs already in the DB (to avoid duplicates).

    Returns:
        Total number of articles saved.
    """
    skip_ids = skip_ids or set()
    pages = [target] if isinstance(target, int) else target

    total_saved = 0
    for page_num in pages:
        total_saved += _scrape_page(page_num, session, save_fn, delay, skip_ids)

    print(f"Done. Total articles saved: {total_saved}")
    return total_saved


def get_total_pages() -> int:
    """Fetch the listing page and extract the total number of pagination pages."""
    response = requests.get(LIST_URL, params={"page": 1}, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")

    pagination_links = soup.select("ul.pagination a, .pagination a, a[href*='page=']")

    max_page = 1
    for link in pagination_links:
        href = link.get("href", "")
        match = re.search(r"page=(\d+)", href)
        if match:
            max_page = max(max_page, int(match.group(1)))

    if max_page == 1:
        for link in pagination_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                max_page = max(max_page, int(text))

    print(f"Total pages found: {max_page}")
    return max_page


def scrape_all(session, save_fn, delay: float = 1, skip_ids=None) -> int:
    """Scrape every page automatically, saving each article as it's fetched."""
    total_pages = get_total_pages()
    all_pages = list(range(1, total_pages + 1))
    return scrape(target=all_pages, session=session, save_fn=save_fn, delay=delay, skip_ids=skip_ids)