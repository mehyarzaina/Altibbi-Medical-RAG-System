import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from database.database import engine  
from database.models import Article                     
from sqlmodel import Session



# --------------------------
# Configuration
# --------------------------
BASE_URL = "https://altibbi.com"
LIST_URL = "https://altibbi.com/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA-%D8%B7%D8%A8%D9%8A%D8%A9"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --------------------------
# Helpers
# --------------------------
def get_article_id(article_url):    #get article_id from URL 
    match = re.search(r'-(\d+)$', article_url)
    return int(match.group(1)) if match else None


def parse_arabic_date(date_text):

    try:

        # remove punctuation and extra text
        date_text = re.sub(r"[^\d\s\w]", "", date_text)

        # Map Arabic month names to month numbers
        months = {
            'يناير': 1,
            'فبراير': 2,
            'مارس': 3,
            'أبريل': 4,
            'مايو': 5,
            'يونيو': 6,
            'يوليو': 7,
            'أغسطس': 8,
            'سبتمبر': 9,
            'أكتوبر': 10,
            'نوفمبر': 11,
            'ديسمبر': 12
        }

        # Extract day, month, year
        match = re.search(r"(\d+)\s+(\w+)\s+(\d+)", date_text)
        if match:
            day, month_name, year = match.groups()
            month = months.get(month_name)
            if month:
                return datetime(int(year), month, int(day))
        return None
    except Exception as e:
        print(f"Date parse error: {e}")
        return None

def get_article_details(article_url):
    try:
        response = requests.get(article_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "N/A"

        # Body
        body_div = soup.find('article', class_='article-body')
        body = body_div.get_text(separator="\n", strip=True) if body_div else ""

        # Author
        author_div = soup.find('div', class_='article-writter')
        author_tag = author_div.find('a') if author_div else None
        author_name = author_tag.get_text(strip=True) if author_tag else "Unknown"

        # Category
        category_tag = soup.select_one('.breadcrumb-item:nth-last-child(2)')
        category = category_tag.get_text(strip=True) if category_tag else "General"

        # -------- FIXED DATE EXTRACTION --------
        pub_date = None
        time_tag = soup.find('time', class_='article-date')
        if time_tag:
            date_text = time_tag.get_text(strip=True).replace("تاريخ النشر", "").strip()
            pub_date = parse_arabic_date(date_text)


        return {
            "article_id": get_article_id(article_url),
            "title": title,
            "body": body,
            "pub_date": pub_date,
            "author_name": author_name,
            "category": category
        }

    except Exception as e:
        print(f"Error scraping {article_url}: {e}")
        return None


def save_article_to_db(session, item: dict):
    """Save a single article dict to DB. Called per-article during scraping."""
    if isinstance(item["pub_date"], str):
        try:
            item["pub_date"] = datetime.fromisoformat(item["pub_date"])
        except ValueError:
            item["pub_date"] = datetime.now()

    article = Article(
        article_id=item["article_id"],
        title=item["title"],
        body=item["body"],
        pub_date=item["pub_date"],
        author_name=item["author_name"],
        category=item.get("category", "General")
    )
    session.add(article)
    # NOTE: commit is called by the caller (_scrape_page) after this



