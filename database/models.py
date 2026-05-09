#create db using sqlModel
from typing import Optional
from datetime import datetime
from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import TEXT

# -------------------------
# Articles Table
# -------------------------
class Article(SQLModel, table=True):
    __tablename__ = "articles"

    article_id: Optional[int] = Field(default=None, primary_key=True)

    title: str = Field(max_length=1024)

    body: str = Field(sa_column=TEXT) 

    pub_date: datetime
    author_name: Optional[str] = Field(default=None, max_length=255)
    category: Optional[str] = Field(default=None, max_length=255, index=True)



