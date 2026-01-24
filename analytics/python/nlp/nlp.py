from transformers import pipeline
import psycopg2
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class SentimentRequest(BaseModel):
    fromDate: Optional[str] = None
    toDate: Optional[str] = None
    limit: Optional[int] = 30

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    truncation=True
)

def load_news(from_date=None, to_date=None, limit=30):
    conn = psycopg2.connect(
        host="postgres",
        database="analytics",
        user="postgres",
        password="admin",
        port=5432
    )

    if from_date is None or from_date == "":
        from_date = None
    if to_date is None or to_date == "":
        to_date = None

    query = """
            SELECT \"date\", content, link 
            FROM news
            WHERE (%s IS NULL OR \"date\" >= %s) AND 
                (%s IS NULL OR \"date\" <= %s)
            ORDER BY \"date\" DESC
            LIMIT %s
            """
    params = (from_date, from_date, to_date, to_date, limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def predict_sentiment(df: pd.DataFrame):

    results = classifier(
        df["content"].tolist(),
        batch_size=16,
        truncation=True
    )

    df["sentiment_label"] = [row["label"] for row in results]
    df["sentiment_score"] = [row["score"] for row in results]

    return df