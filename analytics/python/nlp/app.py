from fastapi import FastAPI
from pydantic import BaseModel

from nlp.nlp import load_news, predict_sentiment

from nlp.nlp import SentimentRequest

# python -m uvicorn app:app --port 8002
# localhost:8002/sentiment
app = FastAPI()

@app.post("/sentiment")
def sentiment(req: SentimentRequest):
    df = load_news(req.fromDate, req.toDate, req.limit)
    df = predict_sentiment(df)
    return df.to_dict(orient="records")  # returns json