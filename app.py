# Steps-
# pip install -r requirements.txt
# python -m uvicorn app:app --reload
# python gradio.py



from fastapi import FastAPI
from pydantic import BaseModel

from model import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment of text",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running"
    }


@app.post("/predict")
def predict(request: TextRequest):
    return predict_sentiment(request.text)
