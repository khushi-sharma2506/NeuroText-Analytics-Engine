from fastapi import FastAPI
from pydantic import BaseModel

from model import predict_sentiment, get_pos_tags, get_tokens, remove_stopwords, get_lemmas, get_ner

app = FastAPI(
    title="NLP AI Toolkit API",
    description="API for various NLP tasks including sentiment analysis, POS tagging, and preprocessing",
    version="1.1.0"
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "NLP AI Toolkit API is running"
    }


@app.post("/predict")
def predict(request: TextRequest):
    return predict_sentiment(request.text)


@app.post("/pos")
def pos_tagging(request: TextRequest):
    return get_pos_tags(request.text)


@app.post("/tokens")
def tokenize(request: TextRequest):
    return {"tokens": get_tokens(request.text)}


@app.post("/lemmatization")
def lemmatize(request: TextRequest):
    return get_lemmas(request.text)

@app.post("/ner")
def named_entities(request: TextRequest):
    return get_ner(request.text)

@app.post("/stopwords")
def stopwords(request: TextRequest):
    return remove_stopwords(request.text)
