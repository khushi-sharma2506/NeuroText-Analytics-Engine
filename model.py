import os
# Force PyTorch and numpy to use a single thread to save massive amounts of RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from transformers import pipeline
import spacy
from pypdf import PdfReader

# Load the sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

# Load spacy model (will download if not present)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


import re

def predict_sentiment(text: str):
    # Overall sentiment (truncated safely by the pipeline)
    overall_result = sentiment_model(text)[0]

    # Split text strictly by . or \n as requested
    raw_sentences = re.split(r'[\.\n]+', text)
    sentences = [sent.strip() for sent in raw_sentences if len(sent.strip()) > 0]
    
    sentence_results = []
    for sent in sentences:
        pred = sentiment_model(sent)[0]
        sentence_results.append({
            "sentence": sent,
            "sentiment": pred["label"],
            "confidence": round(pred["score"], 4)
        })

    return {
        "overall": {
            "sentiment": overall_result["label"],
            "confidence": round(overall_result["score"], 4)
        },
        "sentences": sentence_results
    }


def get_pos_tags(text: str):
    doc = nlp(text)
    return [{"word": token.text, "pos": token.pos_, "tag": token.tag_, "explanation": spacy.explain(token.tag_)} for token in doc]


def get_tokens(text: str):
    doc = nlp(text)
    return [token.text for token in doc]


def remove_stopwords(text: str):
    doc = nlp(text)
    cleaned = [token.text for token in doc if not token.is_stop and not token.is_punct]
    stopwords_found = [token.text for token in doc if token.is_stop]
    return {
        "cleaned_text": " ".join(cleaned),
        "stopwords_removed": stopwords_found
    }
