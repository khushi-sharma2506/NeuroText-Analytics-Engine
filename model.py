from transformers import pipeline

# Load the sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def predict_sentiment(text: str):
    result = sentiment_model(text)[0]

    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }
