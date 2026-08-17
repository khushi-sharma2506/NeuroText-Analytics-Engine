import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text."

    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        return (
            f"Sentiment: {result['sentiment']}\n"
            f"Confidence: {result['confidence']:.2%}"
        )

    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"


demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=5,
        label="Enter text",
        placeholder="Type a review or sentence..."
    ),
    outputs=gr.Textbox(
        label="Sentiment Result"
    ),
    title="🎭 Sentiment Analysis",
    description="Analyze the sentiment of your text using an AI model."
)

demo.launch()
