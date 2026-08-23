import gradio as gr
import requests
import os
from model import extract_text_from_pdf

API_PORT = os.getenv("PORT", "8000")
API_URL = os.getenv("API_URL", f"http://127.0.0.1:{API_PORT}")

def handle_pdf_upload(file_obj):
    if file_obj is None:
        return ""
    return extract_text_from_pdf(file_obj.name)

def analyze_sentiment(text):
    if not text.strip(): return "Please enter some text.", ""
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        overall = result["overall"]
        labels_out = f"Overall Sentiment: {overall['sentiment'].lower()}\n\n"
        scores_out = f"Overall Confidence: {overall['confidence']:.2%}\n\n"
        
        for i, s in enumerate(result.get("sentences", []), 1):
            labels_out += f"{i}. {s['sentence']}\nLabel: {s['sentiment'].lower()}\n\n"
            scores_out += f"{i}. {s['confidence']:.2%}\n\n"
            
        return labels_out.strip(), scores_out.strip()
    except Exception as e: return f"API Error: {e}", ""

def get_pos(text):
    if not text.strip(): return "Please enter some text."
    try:
        response = requests.post(f"{API_URL}/pos", json={"text": text}, timeout=30)
        response.raise_for_status()
        tags = response.json()
        return "\n".join([f"{t['word']} -> {t['pos']} ({t['explanation']})" for t in tags])
    except Exception as e: return f"API Error: {e}"

def get_tokens(text):
    if not text.strip(): return "Please enter some text."
    try:
        response = requests.post(f"{API_URL}/tokens", json={"text": text}, timeout=30)
        response.raise_for_status()
        tokens = response.json().get("tokens", [])
        return f"Total Tokens: {len(tokens)}\n\n" + " | ".join(tokens)
    except Exception as e: return f"API Error: {e}"

def remove_stopwords(text):
    if not text.strip(): return "Please enter some text."
    try:
        response = requests.post(f"{API_URL}/stopwords", json={"text": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        removed = data['stopwords_removed']
        return f"Cleaned Text:\n{data['cleaned_text']}\n\nTotal Stopwords Removed: {len(removed)}\nRemoved Stopwords: {', '.join(removed)}"
    except Exception as e: return f"API Error: {e}"

with gr.Blocks(title="NeuroText Analytics Engine") as demo:
    gr.Markdown("# NeuroText Analytics Engine")
    gr.Markdown("A unified intelligence platform for analyzing, processing, and understanding text data.")
    
    with gr.Row():
        pdf_input = gr.File(label="Optional: Upload PDF to extract text", file_types=[".pdf"])
    
    text_input = gr.Textbox(lines=5, label="Input Text", placeholder="Type here or upload a PDF above...")
    
    pdf_input.change(fn=handle_pdf_upload, inputs=pdf_input, outputs=text_input)

    gr.Examples(
        examples=[
            ["I absolutely loved this movie."],
            ["This was the worst experience of my life."],
            ["The food was delicious. The service was extremely slow."],
            ["This product is not bad. I would happily purchase it again."],
            ["Hello! I am so incredibly excited to announce that our brand-new AI Toolkit was successfully launched in New York City on Monday.\nIt is a fantastic application that can analyze sentiments, identify parts of speech, and extract valuable tokens from PDF documents.\nHowever, I must admit that the initial setup was quite frustrating and a bit difficult to navigate.\nDespite the early challenges, the final product is definitely worth the effort.\nDo you agree?"]
        ],
        inputs=text_input,
        label="Try these examples (Sentences & Paragraphs)"
    )

    with gr.Tabs():
        with gr.TabItem("🎭 Sentiment Analysis"):
            sentiment_btn = gr.Button("Analyze Sentiment")
            with gr.Row():
                sentiment_labels_out = gr.Textbox(label="Sentiment Labels", lines=10)
                sentiment_scores_out = gr.Textbox(label="Confidence Scores", lines=10)
            sentiment_btn.click(analyze_sentiment, inputs=text_input, outputs=[sentiment_labels_out, sentiment_scores_out])
            
        with gr.TabItem("🏷️ POS Tagging"):
            pos_btn = gr.Button("Extract POS Tags")
            pos_out = gr.Textbox(label="Result", lines=10)
            pos_btn.click(get_pos, inputs=text_input, outputs=pos_out)
            
        with gr.TabItem("🔠 Tokens"):
            token_btn = gr.Button("Tokenize Text")
            token_out = gr.Textbox(label="Result", lines=5)
            token_btn.click(get_tokens, inputs=text_input, outputs=token_out)
            
        with gr.TabItem("🛑 Stopwords"):
            stop_btn = gr.Button("Remove Stopwords")
            stop_out = gr.Textbox(label="Result", lines=5)
            stop_btn.click(remove_stopwords, inputs=text_input, outputs=stop_out)

    gr.Markdown("""
    <br><br>
    **Developed by Khushi Sharma**<br>
    B.Tech CSE (AI & ML), Graphic Era Deemed to be University
    """)

if __name__ == "__main__":
    demo.launch()
