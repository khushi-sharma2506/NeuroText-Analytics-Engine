🧠 NLP-AI-Toolkit

A modular Natural Language Processing (NLP) and Artificial Intelligence toolkit built with Python, FastAPI, Hugging Face Transformers, and Gradio.

The project is designed to bring multiple AI-powered text processing capabilities into a single application. The first implemented module is Sentiment Analysis, with additional NLP features planned for future development.

🚀 Current Features
🎭 Sentiment Analysis

The current application analyzes text and classifies its sentiment into three categories:

🟢 Positive
🟡 Neutral
🔴 Negative

The application also provides confidence scores for the predicted sentiment.

🔌 FastAPI Backend

The project provides a REST API for sentiment analysis.

Endpoint:

POST /predict


Example request:

{
  "text": "I really love this product!"
}


Example response:

{
  "sentiment": "Positive",
  "confidence": 0.98,
  "scores": {
    "Positive": 0.98,
    "Neutral": 0.01,
    "Negative": 0.01
  }
}

🎨 Gradio Interface

The project includes a Gradio-based web interface where users can:

Enter text
Analyze sentiment
View the predicted sentiment
View confidence scores
Compare Positive, Neutral, and Negative scores
🏗️ Application Architecture
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │   Gradio UI     │
                  │     ui.py       │
                  └────────┬────────┘
                           │
                           │ HTTP POST
                           │ /predict
                           ▼
                  ┌─────────────────┐
                  │    FastAPI      │
                  │     app.py      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Sentiment Model │
                  │    model.py     │
                  └────────┬────────┘
                           │
                           ▼
                 Hugging Face Model
                           │
                           ▼
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Positive      Neutral      Negative

📁 Project Structure
NLP-AI-Toolkit/
│
├── app.py
├── model.py
├── ui.py
├── requirements.txt
└── README.md

File Description
File	Description
app.py	FastAPI backend and API endpoints
model.py	Sentiment analysis model and prediction logic
ui.py	Gradio-based user interface
requirements.txt	Python dependencies
README.md	Project documentation
🛠️ Technologies Used
Python – Core programming language
FastAPI – REST API backend
Uvicorn – ASGI server
Hugging Face Transformers – NLP and Transformer models
PyTorch – Machine learning framework
Gradio – Interactive web interface
Requests – Communication between frontend and API
🤖 NLP Model

The sentiment-analysis module uses a pretrained Transformer model from Hugging Face:

cardiffnlp/twitter-roberta-base-sentiment-latest


The model is used to classify text into:

Positive
Neutral
Negative


The model returns probability scores, which are displayed by the application as confidence scores.

⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/NLP-AI-Toolkit.git


Replace YOUR-USERNAME with your GitHub username.

2. Open the project directory
cd NLP-AI-Toolkit

3. Install dependencies
python -m pip install -r requirements.txt


If you don't have the requirements file configured yet, you can install the main dependencies with:

python -m pip install fastapi "uvicorn[standard]" transformers torch gradio requests

▶️ Running the Application

The project contains two components:

FastAPI backend
Gradio frontend

Both need to be running at the same time.

Step 1 — Start FastAPI

Open a terminal in the project directory:

python -m uvicorn app:app --reload


The API will run at:

http://127.0.0.1:8000

Step 2 — Open FastAPI Documentation

FastAPI provides interactive API documentation:

http://127.0.0.1:8000/docs


From the documentation page, you can test the /predict endpoint directly.

Step 3 — Start Gradio

Open a second terminal:

python ui.py


The Gradio interface will normally be available at:

http://127.0.0.1:7860


Open this address in your browser.

🧪 Example

Enter a sentence such as:

I absolutely love this application! It is amazing.


The model analyzes the text and returns a result similar to:

Positive

Confidence: 98%


The application also displays the confidence scores for:

Positive
Neutral
Negative

🔌 API Usage

The sentiment-analysis API can be accessed using a POST request.

Endpoint
POST /predict

Request
{
  "text": "The product is excellent and I really enjoyed using it."
}

Response
{
  "sentiment": "Positive",
  "confidence": 0.98,
  "scores": {
    "Positive": 0.98,
    "Neutral": 0.01,
    "Negative": 0.01
  }
}


This API architecture allows other applications to use the sentiment-analysis model without directly interacting with the model code.

🔮 Planned Features

The project is designed to grow into a broader NLP toolkit.

Planned features include:

 Sentiment Analysis
 Text Translation
 Part-of-Speech (POS) Tagging
 Named Entity Recognition (NER)
 Text Summarization
 Text Classification
 Keyword Extraction
 Language Detection
 Text Preprocessing
 Batch Text Analysis
 CSV Upload and Analysis
 Sentiment Visualization
 Analytics Dashboard
 Multiple NLP Models
 Additional FastAPI endpoints
🎯 Project Goal

The main goal of NLP-AI-Toolkit is to develop a modular platform that brings different Natural Language Processing capabilities together in one application.

The project starts with sentiment analysis and will gradually expand into a collection of NLP tools that can be accessed through:

🌐 Web interfaces
🔌 REST APIs
🤖 Machine Learning models
📊 Data visualization tools

The architecture is designed so that new NLP features can be added without significantly changing the existing modules.

📈 Future Architecture

As more features are added, the project can evolve toward a modular structure such as:

NLP-AI-Toolkit/
│
├── backend/
│   ├── app.py
│   │
│   ├── routes/
│   │   ├── sentiment.py
│   │   ├── translation.py
│   │   ├── pos.py
│   │   └── ner.py
│   │
│   └── models/
│       ├── sentiment.py
│       ├── translation.py
│       ├── pos.py
│       └── ner.py
│
├── frontend/
│   └── ui.py
│
├── tests/
│
├── requirements.txt
└── README.md


This structure will make it easier to maintain and expand the toolkit as new NLP capabilities are introduced.

🔐 Security

Do not upload sensitive information to this repository.

Never commit:

API keys
Passwords
Access tokens
Hugging Face tokens
Secret credentials
.env files containing secrets


If API keys are added in future versions, they should be stored using environment variables.

📌 Project Status

Current Version: v1.0 – Sentiment Analysis

Current Status

🟢 Working

The current version includes:

FastAPI backend
Hugging Face Transformer sentiment model
Three-class sentiment classification
Confidence scores
Gradio web interface
Interactive API documentation

Future versions will introduce additional NLP capabilities.

👨‍💻 Developed By
Khushi Sharma

Graphic Era Deemed to be University

This project is developed as part of an academic and practical exploration of:

Natural Language Processing
Machine Learning
Artificial Intelligence
API Development
AI-powered Applications
🌟 Future Vision

NLP-AI-Toolkit aims to become a unified platform for experimenting with and deploying different NLP capabilities.

The long-term vision is to provide multiple text intelligence tools through a single, easy-to-use application and API.

📄 License

License information will be added in a future version.
