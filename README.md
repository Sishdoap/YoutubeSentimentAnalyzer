# YouTube Comment Sentiment Analyzer

A web app that extracts comments from YouTube videos, detects spam, analyzes sentiment, generates summaries using LLMs, and visualizes data with tools like word clouds and time-based charts.

## 🚀 Features

- 🔍 Extracts comments from YouTube videos
- 🧠 Sentiment analysis with fine-tuned transformer models
- 🛡️ Spam comment detection with locally trained transformer models
- 📈 Sentiment over time visualization
- ☁️ Word cloud generation
- ✨ LLM-generated summary of comments (via OpenRouter)
- 🎨 Streamlit web interface

## 🛠️ Tech Stack

- Python
- Streamlit
- Transformers (Hugging Face)
- YouTube Data API v3
- OpenRouter API (for LLM summaries)
- Scikit-learn, pandas, matplotlib, wordcloud

## Notes

- The locally trained transformer model is fetched from a HuggingFace repository here: https://huggingface.co/Sishdoap/spam-detector-transformer/tree/main

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

