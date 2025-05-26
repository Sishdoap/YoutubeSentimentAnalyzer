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
- The app uses a beta version of PyTorch, version 2.8.0 with CUDA 12.8 support. Within the requirements, I have listed it as PyTorch version 2.7.0. This should work for the majority of machines. If you have any issues, try installing PyTorch via pip with this: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set up API keys in .env

The app depends on 2 API keys, OPENROUTER_API_KEY and YOUTUBE_API_KEY. You need to make your own API keys, they are free and can be done here:
- Youtube API V3: https://developers.google.com/youtube/v3/getting-started
- OpenRouter API: https://openrouter.ai/docs/api-reference/authentication
