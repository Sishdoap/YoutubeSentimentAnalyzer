import streamlit as st
from youtube import extract_video_id, get_comments, get_comments_with_spam_check, get_video_metadata
from collections import Counter
import matplotlib.pyplot as plt
from sentiment_analyzers.sentiment_hybrid import analyze_sentiment_hybrid
from wordcloud import WordCloud, STOPWORDS
import requests
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Testing mode

# --- CACHE WRAPPERS ---

@st.cache_data(show_spinner="Fetching video metadata...")
def cached_get_video_metadata(video_id):
    return get_video_metadata(video_id)

@st.cache_data(show_spinner="Fetching comments...")
def cached_get_comments(video_id, max_results):
    return get_comments(video_id, max_results)

@st.cache_data(show_spinner="Fetching spam-filtered comments...")
def cached_get_comments_with_spam_check(video_id, max_results):
    return get_comments_with_spam_check(video_id, max_results)

@st.cache_data(show_spinner="Analyzing sentiments...")
def cached_analyze_sentiments(comments):
    return [analyze_sentiment_hybrid(comment)[0] for comment in comments]

# --- LLM SUMMARY ---

@st.cache_data(show_spinner="Generating summary using LLM...")
def cached_summarize_with_llm(comments):
    prompt = (
        "You are an expert at analyzing YouTube comment sections.\n"
        "Here are comments extracted from a video:\n\n"
        + "\n".join(f"- {c}" for c in comments[:100]) +
        "\n\nBased on these comments, provide a summary of the general sentiment, the kinds of opinions expressed, and any recurring themes or arguments. Be concise but insightful."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "streamlit-youtube-sentiment-app"
    }

    body = {
        "model": "qwen/qwen3-14b:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ LLM request failed: {response.status_code} - {response.text}"

# --- CSS ---

st.markdown("""
<style>
    .comment-block {
        background-color: #f0f2f6;
        padding: 10px;
        margin-bottom: 8px;
        border-left: 4px solid #4A90E2;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📺 YouTube Comment Sentiment Analyzer")

with st.sidebar:
    st.header("🔧 Settings")
    url = st.text_input("Enter YouTube video URL:")
    mode = st.radio("Comment Retrieval Mode:", ["Normal", "Spam-filtered"])
    max_comments = st.slider("Max comments to fetch", 20, 500, 200, 20)

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL!")
    else:
        metadata = cached_get_video_metadata(video_id)
        if metadata:
            st.image(metadata["thumbnail_url"], width=400)
            st.markdown(f"### {metadata['title']}")
            st.markdown(f"Channel: **{metadata['channel']}**")
        else:
            st.warning("Could not retrieve video metadata.")

        st.info("Fetching comments...")

        if mode == "Normal":
            comments = cached_get_comments(video_id, max_comments)
            st.success(f"Fetched {len(comments)} comments (no spam filtering).")
        else:
            comments_with_labels = cached_get_comments_with_spam_check(video_id, max_comments)
            total_comments = len(comments_with_labels)
            comments = [c["comment"] for c in comments_with_labels if c["label"] == "NOT SPAM"]
            spam_count = total_comments - len(comments)
            st.success(f"Fetched {total_comments} comments, filtered out {spam_count} spam comments.")

        if not comments:
            st.warning("No comments found or all comments filtered out.")
        else:
            sentiments = cached_analyze_sentiments(comments)
            counts = Counter(sentiments)
            total = sum(counts.values())

            tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "☁️ Word Cloud", "🧠 LLM Summary"])

            with tab1:
                st.write("### Sentiment Breakdown")
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", f"{counts.get('positive', 0)} ({counts.get('positive', 0) / total * 100:.1f}%)")
                col2.metric("Negative", f"{counts.get('negative', 0)} ({counts.get('negative', 0) / total * 100:.1f}%)")
                col3.metric("Neutral", f"{counts.get('neutral', 0)} ({counts.get('neutral', 0) / total * 100:.1f}%)")

                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [counts.get('positive', 0), counts.get('negative', 0), counts.get('neutral', 0)]

                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c', '#95a5a6'])
                ax.axis('equal')
                st.pyplot(fig)

                st.write("### Sample Comments by Sentiment")
                sentiment_filter = st.selectbox("Show comments for: ", ["All", "positive", "negative", "neutral"])
                for label in ['positive', 'negative', 'neutral']:
                    if sentiment_filter == "All" or sentiment_filter == label:
                        st.write(f"**{label.capitalize()} comments:**")
                        filtered = [c for c, s in zip(comments, sentiments) if s == label]
                        for c in filtered[:3]:
                            st.markdown(
                                f"""
                                    <div style='
                                        background-color: rgba(255, 255, 255, 0.05);
                                        padding: 10px 15px;
                                        margin-bottom: 10px;
                                        border-radius: 8px;
                                        color: var(--text-color);
                                    '>
                                        {c}
                                    </div>
                                    """,
                                unsafe_allow_html=True
                            )

            with tab2:
                st.write("### Word Cloud of Comments")
                all_text = " ".join(comments).lower()
                custom_stopwords = set(STOPWORDS).union({"https", "www", "video", "youtube"})
                wordcloud = WordCloud(width=800, height=400,
                                      stopwords=custom_stopwords,
                                      background_color='white',
                                      colormap='viridis').generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(20, 10))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)

            with tab3:
                with st.spinner("Generating summary using LLM..."):
                    llm_summary = cached_summarize_with_llm(comments)
                    st.subheader("💡 LLM-Generated Summary of Sentiment")
                    st.markdown(llm_summary)

