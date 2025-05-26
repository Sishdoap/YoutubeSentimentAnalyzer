from sentiment_analyzers.sentiment_hybrid import analyze_sentiment_hybrid

comments = [
    "I absolutely loved this video! ❤️",
    "This was okay I guess...",
    "Terrible content, not worth watching.",
    "Meh, nothing special.",
    "W video 🔥",
    "I didn't like it much, but it had some good points."
]

for comment in comments:
    label, model_used = analyze_sentiment_hybrid(comment)
    print(f"[{model_used}] '{comment}' → Sentiment: {label}\n")
