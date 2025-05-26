from sentiment_analyzers.sentiment_transformer import analyze_sentiment_transformer

texts = [
    "I love this video!",
    "This is the worst thing I've ever seen.",
    "Meh, it was okay, nothing special."
]

for text in texts:
    label, scores = analyze_sentiment_transformer(text)
    print(f"Text: {text}\nSentiment: {label}, Scores: {scores}\n")
