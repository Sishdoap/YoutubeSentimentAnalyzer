# sentiment_hybrid.py
from sentiment_analyzers.sentiment import analyze_sentiment
from sentiment_analyzers.sentiment_transformer import analyze_sentiment_transformer


def analyze_sentiment_hybrid(text):
    vader_label, vader_score = analyze_sentiment(text)

    print(f"[Hybrid] VADER: {vader_label} (compound={vader_score:.3f})")

    # If VADER is very confident, just use it
    if vader_score >= 0.7:
        return "positive", "VADER"
    elif vader_score <= -0.7:
        return "negative", "VADER"
    elif -0.4 < vader_score < 0.4:
        # Use transformer for uncertain cases
        transformer_label, _ = analyze_sentiment_transformer(text)
        print(f"[Hybrid] Transformer override: {transformer_label}")
        return transformer_label, "Transformer"
    else:
        # If in-between, stick to VADER
        return vader_label, "VADER"
