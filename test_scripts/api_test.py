from youtube import extract_video_id, get_comments
from sentiment_analyzers.sentiment import analyze_sentiment

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_id = extract_video_id(url)
comments = get_comments(video_id, max_results=20)

pos, neg, neu = 0, 0, 0

for i, comment in enumerate(comments, 1):
    label, score = analyze_sentiment(comment)
    print(f"{i}. [{label.upper()}] {comment}")

    if label == 'positive':
        pos += 1
    elif label == 'negative':
        neg += 1
    else:
        neu += 1

total = pos + neg + neu
print("\n=== Sentiment Breakdown ===")
print(f"Positive: {pos} ({(pos / total) * 100:.1f}%)")
print(f"Negative: {neg} ({(neg / total) * 100:.1f}%)")
print(f"Neutral:  {neu} ({(neu / total) * 100:.1f}%)")