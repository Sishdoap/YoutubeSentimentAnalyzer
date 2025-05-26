from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(comment):
    score = analyzer.polarity_scores(comment)
    compound = score['compound']
    if compound > 0.05:
        label = 'positive'
    elif compound < -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return label, compound