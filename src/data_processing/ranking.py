import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from datetime import datetime
from scipy.special import softmax
from src.models.sentiment_trainer import predict_sentiment

def compute_sentiment_strength(sentiments: list) -> float:
    """
    Compute sentiment strength using weighted polarity and confidence scores.
    """
    sentiment_weights = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    sentiment_scores = [
        s["confidence"] * sentiment_weights.get(s["label"], 0.0) for s in sentiments
    ]
    return np.mean(sentiment_scores) if sentiment_scores else 0.0

def compute_recency_factor(timestamps: list) -> float:
    """
    Apply exponential decay to weigh recent articles more heavily.
    """
    now = datetime.utcnow()
    decay_rate = 0.1  # Adjust decay sensitivity
    recency_weights = np.exp(-decay_rate * np.array([(now - t).days for t in timestamps]))
    return np.sum(recency_weights) / len(timestamps) if timestamps else 0.0

def compute_market_impact(impact_factors: list) -> float:
    """
    Normalize market impact factors if provided, otherwise default to 1.0.
    """
    return np.mean(impact_factors) if impact_factors else 1.0

def compute_article_volume(sentiments: list) -> float:
    """
    Use a log-based scaling to incorporate the number of analyzed articles.
    """
    return np.log(len(sentiments) + 1)

def compute_source_credibility(sources: list) -> float:
    """
    Compute source credibility based on a predefined credibility score.
    """
    credibility_scores = {
        "Reuters": 1.0,
        "Bloomberg": 1.0,
        "CNN": 0.8,
        "BBC": 0.9,
        "Unknown": 0.5
    }
    return np.mean([credibility_scores.get(source, 0.5) for source in sources])

def compute_ranking(sentiments: list, timestamps: list, sources: list, impact_factors: list = None) -> float:
    """
    Compute stock ranking with additional source credibility factor.
    """
    if not sentiments:
        return 0.0

    sentiment_strength = compute_sentiment_strength(sentiments)
    recency_factor = compute_recency_factor(timestamps)
    market_impact = compute_market_impact(impact_factors)
    article_volume = compute_article_volume(sentiments)
    source_credibility = compute_source_credibility(sources)

    # Final ranking calculation with weighted factors
    ranking_score = (
        (sentiment_strength * 0.4) +
        (article_volume * 0.2) +
        (recency_factor * 0.2) +
        (market_impact * 0.1) +
        (source_credibility * 0.1)
    )
    return round(ranking_score, 3)

def display_results(articles: list, sentiments: list, ranking_score: float):
    """
    Display sentiment analysis results and ranking score.
    """
    print("📊 Sentiment Analysis Results (Ranked by Investment Importance):")
    
    # Combine articles, sentiments, and confidence scores
    combined_data = list(zip(articles, sentiments))
    
    # Sort by confidence and sentiment strength
    combined_data.sort(key=lambda x: (x[1]["confidence"], -1 if x[1]["label"] == "negative" else 1), reverse=True)
    
    # Display ranked results
    for i, (article, sentiment) in enumerate(combined_data, start=1):
        emoji = "📈" if sentiment["label"] == "positive" else "📉" if sentiment["label"] == "negative" else "➖"
        print(f"{i}. {emoji} {article[:50]}... -> Sentiment: {sentiment['label'].upper()} (Confidence: {sentiment['confidence']:.2f})")
    
    print(f"\n🏆 Ranking Score: {ranking_score}")
    print("---------------------------------")
    print("Score Interpretation:")
    print("+1.0: Strong Positive Outlook\n 0.0: Neutral Outlook\n-1.0: Strong Negative Outlook")

def main():
    # Example data for testing
    example_articles = [
        "The stock market surged today after the Federal Reserve announced lower interest rates.",
        "Investors are worried as inflation continues to rise, impacting global markets.",
        "Tech companies saw a sharp decline in their stock prices following new regulations.",
        "Oil prices hit a record high due to geopolitical tensions in the Middle East.",
        "The banking sector is experiencing significant growth after a strong earnings report.",
    ]
    
    # Simulate sentiment analysis
    sentiments = [predict_sentiment(article) for article in example_articles]
    timestamps = [datetime.utcnow()] * len(example_articles)  # Simulated timestamps
    
    # Simulate sources (replace with actual sources if available)
    sources = ["Reuters", "Bloomberg", "CNN", "BBC", "Unknown"]
    
    # Compute ranking score
    ranking_score = compute_ranking(sentiments, timestamps, sources)
    
    # Display results
    display_results(example_articles, sentiments, ranking_score)

if __name__ == "__main__":
    main()