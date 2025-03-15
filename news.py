import sys
import os
import numpy as np
import torch
import streamlit as st
from datetime import datetime
from huggingface_hub import InferenceClient  # Added missing import

# Initialize Hugging Face Inference Client
client = InferenceClient(
    api_key="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"
)

def predict_sentiment(text: str):
    """
    Preprocess the text, tokenize it, and predict the class using the model.
    """
    results = client.text_classification(text, model="ProsusAI/finbert")

    if results:
        sentiments = {res['label']: res['score'] for res in results}
        predicted_sentiment = max(sentiments, key=sentiments.get)  # Choose highest score
        confidence = sentiments[predicted_sentiment]
        return {"label": predicted_sentiment.lower(), "confidence": confidence}
    else:
        return {"label": "negative", "confidence": 0.0}

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define credibility scores
CREDIBILITY_SCORES = {
    "Reuters": 1.0,
    "Bloomberg": 1.0,
    "CNN": 0.8,
    "BBC": 0.9,
    "Unknown": 0.5
}

# Compute ranking components
def compute_sentiment_strength(sentiments):
    sentiment_weights = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    return np.mean([s["confidence"] * sentiment_weights.get(s["label"], 0.0) for s in sentiments]) if sentiments else 0.0

def compute_recency_factor(timestamps):
    now = datetime.utcnow()
    decay_rate = 0.1
    return np.sum(np.exp(-decay_rate * np.array([(now - t).days for t in timestamps]))) / len(timestamps) if timestamps else 0.0

def compute_market_impact(impact_factors):
    return np.mean(impact_factors) if impact_factors else 1.0

def compute_article_volume(sentiments):
    return np.log(len(sentiments) + 1)

def compute_source_credibility(sources):
    return np.mean([CREDIBILITY_SCORES.get(source, 0.5) for source in sources])

def compute_ranking(sentiments, timestamps, sources, impact_factors=None):
    if not sentiments:
        return 0.0
    return round(
        (compute_sentiment_strength(sentiments) * 0.4) +
        (compute_article_volume(sentiments) * 0.2) +
        (compute_recency_factor(timestamps) * 0.2) +
        (compute_market_impact(impact_factors) * 0.1) +
        (compute_source_credibility(sources) * 0.1), 3
    )

# Streamlit UI
st.title("📈 Stock Sentiment Analysis & Ranking")
st.image("https://img.freepik.com/free-photo/robot-with-planet-earth_1048-4552.jpg?t=st=1742057134~exp=1742060734~hmac=484382355e0de04c83e79ef55e8606e4d72692e6871115b85b94343f3dfac8b7&w=1380")

st.markdown("### 📰 Enter News Articles")
news_input = st.text_area("News Articles", placeholder="Paste news articles here, one per line...", height=150)

if st.button("🚀 Analyze Sentiment"):
    if not news_input.strip():
        st.warning("Please enter some news articles!")
    else:
        articles = [article.strip() for article in news_input.split("\n") if article.strip()]
        sentiments = [predict_sentiment(article) for article in articles]
        timestamps = [datetime.utcnow()] * len(articles)
        sources = ["Unknown"] * len(articles)
        ranking_score = compute_ranking(sentiments, timestamps, sources)

        st.write("### 📊 Sentiment Analysis")
        for article, sentiment in zip(articles, sentiments):
            sentiment_emoji = "😊" if sentiment["label"] == "positive" else "😠" if sentiment["label"] == "negative" else "😐"
            st.markdown(f"**{article[:50]}...** → {sentiment_emoji} {sentiment['label'].upper()} (Confidence: {sentiment['confidence']:.2f})")

        st.write("### 🏆 Ranking Score")
        st.markdown(f"**Stock Ranking: {ranking_score}**")

        st.write("### 📝 Score Interpretation")
        st.markdown("- **+1.0**: Strong Positive Outlook\n- **0.0**: Neutral Outlook\n- **-1.0**: Strong Negative Outlook")

st.markdown("---")
st.markdown('<p class="footer">Made with ❤️ by Ayah Abu Thib</p>', unsafe_allow_html=True)
