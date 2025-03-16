# File: src/api/streamlit_ui.py
import streamlit as st
import sys
import os
from datetime import datetime

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.sentiment_trainer import predict_sentiment
from src.data_processing.ranking import compute_ranking  

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown h3 {
        color: #4CAF50;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with emoji and image
st.title("📈 Stock Sentiment Analysis & Ranking")
st.image("https://img.freepik.com/free-photo/robot-with-planet-earth_1048-4552.jpg?t=st=1742057134~exp=1742060734~hmac=484382355e0de04c83e79ef55e8606e4d72692e6871115b85b94343f3dfac8b7&w=1380")

# Input area for news articles
st.markdown("### 📰 Enter News Articles")
news_input = st.text_area("News Articles", placeholder="Paste news articles here, one per line...", height=150, label_visibility="hidden")

# Analyze button
if st.button("🚀 Analyze Sentiment"):
    if not news_input.strip():
        st.warning("Please enter some news articles!")
    else:
        articles = [article.strip() for article in news_input.split("\n") if article.strip()]
        sentiments = [predict_sentiment(article) for article in articles]
        timestamps = [datetime.utcnow()] * len(articles)  # Simulate timestamps
        sources = ["Unknown"] * len(articles)  # Default source

        # Compute ranking score
        ranking_score = compute_ranking(sentiments, timestamps, sources)

        # Display sentiment analysis results
        st.write("### 📊 Sentiment Analysis")
        for article, sentiment in zip(articles, sentiments):
            sentiment_emoji = "😊" if sentiment["label"] == "positive" else "😠" if sentiment["label"] == "negative" else "😐"
            st.markdown(
                f"""
                <div style="background-color: #ffffff; padding: 10px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                    <p><strong>{article[:50]}...</strong> → {sentiment_emoji} {sentiment['label'].upper()} (Confidence: {sentiment['confidence']:.2f})</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display ranking score
        st.write("### 🏆 Ranking Score")
        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                <h2 style="color: #4CAF50;">Stock Ranking: {ranking_score}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Score interpretation
        st.write("### 📝 Score Interpretation")
        st.markdown(
            """
            - **+1.0**: Strong Positive Outlook
            - **0.0**: Neutral Outlook
            - **-1.0**: Strong Negative Outlook
            """
        )

# Footer
st.markdown("---")
st.markdown('<p class="footer">Made with ❤️ by Ayah Abu Thib</p>', unsafe_allow_html=True)