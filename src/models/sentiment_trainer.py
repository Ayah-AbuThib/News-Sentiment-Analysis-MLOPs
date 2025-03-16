# File: src/models/sentiment_trainer.py
"""Model training logic for finbert sentiment analysis."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

def predict_sentiment(text: str) -> dict:
    """Classify sentiment using the pre-trained FinBERT model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    return {
        "label": model.config.id2label[logits.argmax().item()],
        "confidence": max(probabilities)
    }

# ✅ Test to verify the model works
sample_texts = [
    "The stock market surged today after the Federal Reserve announced lower interest rates.",
    "Investors are worried as inflation continues to rise, impacting global markets.",
    "Tech companies saw a sharp decline in their stock prices following new regulations.",
    "Oil prices hit a record high due to geopolitical tensions in the Middle East.",
    "The banking sector is experiencing significant growth after a strong earnings report."
]

for text in sample_texts:
    result = predict_sentiment(text)
    print(f"Text: {text} -> Sentiment: {result['label']} (Confidence: {result['confidence']:.2f})")