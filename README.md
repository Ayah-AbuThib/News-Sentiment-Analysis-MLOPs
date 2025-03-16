# 📈 Stock Sentiment Analysis and Classification

This project performs **sentiment analysis** on financial news articles and calculates a **stock rating** based on sentiment, article volume, recency, and source credibility. The application is built using **Streamlit** for the interactive user interface and **Hugging Face's Inference API** for text classification.

## 🚀 Live Demo

You can view a live demo of the application hosted on AWS [here](http://3.86.209.126:8501/).

## 📌 Project Overview

This project analyzes financial news articles, evaluates their sentiment (**positive, negative, or neutral**), and calculates a ranking score based on:

- **Sentiment Strength**: Determines if articles are positive, negative, or neutral.
- **Article Volume**: Number and distribution of articles.
- **Freshness Factor**: How recent the news articles are.
- **Market Impact**: The effect of the news on stock prices.
- **Source Credibility**: The trustworthiness of the news source.

The project is deployed using **AWS EC2** and is accessible via a **Streamlit** application.

## 🛠 Technologies Used

### 🌐 Core Technologies

- **Python 3.10+**: Core programming language for backend logic and data processing.
- **Streamlit**: Interactive web application framework for building the UI.
- **Hugging Face Transformers**: FinBERT model for financial sentiment analysis.
- **AWS EC2**: Cloud hosting and deployment infrastructure.

### 📊 Data Processing & Analysis

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing operations.
- **DVC (Data Version Control)**: Dataset versioning and pipeline management.

### 🤖 Machine Learning & NLP

- **PyTorch**: Deep learning framework.
- **Transformers**: State-of-the-art NLP models.

### 📈 Data Visualization

- **Matplotlib/Seaborn**: Static visualizations.

### 📦 Utility Libraries

- **Python-dotenv**: Environment management.
- **Emoji**: Emoji handling in text.

## 🛠 Setup and Run Locally

1. Clone the repository:
    
    ```bash
    git clone https://github.com/Ayah-AbuThib/News-Sentiment-Analysis-MLOPs.git
    cd test
    ```
    
2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the Streamlit application:
    
    ```bash
    streamlit run news.py
    ```
    
    The application will be available at [http://localhost:8501](http://localhost:8501/).
    

## ☁️ Deploy on AWS

1. **Launch an EC2 instance** and connect via SSH.
2. **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Upload the code** to the EC2 instance.
4. **Run the application**:
    
    ```bash
    streamlit run news.py
    ```
    
5. **Access** the application at `http://<your-ec2-ip>:8501`.

## 🧠 Code Explanation

### 📌 `predict_sentiment(text: str)`

Analyzes sentiment using the Hugging Face API and returns a confidence score.

```python
def predict_sentiment(text: str):
    results = client.text_classification(text, model="ProsusAI/finbert")
    if results:
        sentiments = {res['label']: res['score'] for res in results}
        predicted_sentiment = max(sentiments, key=sentiments.get)
        confidence = sentiments[predicted_sentiment]
        return {"label": predicted_sentiment.lower(), "confidence": confidence}
    else:
        return {"label": "negative", "confidence": 0.0}
```

### 📌 Classification Functions

The ranking system considers multiple factors like sentiment strength, recency, market impact, and source credibility to generate an overall stock rating.

---

## 📂 Project Structure

```
News-Sentiment-Analysis/
│
├── models/                  
│   └── model_logs/          # Model training logs
│
├── notebooks/               
│   └── EDA.ipynb            # Exploratory Data Analysis
│
├── src/                     
│   ├── data_processing/     # Data cleaning and analysis
│   ├── ranking.py
│   ├── preprocess.py        # Data preprocessing
│   ├── models/              # Modeling code
│   │   └── sentiment_trainer.py  # Model training
│   ├── config/ 
│   │   └── settings.py
│   └── news.py              # Main application logic
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── docker-compose.yml
├── Dockerfile
├── news.py
└── .gitignore               # Specifies files to ignore in Git
```

---

## 🎯 Future Enhancements

- 🔹 **Improve sentiment classification** by fine-tuning FinBERT on financial-specific datasets.
- 🔹 **Enhance stock ranking algorithms** by incorporating real-time stock price movements.
- 🔹 **Expand deployment** to include Docker and Kubernetes for scalable cloud hosting.
