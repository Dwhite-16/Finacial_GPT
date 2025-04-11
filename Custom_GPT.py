import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from newsapi import NewsApiClient
from dotenv import load_dotenv
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    logging.error("NEWSAPI_KEY is missing. Please set it in the .env file.")
    exit(1)

# Download necessary NLP resources once
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Initialize sentiment analysis models
finbert_sentiment = pipeline("text-classification", model="ProsusAI/finbert")
sia = SentimentIntensityAnalyzer()


# Function to fetch financial news with retry mechanism
def fetch_financial_news(max_retries=3, backoff_factor=2):
    logging.info("📰 Fetching financial news...")
    newsapi = NewsApiClient(api_key=API_KEY)
    sources = ['bbc-news', 'cnn', 'business-insider', 'reuters', 'the-wall-street-journal']
    news_data = []

    for source in sources:
        for attempt in range(max_retries):
            try:
                articles = newsapi.get_top_headlines(sources=source, language="en")
                if articles and "articles" in articles:
                    news_data.extend([article["title"] for article in articles["articles"] if article["title"]])
                break  # Exit loop if successful
            except Exception as e:
                logging.warning(f"⚠️ Error fetching news from {source} (attempt {attempt + 1}): {e}")
                time.sleep(backoff_factor * (attempt + 1))  # Exponential backoff

    if not news_data:
        logging.warning("⚠️ No news articles found.")
        return None
    return pd.DataFrame(news_data, columns=["Headline"])

# Fetch SEC Filings
def fetch_sec_filings(cik="0001318605"):
    logging.info(f"📜 Fetching SEC filings for CIK {cik}...")
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json"
    headers = {"User-Agent": "your-email@example.com"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logging.warning(f"⚠️ SEC API request failed! Status Code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"❌ Error fetching SEC data: {e}")
        return None

# Load Financial Phrasebank Dataset
def load_financial_phrasebank():
    logging.info("📚 Loading financial_phrasebank dataset...")
    try:
        dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
        return pd.DataFrame(dataset["train"])
    except Exception as e:
        logging.error(f"❌ Error loading financial_phrasebank dataset: {e}")
        return None

# Perform Sentiment Analysis
def analyze_sentiment(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for sentiment analysis.")
        return None

    logging.info("📈 Performing Vader sentiment analysis...")
    df["Vader Sentiment Score"] = df[column].apply(lambda text: sia.polarity_scores(str(text))["compound"])
    df["Vader Sentiment"] = df["Vader Sentiment Score"].apply(lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
    return df

# Perform FinBERT Sentiment Analysis
def finbert_sentiment_analysis(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for FinBERT sentiment analysis.")
        return None

    logging.info("🤖 Performing FinBERT sentiment analysis...")
    df["FinBERT Sentiment"] = df[column].apply(lambda text: finbert_sentiment(str(text))[0]["label"])
    return df

# Save Cleaned Data
def save_data(cleaned_stock_data, analyzed_news_data, phrasebank_data):
    if not any([cleaned_stock_data, analyzed_news_data, phrasebank_data]):
        logging.warning("⚠️ Skipping data saving as required datasets are missing!")
        return

    logging.info("💾 Saving cleaned data to 'financial_data.xlsx'...")
    with pd.ExcelWriter("financial_data.xlsx") as writer:
        if cleaned_stock_data:
            for ticker, data in cleaned_stock_data.items():
                if data is not None:
                    data.to_excel(writer, sheet_name=f"Stock Data - {ticker}", index=False)
        if analyzed_news_data is not None:
            analyzed_news_data.to_excel(writer, sheet_name="Finance News", index=False)
        if phrasebank_data is not None:
            phrasebank_data.to_excel(writer, sheet_name="Financial Phrasebank", index=False)
    
    logging.info("✅ Data saved successfully!")

# 🚀 **Execute the Enhanced Pipeline**
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
stock_data = fetch_all_stock_data(tickers)
news_data = fetch_financial_news()

# ✅ Load Financial Phrasebank Data before saving
phrasebank_data = load_financial_phrasebank()


# Apply Sentiment Analysis
analyzed_news_data = analyze_sentiment(news_data)
analyzed_news_data = finbert_sentiment_analysis(analyzed_news_data)

# Save the processed data
save_data(cleaned_stock_data, analyzed_news_data, phrasebank_data)


