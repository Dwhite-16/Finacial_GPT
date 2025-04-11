import os
import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore
import requests
from newsapi import NewsApiClient  # type: ignore
from dotenv import load_dotenv  # type: ignore
from datasets import load_dataset
import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import re
nltk.download("stopwords")
from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
from transformers import pipeline

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("NEWSAPI_KEY")
if API_KEY is None:
    raise ValueError("Please set your NEWSAPI_KEY in the .env file.")

# Ensure necessary NLP resources are available
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Initialize financial sentiment model
finbert_sentiment = pipeline("text-classification", model="ProsusAI/finbert")


# 📌 **Step 2: Fetch Financial News from Multiple Sources**
def fetch_financial_news():
    print("📰 Fetching financial news...")
    newsapi = NewsApiClient(api_key=API_KEY)
    sources = ['bbc-news', 'cnn', 'business-insider', 'reuters', 'the-wall-street-journal']  # Add more sources
    news_data = []
    
    for source in sources:
        try:
            articles = newsapi.get_top_headlines(sources=source, category="business", language="en")
            if articles and "articles" in articles:
                news_data.extend([article["title"] for article in articles["articles"] if article["title"]])
        except Exception as e:
            print(f"❌ Error fetching news from {source}: {e}")
    
    if not news_data:
        print("⚠️ No news articles found.")
        return None
    return pd.DataFrame(news_data, columns=["Headline"])

# 📌 **Step 3: Fetch Tesla (TSLA) SEC Filings**
def fetch_sec_filings(cik="0001318605"):
    print(f"📜 Fetching SEC filings for CIK {cik}...")
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json"
    headers = {"User-Agent": "your-email@example.com"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ SEC API request failed! Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching SEC data: {e}")
        return None

# 📌 **Step 4: Load Financial Phrasebank Dataset**
def load_financial_phrasebank():
    print("📚 Loading financial_phrasebank dataset...")
    try:
        dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
        return pd.DataFrame(dataset["train"])  # Convert to DataFrame
    except Exception as e:
        print(f"❌ Error loading financial_phrasebank dataset: {e}")
        return None

# 📌 **Step 5: Clean Financial Text Data**
def clean_financial_text(df):
    if df is None:
        return None

    print("🧹 Cleaning financial text data...")
    
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
        words = text.split()
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return " ".join(words)

    if "Headline" in df.columns:
        df["Headline"] = df["Headline"].astype(str).apply(preprocess_text)

    return df

# 📌 **Step 6: Perform Sentiment Analysis**
def analyze_sentiment(df, column="Headline"):
    if df is None or column not in df.columns:
        return None

    print("📈 Performing sentiment analysis...")
    sia = SentimentIntensityAnalyzer()

    df["Vader Sentiment Score"] = df[column].apply(lambda text: sia.polarity_scores(str(text))["compound"])
    df["Vader Sentiment"] = df["Vader Sentiment Score"].apply(lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))

    return df

# 📌 **Step 7: Perform FinBERT Sentiment Analysis**
def finbert_sentiment_analysis(df, column="Headline"):
    if df is None or column not in df.columns:
        return None

    print("🤖 Performing FinBERT sentiment analysis...")
    df["FinBERT Sentiment"] = df[column].apply(lambda text: finbert_sentiment(str(text))[0]["label"])
    
    return df

# 📌 **Step 8: Validate & Fix Stock Data (Remove Timezones)**
def validate_stock_data(df):
    if df is None or "Close" not in df.columns:
        return None

    print("✅ Validating stock data accuracy...")
    
    # ✅ Remove timezone from datetime column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    df["Stock Price Change"] = df["Close"].pct_change()
    threshold = 0.2  # 20% change is an anomaly
    df["Anomaly"] = np.where(abs(df["Stock Price Change"]) > threshold, "Potential Outlier", "Normal")

    return df

# 📌 **Step 9: Save Cleaned Data**
def save_data():
    print("💾 Saving cleaned data to 'financial_data.xlsx'...")
    with pd.ExcelWriter("financial_data.xlsx") as writer:
        # Ensure at least one sheet is saved
        sheet_written = False
        
        # Save stock data (multiple stocks)
        if cleaned_stock_data:
            for ticker, data in cleaned_stock_data.items():
                if data is not None:
                    data.to_excel(writer, sheet_name=f"Stock Data - {ticker}", index=False)
                    sheet_written = True
        
        # Save news data
        if analyzed_news_data is not None:
            analyzed_news_data.to_excel(writer, sheet_name="Finance News", index=False)
            sheet_written = True
        
        # Save financial phrasebank data
        if phrasebank_data is not None:
            phrasebank_data.to_excel(writer, sheet_name="Financial Phrasebank", index=False)
            sheet_written = True
        
        # Check if at least one sheet was written
        if not sheet_written:
            raise ValueError("At least one sheet must be visible in the Excel file.")
    
    print("✅ Data saved successfully!")


# 🚀 **Execute the Enhanced Pipeline**
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]  # Example of S&P 500 companies
stock_data = fetch_stock_data(tickers)

news_data = fetch_financial_news()
sec_data = fetch_sec_filings()
phrasebank_data = load_financial_phrasebank()

# Clean & Validate Data
cleaned_stock_data = {ticker: validate_stock_data(data) for ticker, data in stock_data.items()}  # ✅ NOW FIXES TIMEZONE ISSUES
cleaned_news_data = clean_financial_text(news_data)

# Apply Sentiment Analysis
analyzed_news_data = analyze_sentiment(cleaned_news_data)
analyzed_news_data = finbert_sentiment_analysis(analyzed_news_data)

# Save the processed data
save_data()
