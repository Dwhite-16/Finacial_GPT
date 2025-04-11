import os
import time
import logging
import pandas as pd
import yfinance as yf
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
from newsapi import NewsApiClient
from dotenv import load_dotenv
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Caching config
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Login setup
login_manager = LoginManager()
login_manager.init_app(app)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Cached news storage
latest_news_df = None

# API Keys
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not NEWS_API_KEY or not HUGGINGFACE_API_TOKEN:
    logging.error("API keys missing. Please check your .env file.")
    exit(1)

# User class
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id, "example_user")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User(1, username)
        login_user(user)
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# ✅ Scheduled background job to auto-refresh news
def scheduled_news_update():
    global latest_news_df
    logging.info("🔁 Scheduled update: Fetching and analyzing financial news...")
    df = fetch_financial_news()
    if df is not None:
        df = analyze_sentiment(df)
        df = textblob_sentiment_analysis(df)
        latest_news_df = df
        logging.info("✅ News and sentiment data updated and cached.")

scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_news_update, trigger="interval", minutes=30)
scheduler.start()

# ✅ Updated /get_news with cache fallback
@app.route('/get_news', methods=['GET'])
def get_news():
    global latest_news_df
    if latest_news_df is None:
        logging.info("📡 No cached news found, fetching now...")
        scheduled_news_update()
    if latest_news_df is not None:
        return jsonify(latest_news_df.to_dict(orient="records"))
    else:
        return jsonify({"error": "No financial news available."}), 400

# Hugging Face question answering
@app.route("/ask-question", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No question provided"}), 400

    response = process_question(query)
    return jsonify({"answer": response})

def process_question(query):
    try:
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": f"<|system|>You are a helpful financial assistant.<|user|>{query}<|assistant|>",
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            return generated_text.strip()
        else:
            logging.error(f"❌ Hugging Face API Error: {response.status_code} - {response.text}")
            return "Sorry, Hugging Face API could not process the question."
    except Exception as e:
        logging.error(f"❌ Error in process_question: {e}")
        return "Sorry, something went wrong while processing your question."

# ✅ Improved financial news fetcher
def fetch_financial_news(max_retries=3, backoff_factor=2):
    logging.info("📰 Fetching financial news...")
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    sources = ['bbc-news', 'cnn', 'reuters', 'business-insider']
    news_data = []

    for source in sources:
        for attempt in range(max_retries):
            try:
                logging.info(f"🔍 Fetching from source: {source} (attempt {attempt + 1})")
                articles = newsapi.get_top_headlines(sources=source, language="en")
                if articles and isinstance(articles, dict) and "articles" in articles:
                    titles = [article.get("title", "") for article in articles["articles"] if article.get("title")]
                    news_data.extend(titles)
                    break
                else:
                    logging.warning(f"⚠️ Unexpected response from {source}: {articles}")
            except Exception as e:
                logging.warning(f"⚠️ Error fetching from {source} (attempt {attempt + 1}): {e}")
                time.sleep(backoff_factor * (attempt + 1))

    if not news_data:
        logging.info("📉 No news from sources. Falling back to category='business'")
        try:
            articles = newsapi.get_top_headlines(category='business', language='en', country='us')
            if articles and "articles" in articles:
                news_data = [article.get("title", "") for article in articles["articles"] if article.get("title")]
        except Exception as e:
            logging.error(f"❌ Error fetching fallback news: {e}")

    return pd.DataFrame(news_data, columns=["Headline"]) if news_data else None

# ✅ Sentiment analyzers
def analyze_sentiment(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for Vader sentiment analysis.")
        return None
    logging.info("📈 Performing Vader sentiment analysis...")
    df["Vader Sentiment Score"] = df[column].apply(lambda text: sia.polarity_scores(str(text))["compound"])
    df["Vader Sentiment"] = df["Vader Sentiment Score"].apply(lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral"))
    return df

def textblob_sentiment_analysis(df, column="Headline"):
    if df is None or column not in df.columns:
        logging.warning("⚠️ No valid data for TextBlob sentiment analysis.")
        return None
    logging.info("💬 Performing TextBlob sentiment analysis...")
    sentiments = []
    for text in df[column]:
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "Positive"
            elif polarity < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            sentiments.append(sentiment)
        except Exception as e:
            logging.error(f"Error analyzing sentiment for '{text}': {e}")
            sentiments.append("Error")
    df["TextBlob Sentiment"] = sentiments
    return df

# Shutdown scheduler on exit
import atexit
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
