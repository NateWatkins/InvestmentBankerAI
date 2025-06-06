import os
import requests
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import pytz

# --- CONFIGURE ---
FINNHUB_KEY = "d0v4m71r01qmg3uk7fo0d0v4m71r01qmg3uk7fog"
TICKER = "NVDA"
NEWS_DIR = os.path.join("data", "news")
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_sentiment.csv")

# Initialize VADER
nltk.download('vader_lexicon', quiet=True)
VADER = SentimentIntensityAnalyzer()

def fetch_finnhub_news(ticker, start_unix, end_unix, api_key):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d"),
        "to": datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d"),
        "token": api_key
    }
    resp = requests.get(url, params=params)
    print("Requested URL:", resp.url)
    resp.raise_for_status()
    return resp.json()

def compute_sentiment_from_list(texts):
    if not texts:
        return 0.0
    scores = [VADER.polarity_scores(txt)["compound"] for txt in texts]
    return float(sum(scores) / len(scores))

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    # Use Pacific timezone
    pacific = pytz.timezone("US/Pacific")
    now_pacific = datetime.now(pacific).replace(second=0, microsecond=0)
    window_ago = now_pacific - timedelta(days=1)

    # Convert Pacific-aware datetimes to UTC timestamps (for Finnhub)
    start_utc = window_ago.astimezone(pytz.utc)
    end_utc = now_pacific.astimezone(pytz.utc)
    start_unix = int(start_utc.timestamp())
    end_unix = int(end_utc.timestamp())

    # Fetch and parse articles
    raw_articles = fetch_finnhub_news(TICKER, start_unix, end_unix, FINNHUB_KEY)

    print(f"Fetched {len(raw_articles)} total articles for {TICKER}.")
    for art in raw_articles[:5]:
        art_time = datetime.utcfromtimestamp(art["datetime"]).astimezone(pacific)
        print(f"  • {art_time.isoformat()} → {art['headline'][:80]}…")

    # Filter by time (in Pacific)
    recent_texts = []
    for art in raw_articles:
        art_time = datetime.utcfromtimestamp(art.get("datetime", 0)).astimezone(pacific)
        if art_time >= window_ago:
            title = art.get("headline", "")
            summary = art.get("summary", "")
            recent_texts.append(f"{title} {summary}")

    sentiment_score = compute_sentiment_from_list(recent_texts)

    # Record sentiment
    df_out = pd.DataFrame({
        "Datetime": [now_pacific.isoformat()],
        "Sentiment": [sentiment_score]
    })

    if os.path.exists(OUTPUT_CSV):
        df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"Appended Finnhub sentiment {sentiment_score:.4f} at {now_pacific.isoformat()} to {OUTPUT_CSV}")
