import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- CONFIGURE ---
FINNHUB_KEY = "d0v4m71r01qmg3uk7fo0d0v4m71r01qmg3uk7fog"   # Replace with your Finnhub key
TICKER      = "SPY"
NEWS_DIR    = os.path.join("data", "news")
OUTPUT_CSV  = os.path.join(NEWS_DIR, f"{TICKER}_sentiment.csv")

# Initialize VADER
nltk.download('vader_lexicon', quiet=True)
VADER = SentimentIntensityAnalyzer()

def fetch_finnhub_news(ticker, start_unix, end_unix, api_key):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d"),
        "to":   datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d"),
        "token": api_key
    }
    resp = requests.get(url, params=params)
    print("Requested URL:", resp.url)
    resp.raise_for_status()
    return resp.json()

def compute_sentiment_from_list(texts):
    if not texts:
        return 0.0  # Sentinel for "no news"
    scores = [VADER.polarity_scores(txt)["compound"] for txt in texts]
    return float(sum(scores) / len(scores))


if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    now_utc = datetime.utcnow()
    one_hour_ago = now_utc - timedelta(minutes=300) ##Five hours instead of an hour 
    start_of_day = (datetime.utcnow() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) #Pushes the time to be yesterday for testing 

    start_unix = int(start_of_day.timestamp())
    end_unix   = int(now_utc.timestamp())

    raw_articles = fetch_finnhub_news(TICKER, start_unix, end_unix, FINNHUB_KEY)

    # DEBUG: Show count and some article previews
    print(f"Fetched {len(raw_articles)} total articles for {TICKER} today.")
    for art in raw_articles[:5]:
        art_time = datetime.utcfromtimestamp(art["datetime"])
        print(f"  • {art_time.isoformat()} → {art['headline'][:80]}…")

    # Filter by time
    recent_texts = [] 
    for art in raw_articles:
        art_time = datetime.utcfromtimestamp(art.get("datetime", 0))
        if art_time >= one_hour_ago:
            title   = art.get("headline", "")
            summary = art.get("summary", "")
            recent_texts.append(f"{title} {summary}")

    sentiment_score = compute_sentiment_from_list(recent_texts)
    timestamp_min = now_utc.replace(second=0, microsecond=0)

    df_out = pd.DataFrame({
        "Datetime": [timestamp_min],
        "Sentiment": [sentiment_score]
    })
    if os.path.exists(OUTPUT_CSV):
        df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"Appended Finnhub sentiment {sentiment_score:.4f} at {timestamp_min.isoformat()} to {OUTPUT_CSV}")
