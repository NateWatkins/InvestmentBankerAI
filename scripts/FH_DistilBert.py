import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from transformers import pipeline

# --- CONFIGURE ---
FINNHUB_KEY = "d0v4m71r01qmg3uk7fo0d0v4m71r01qmg3uk7fog"
TICKER = "TSLA"
NEWS_DIR = "/Users/natwat/Desktop/CPSC_Projects/Trader/data/news"
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_distilbert_sentiment.csv")

# Initialize DistilBERT SST-2 Sentiment Classifier
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_finnhub_news(ticker, start_unix, end_unix, api_key):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d"),
        "to": datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d"),
        "token": api_key
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def classify_sentiment(text):
    # Truncate to 512 tokens and get sentiment
    result = classifier(text[:512])[0]
    label = result["label"].lower()  # 'positive' or 'negative'
    score = result["score"]
    if label == "positive":
        return label, [1 - score, score]  # [Prob_Negative, Prob_Positive]
    else:
        return label, [score, 1 - score]

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    # Time range: past 24 hours in Pacific Time
    pacific = pytz.timezone("US/Pacific")
    now_pacific = datetime.now(pacific).replace(second=0, microsecond=0)
    window_ago = now_pacific - timedelta(days=1)

    # Convert to UTC for Finnhub API
    start_unix = int(window_ago.astimezone(pytz.utc).timestamp())
    end_unix = int(now_pacific.astimezone(pytz.utc).timestamp())

    # Fetch news from Finnhub
    articles = fetch_finnhub_news(TICKER, start_unix, end_unix, FINNHUB_KEY)
    print(f"Fetched {len(articles)} articles.")

    rows = []
    for art in articles:
        dt = datetime.utcfromtimestamp(art.get("datetime", 0)).astimezone(pacific)
        if dt < window_ago:
            continue
        text = f"{art.get('headline', '')} {art.get('summary', '')}".strip()
        if not text:
            continue
        sentiment, scores = classify_sentiment(text)
        row = {
            "Datetime": dt.isoformat(),
            "Ticker": TICKER,
            "Sentiment": sentiment,
            "Prob_Positive": scores[1],
            "Prob_Neutral": 0.0,
            "Prob_Negative": scores[0]
        }
        rows.append(row)

    if rows:
        df_out = pd.DataFrame(rows)
        if os.path.exists(OUTPUT_CSV):
            df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        else:
            df_out.to_csv(OUTPUT_CSV, index=False)

        print(f"Wrote {len(rows)} sentiment entries to {OUTPUT_CSV}")
    else:
        print("No qualifying news found.")
