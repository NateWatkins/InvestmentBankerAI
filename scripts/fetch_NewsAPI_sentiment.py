import os
import requests
import pandas as pd
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import pytz  # <-- for timezone conversion

# --- CONFIGURE ---
NEWSAPI_KEY = "36c027b1f746405b92335f0c0f06494b"
TICKER = "Nvidia"
NEWS_DIR = os.path.join("data", "news")
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_sentiment.csv")

# --- Initialize FinBERT ---
FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def fetch_newsapi_articles(keyword, from_date, to_date, api_key):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,
        "apiKey": api_key
    }
    resp = requests.get(url, params=params)
    print("Requested URL:", resp.url)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return articles

def compute_sentiment_from_list(texts):
    if not texts:
        return 0.0
    scores = []
    for txt in texts:
        encoded_input = FINBERT_TOKENIZER(txt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = FINBERT_MODEL(**encoded_input)
            logits = output.logits.numpy()[0]
            probs = softmax(logits)
            score = probs[2] - probs[0]
            scores.append(score)
    return float(sum(scores) / len(scores))

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    # Use Pacific time for now and 24-hour window
    pacific = pytz.timezone("US/Pacific")
    now_pacific = datetime.now(pacific).replace(second=0, microsecond=0)
    window_ago = now_pacific - timedelta(days=1)

    # Fetch articles from date range (use date only for NewsAPI param)
    start_of_range = window_ago
    end_of_range = now_pacific

    raw_articles = fetch_newsapi_articles(TICKER, start_of_range, end_of_range, NEWSAPI_KEY)

    print(f"Fetched {len(raw_articles)} total articles for {TICKER}.")
    for art in raw_articles[:5]:
        published_time = art.get("publishedAt", "")
        print(f"  • {published_time} → {art.get('title', '')[:80]}…")

    # Filter recent articles from last 24 hours (aware-aware comparison)
    recent_texts = []
    for art in raw_articles:
        try:
            published_raw = art.get("publishedAt", "")
            art_time = dateparser.parse(published_raw).astimezone(pacific)
            if art_time >= window_ago:
                title = art.get("title", "")
                summary = art.get("description", "")
                recent_texts.append(f"{title} {summary}")
        except Exception as e:
            print(f"[ERROR] Failed to parse time: {published_raw} → {e}")
            continue

    print(f"\nFiltered {len(recent_texts)} articles from the past 24 hours (Pacific Time).")
    for txt in recent_texts:
        print("•", txt)

    # Optional: Compute sentiment
    sentiment_score = compute_sentiment_from_list(recent_texts)


    # Optional: Save to CSV
    df_out = pd.DataFrame({
        "Datetime": [now_pacific.isoformat()],
        "Sentiment": [sentiment_score]
    })

    if os.path.exists(OUTPUT_CSV):
        df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"Appended FinBERT sentiment {sentiment_score:.4f} at {now_pacific.isoformat()} to {OUTPUT_CSV}")
