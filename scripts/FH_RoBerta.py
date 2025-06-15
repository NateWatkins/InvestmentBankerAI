import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# --- CONFIGURE ---
FINNHUB_KEY = "d0v4m71r01qmg3uk7fo0d0v4m71r01qmg3uk7fog"
TICKER = "TSLA"
NEWS_DIR = "/Users/natwat/Desktop/CPSC_Projects/Trader/data/news"
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_roberta_sentiment.csv")

# Initialize Twitter RoBERTa sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['negative', 'neutral', 'positive']

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
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        sentiment = labels[pred_idx]
        return sentiment, probs.tolist()

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    pacific = pytz.timezone("US/Pacific")
    now_pacific = datetime.now(pacific).replace(second=0, microsecond=0)
    window_ago = now_pacific - timedelta(days=1)

    # Convert to UTC
    start_unix = int(window_ago.astimezone(pytz.utc).timestamp())
    end_unix = int(now_pacific.astimezone(pytz.utc).timestamp())

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
            "Prob_Positive": scores[2],
            "Prob_Neutral": scores[1],
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
