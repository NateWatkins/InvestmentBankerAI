import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- CONFIGURE ---
FINNHUB_KEY = "d0v4m71r01qmg3uk7fo0d0v4m71r01qmg3uk7fog"
TICKER = "TSLA"
NEWS_DIR = "data/news"
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_sentiment_combined.csv")

# --- Load Models ---
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

distilbert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- Helpers ---
def fetch_finnhub_news(ticker, start_unix, end_unix, api_key):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d"),
        "to": datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d"),
        "token": api_key
    }
    return requests.get(url, params=params).json()

def classify_finbert(text):
    inputs = finbert_tokenizer(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(finbert_model(**inputs).logits, dim=1)[0]
    return probs.tolist()

def classify_roberta(text):
    inputs = roberta_tokenizer(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(roberta_model(**inputs).logits, dim=1)[0]
    return probs.tolist()

def classify_distilbert(text):
    result = distilbert_classifier(text[:512])[0]
    label = result["label"].lower()
    score = result["score"]
    return [score, 0.0, 1 - score] if label == "negative" else [1 - score, 0.0, score]

# --- Main Logic ---
if __name__ == "__main__":
    now_utc = datetime.now(pytz.UTC).replace(second=0, microsecond=0)
    window_ago = now_utc - timedelta(days=5)
    start_unix = int(window_ago.timestamp())
    end_unix = int(now_utc.timestamp())

    articles = fetch_finnhub_news(TICKER, start_unix, end_unix, FINNHUB_KEY)
    rows = []

    for art in articles:
        dt = datetime.utcfromtimestamp(art.get("datetime", 0)).replace(tzinfo=pytz.UTC)
        if dt < window_ago:
            continue

        text = f"{art.get('headline', '')} {art.get('summary', '')}".strip()
        if not text:
            continue

        try:
            probs_fin = classify_finbert(text)
            probs_rob = classify_roberta(text)
            probs_dis = classify_distilbert(text)

            rows.append({
                "Datetime": dt.isoformat(),
                "Ticker": TICKER,
                "Prob_Pos_finbert": probs_fin[2],
                "Prob_Neu_finbert": probs_fin[1],
                "Prob_Neg_finbert": probs_fin[0],
                "Prob_Pos_roberta": probs_rob[2],
                "Prob_Neu_roberta": probs_rob[1],
                "Prob_Neg_roberta": probs_rob[0],
                "Prob_Pos_distilbert": probs_dis[2],
                "Prob_Neu_distilbert": probs_dis[1],
                "Prob_Neg_distilbert": probs_dis[0],
            })
        except Exception as e:
            print(f"❌ Error: {e}")

    if rows:
        df_out = pd.DataFrame(rows)
        df_out["Datetime"] = pd.to_datetime(df_out["Datetime"])

        # 🧠 Group by timestamp and average duplicate rows
        df_out = df_out.groupby("Datetime", as_index=False).mean(numeric_only=True)

        os.makedirs(NEWS_DIR, exist_ok=True)
        df_out = df_out.sort_values("Datetime")
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Saved {len(df_out)} averaged sentiment entries to {OUTPUT_CSV}")
