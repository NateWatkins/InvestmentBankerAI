import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time
import logging
import re
import argparse

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURE (Defaults) ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "maJATpmcBQDmp40WauaZhtVaK2UvBmC3")
TICKER = "TSLA"
NEWS_DIR = "data/news"
OUTPUT_CSV = os.path.join(NEWS_DIR, f"{TICKER}_sentiment_combined.csv")
LOOKBACK_DAYS = 5
BATCH_SIZE = 10  # For model inference
MAX_TEXT_LEN = 512  # Truncate to avoid OOM
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls if paginating
LAST_FETCH_FILE = os.path.join(NEWS_DIR, f"{TICKER}_last_fetch.txt")  # For incremental

# --- Load Models (with device auto-detection) ---
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)

roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device)

distilbert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if device in ["cuda", "mps"] else -1)

# --- Helper: Clean Text ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9.,!?\' ]+', '', text)  # Remove special chars
    return text[:MAX_TEXT_LEN]

# --- Helper: Polygon.io News Fetch with Pagination and Incremental ---
def fetch_polygon_news(ticker, start_unix, end_unix, api_key, limit=1000):
    results = []
    next_url = None
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "published_utc.gte": datetime.utcfromtimestamp(start_unix).strftime("%Y-%m-%d"),
        "published_utc.lte": datetime.utcfromtimestamp(end_unix).strftime("%Y-%m-%d"),
        "order": "desc",
        "limit": 100,  # Max per page
        "apiKey": api_key
    }

    while True:
        if next_url:
            resp = requests.get(next_url)
            time.sleep(RATE_LIMIT_DELAY)
        else:
            resp = requests.get(base_url, params=params)

        if resp.status_code != 200:
            logger.error(f"API error: {resp.status_code} - {resp.text}")
            break

        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"JSON parse error: {e} - Response: {resp.text[:500]}")
            break

        articles = data.get("results", [])
        if not articles:
            break

        for art in articles:
            dt_str = art.get("published_utc")
            if not dt_str:
                continue
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(pytz.UTC)
            except Exception as e:
                logger.warning(f"Date parse error for {dt_str}: {e}")
                continue
            results.append({
                "datetime": int(dt.timestamp()),
                "headline": art.get("title", ""),
                "summary": art.get("description", ""),
            })

        next_url = data.get("next_url")
        if not next_url or len(results) >= limit:
            break

    logger.info(f"Fetched {len(results)} articles")
    return results

# --- Sentiment Classification Functions ---
def classify_finbert(texts):
    inputs = finbert_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TEXT_LEN).to(device)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().tolist()  # [neg, neu, pos]
    return probs

def classify_roberta(texts):
    inputs = roberta_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TEXT_LEN).to(device)
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().tolist()  # [neg, neu, pos]
    return probs

def classify_distilbert(texts):
    results = distilbert_classifier(texts, truncation=True, max_length=MAX_TEXT_LEN)
    probs = []
    for res in results:
        label = res["label"].lower()
        score = res["score"]
        if label == "negative":
            probs.append([score, 0.0, 1 - score])  # neg, neu, pos
        else:
            probs.append([1 - score, 0.0, score])
    return probs

# --- Main Logic ---
def main(start_date=None, end_date=None, incremental=True):
    now_utc = datetime.now(pytz.UTC).replace(second=0, microsecond=0)
    
    if incremental and os.path.exists(LAST_FETCH_FILE):
        with open(LAST_FETCH_FILE, "r") as f:
            last_fetch = int(f.read().strip())
        start_unix = last_fetch + 1  # Start just after last
    else:
        start_unix = int((now_utc - timedelta(days=LOOKBACK_DAYS)).timestamp())

    if start_date:
        start_unix = int(datetime.fromisoformat(start_date).timestamp())
    end_unix = int(now_utc.timestamp()) if not end_date else int(datetime.fromisoformat(end_date).timestamp())

    articles = fetch_polygon_news(TICKER, start_unix, end_unix, POLYGON_KEY)
    if not articles:
        logger.info("No new articles found.")
        return

    rows = []
    texts = []
    timestamps = []

    for art in articles:
        dt = datetime.utcfromtimestamp(art.get("datetime", 0)).replace(tzinfo=pytz.UTC)
        if dt.timestamp() < start_unix:
            continue

        text = clean_text(f"{art.get('headline', '')} {art.get('summary', '')}")
        if not text:
            continue

        texts.append(text)
        timestamps.append(dt.isoformat())

        if len(texts) == BATCH_SIZE:
            process_batch(texts, timestamps, rows)
            texts, timestamps = [], []

    if texts:  # Process remaining
        process_batch(texts, timestamps, rows)

    if rows:
        df_out = pd.DataFrame(rows)
        df_out["Datetime"] = pd.to_datetime(df_out["Datetime"])
        df_out = df_out.groupby("Datetime", as_index=False).mean(numeric_only=True)

        os.makedirs(NEWS_DIR, exist_ok=True)
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV, parse_dates=["Datetime"])
            df_out = pd.concat([df_existing, df_out]).drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
        
        df_out.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved {len(df_out)} sentiment entries (including existing) to {OUTPUT_CSV}")

        # Update last fetch
        with open(LAST_FETCH_FILE, "w") as f:
            f.write(str(end_unix))

def process_batch(texts, timestamps, rows):
    try:
        probs_fin = classify_finbert(texts)
        probs_rob = classify_roberta(texts)
        probs_dis = classify_distilbert(texts)

        for i in range(len(texts)):
            rows.append({
                "Datetime": timestamps[i],
                "Ticker": TICKER,
                "Prob_Pos_finbert": probs_fin[i][2],
                "Prob_Neu_finbert": probs_fin[i][1],
                "Prob_Neg_finbert": probs_fin[i][0],
                "Prob_Pos_roberta": probs_rob[i][2],
                "Prob_Neu_roberta": probs_rob[i][1],
                "Prob_Neg_roberta": probs_rob[i][0],
                "Prob_Pos_distilbert": probs_dis[i][2],
                "Prob_Neu_distilbert": probs_dis[i][1],
                "Prob_Neg_distilbert": probs_dis[i][0],
            })
    except Exception as e:
        logger.error(f"Batch processing error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Sentiment Fetcher")
    parser.add_argument("--start", type=str, help="Start date (ISO8601)")
    parser.add_argument("--end", type=str, help="End date (ISO8601)")
    parser.add_argument("--no-incremental", action="store_false", dest="incremental", help="Disable incremental fetch")
    args = parser.parse_args()
    main(args.start, args.end, args.incremental)