import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# --- CONFIGURE ---
NEWSAPI_KEY = "36c027b1f746405b92335f0c0f06494b"  # ← Replace with your NewsAPI key
TICKER      = "Nvidia"  # Used as keyword in NewsAPI
NEWS_DIR    = os.path.join("data", "news")
OUTPUT_CSV  = os.path.join(NEWS_DIR, f"{TICKER}_sentiment.csv")

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
    print(articles)
    return articles

def compute_sentiment_from_list(texts):
    if not texts:
        return 0.0  # Sentinel value when no news found
    scores = []
    for txt in texts:
        encoded_input = FINBERT_TOKENIZER(txt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = FINBERT_MODEL(**encoded_input)
            logits = output.logits.numpy()[0]
            probs = softmax(logits)
            # probs = [negative, neutral, positive]
            score = probs[2] - probs[0]  # Positive - Negative
            scores.append(score)
    return float(sum(scores) / len(scores))

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)

    # Accurate UTC timestamps
    now_utc = datetime.utcnow().replace(second=0, microsecond=0)
    window_ago = now_utc - timedelta(days=1)
    start_of_day = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # 1) Fetch articles
    raw_articles = fetch_newsapi_articles(TICKER, start_of_day, now_utc, NEWSAPI_KEY)

    print(f"Fetched {len(raw_articles)} total articles for {TICKER}.")
    for art in raw_articles[:5]:
        published_time = art.get("publishedAt", "")
        print(f"  • {published_time} → {art.get('title', '')[:80]}…")

    # 2) Filter for last 5 hours
    recent_texts = []
    for art in raw_articles:
        print(art.keys())

    for art in raw_articles:
        try:
            art_time = datetime.strptime(art.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ")
            if art_time >= window_ago:
                title = art.get("title", "")
                summary = art.get("description", "")
                recent_texts.append(f"{title} {summary}")
        except Exception as e:
            continue
    print(recent_texts)
    # 3) Compute FinBERT sentiment
    #sentiment_score = compute_sentiment_from_list(recent_texts)
    '''
    # 4) Save to CSV
    df_out = pd.DataFrame({
        "Datetime": [now_utc.isoformat()],
        "Sentiment": [sentiment_score]
    })

    if os.path.exists(OUTPUT_CSV):
        df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_out.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Appended FinBERT sentiment {sentiment_score:.4f} at {now_utc.isoformat()} to {OUTPUT_CSV}")
'''