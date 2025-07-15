import os
import sys
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
import numpy as np
from urllib.parse import quote_plus
import json
from collections import defaultdict, deque
import hashlib

# Optional imports for enhanced features
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_ticker, get_paths, Config

# Import the simple archiver
from simple_archiver import archive_file

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=Config.ENV_PATH)
print(f"✅ Loaded environment variables from {Config.ENV_PATH}")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURE (Defaults) ---
# API Keys - Load from environment with clear status messages
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FMP_KEY = os.getenv("FMP_KEY", "")

# Twitter API Keys (v2)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Reddit API Keys
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sentiment_analyzer/1.0")

# Check API key availability
print("🔑 API Key Status:")
print(f"   Polygon API: {'✅ Available' if POLYGON_KEY else '❌ Missing'}")
print(f"   NewsAPI: {'✅ Available' if NEWSAPI_KEY else '❌ Missing'}")
print(f"   Alpha Vantage: {'✅ Available' if ALPHA_VANTAGE_KEY else '❌ Missing'}")
print(f"   FMP: {'✅ Available' if FMP_KEY else '❌ Missing'}")
print(f"   Twitter: {'✅ Available' if TWITTER_BEARER_TOKEN else '❌ Missing'}")
print(f"   Reddit: {'✅ Available' if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET else '❌ Missing'}")

if not POLYGON_KEY:
    print("⚠️  WARNING: No Polygon API key found. Only basic functionality will work.")
    print("   Add POLYGON_API_KEY=your_key_here to your env/.env file")

# Core Configuration
TICKER = get_ticker()  # Get from config
PATHS = get_paths()  # Get all paths from config
NEWS_DIR = PATHS["news_dir"]
OUTPUT_CSV = PATHS["sentiment_combined"]
LOOKBACK_DAYS = 7
BATCH_SIZE = 10  # For model inference
MAX_TEXT_LEN = 1028  # Truncate to avoid OOM
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls if paginating
LAST_FETCH_FILE = PATHS["last_fetch"]  # For incremental

# Enhanced Configuration
SENTIMENT_WINDOW_HOURS = 24  # Hours for velocity/momentum calculations
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for sentiment scores
SOCIAL_MEDIA_WEIGHT = 0.3  # Weight for social media vs news sentiment
NEWS_WEIGHT = 0.7

# --- Load Models (with device auto-detection) ---
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Log availability of optional dependencies
if not TWEEPY_AVAILABLE:
    logger.warning("tweepy not installed - Twitter functionality disabled")
if not PRAW_AVAILABLE:
    logger.warning("praw not installed - Reddit functionality disabled")
if not TEXTBLOB_AVAILABLE:
    logger.warning("textblob not installed - additional sentiment features disabled")
if not YFINANCE_AVAILABLE:
    logger.warning("yfinance not installed - company name lookup disabled")

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

# --- Enhanced News Source Fetchers ---

def fetch_newsapi_articles(ticker, start_date, end_date, api_key, limit=100):
    """Fetch news from NewsAPI.org"""
    if not api_key:
        logger.warning("NewsAPI key not provided, skipping NewsAPI")
        return []
    
    results = []
    url = "https://newsapi.org/v2/everything"
    
    # Create search query
    company_name = get_company_name(ticker)
    query = f'"{ticker}" OR "{company_name}" stock OR shares'
    
    params = {
        "q": query,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(limit, 100),
        "apiKey": api_key
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.error(f"NewsAPI error: {resp.status_code}")
            return []
        
        data = resp.json()
        articles = data.get("articles", [])
        
        for art in articles:
            pub_date = art.get("publishedAt")
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).astimezone(pytz.UTC)
                    results.append({
                        "datetime": int(dt.timestamp()),
                        "headline": art.get("title", ""),
                        "summary": art.get("description", ""),
                        "source": "newsapi",
                        "url": art.get("url", "")
                    })
                except Exception as e:
                    logger.warning(f"Date parse error for NewsAPI: {e}")
                    
    except Exception as e:
        logger.error(f"NewsAPI fetch error: {e}")
    
    logger.info(f"Fetched {len(results)} articles from NewsAPI")
    return results

def fetch_alpha_vantage_news(ticker, api_key, limit=50):
    """Fetch news from Alpha Vantage"""
    if not api_key:
        logger.warning("Alpha Vantage key not provided, skipping Alpha Vantage")
        return []
    
    results = []
    url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.error(f"Alpha Vantage error: {resp.status_code}")
            return []
        
        data = resp.json()
        articles = data.get("feed", [])
        
        for art in articles:
            pub_date = art.get("time_published")
            if pub_date:
                try:
                    # Alpha Vantage time format: YYYYMMDDTHHMMSS
                    dt = datetime.strptime(pub_date, "%Y%m%dT%H%M%S").replace(tzinfo=pytz.UTC)
                    results.append({
                        "datetime": int(dt.timestamp()),
                        "headline": art.get("title", ""),
                        "summary": art.get("summary", ""),
                        "source": "alphavantage",
                        "url": art.get("url", ""),
                        "overall_sentiment": art.get("overall_sentiment_label", "")
                    })
                except Exception as e:
                    logger.warning(f"Date parse error for Alpha Vantage: {e}")
                    
    except Exception as e:
        logger.error(f"Alpha Vantage fetch error: {e}")
    
    logger.info(f"Fetched {len(results)} articles from Alpha Vantage")
    return results

def fetch_fmp_news(ticker, api_key, limit=50):
    """Fetch news from Financial Modeling Prep"""
    if not api_key:
        logger.warning("FMP key not provided, skipping FMP")
        return []
    
    results = []
    url = f"https://financialmodelingprep.com/api/v3/stock_news"
    
    params = {
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key
    }
    
    try:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            logger.error(f"FMP error: {resp.status_code}")
            return []
        
        articles = resp.json()
        
        for art in articles:
            pub_date = art.get("publishedDate")
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).astimezone(pytz.UTC)
                    results.append({
                        "datetime": int(dt.timestamp()),
                        "headline": art.get("title", ""),
                        "summary": art.get("text", ""),
                        "source": "fmp",
                        "url": art.get("url", "")
                    })
                except Exception as e:
                    logger.warning(f"Date parse error for FMP: {e}")
                    
    except Exception as e:
        logger.error(f"FMP fetch error: {e}")
    
    logger.info(f"Fetched {len(results)} articles from FMP")
    return results

def get_company_name(ticker):
    """Get company name from ticker for better search queries"""
    if not YFINANCE_AVAILABLE:
        return ticker
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', ticker)
    except:
        return ticker

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

    logger.info(f"Fetched {len(results)} articles from Polygon")
    return results

# --- Social Media Sentiment Collection ---

def fetch_twitter_sentiment(ticker, start_date, end_date, bearer_token, limit=100):
    """Fetch Twitter sentiment using Twitter API v2"""
    if not TWEEPY_AVAILABLE:
        logger.warning("tweepy not installed, skipping Twitter")
        return []
        
    if not bearer_token:
        logger.warning("Twitter Bearer Token not provided, skipping Twitter")
        return []
    
    results = []
    try:
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        
        # Get company name for better search
        company_name = get_company_name(ticker)
        
        # Create search queries
        queries = [
            f"${ticker} lang:en -is:retweet",
            f'"{company_name}" stock lang:en -is:retweet',
            f'"{company_name}" share lang:en -is:retweet'
        ]
        
        for query in queries:
            try:
                tweets = tweepy.Paginator(
                    client.search_recent_tweets,
                    query=query,
                    start_time=start_date,
                    end_time=end_date,
                    tweet_fields=['created_at', 'public_metrics', 'author_id'],
                    max_results=min(limit // len(queries), 100)
                ).flatten(limit=limit // len(queries))
                
                for tweet in tweets:
                    if tweet.created_at:
                        dt = tweet.created_at.replace(tzinfo=pytz.UTC)
                        
                        # Calculate engagement weight based on metrics
                        metrics = tweet.public_metrics
                        engagement = (
                            metrics.get('like_count', 0) + 
                            metrics.get('retweet_count', 0) * 2 + 
                            metrics.get('reply_count', 0)
                        )
                        
                        results.append({
                            "datetime": int(dt.timestamp()),
                            "text": tweet.text,
                            "source": "twitter",
                            "engagement": engagement,
                            "tweet_id": tweet.id
                        })
                        
            except Exception as e:
                logger.warning(f"Twitter query error for '{query}': {e}")
                
    except Exception as e:
        logger.error(f"Twitter API error: {e}")
    
    logger.info(f"Fetched {len(results)} tweets")
    return results

def fetch_reddit_sentiment(ticker, start_unix, end_unix, client_id, client_secret, user_agent, limit=100):
    """Fetch Reddit sentiment from relevant subreddits"""
    if not PRAW_AVAILABLE:
        logger.warning("praw not installed, skipping Reddit")
        return []
        
    if not client_id or not client_secret:
        logger.warning("Reddit API credentials not provided, skipping Reddit")
        return []
    
    results = []
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Relevant subreddits for financial discussion
        subreddits = [
            'stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting',
            'StockMarket', 'financialindependence', 'options', 'wallstreetbets'
        ]
        
        company_name = get_company_name(ticker)
        search_terms = [ticker, company_name, f"${ticker}"]
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Search for posts
                for term in search_terms:
                    for submission in subreddit.search(term, time_filter="week", limit=limit//len(subreddits)//len(search_terms)):
                        if submission.created_utc >= start_unix and submission.created_utc <= end_unix:
                            
                            # Get post text
                            text = f"{submission.title} {submission.selftext}"
                            if len(text.strip()) < 10:  # Skip very short posts
                                continue
                            
                            results.append({
                                "datetime": int(submission.created_utc),
                                "text": text,
                                "source": f"reddit_{subreddit_name}",
                                "score": submission.score,
                                "num_comments": submission.num_comments,
                                "post_id": submission.id
                            })
                            
                        # Also check top-level comments
                        try:
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments[:5]:  # Top 5 comments
                                if (hasattr(comment, 'created_utc') and 
                                    comment.created_utc >= start_unix and 
                                    comment.created_utc <= end_unix and
                                    len(comment.body) > 20):
                                    
                                    results.append({
                                        "datetime": int(comment.created_utc),
                                        "text": comment.body,
                                        "source": f"reddit_{subreddit_name}_comment",
                                        "score": comment.score,
                                        "comment_id": comment.id
                                    })
                        except:
                            pass
                            
            except Exception as e:
                logger.warning(f"Reddit subreddit error for {subreddit_name}: {e}")
                
    except Exception as e:
        logger.error(f"Reddit API error: {e}")
    
    logger.info(f"Fetched {len(results)} Reddit posts/comments")
    return results

def deduplicate_content(articles):
    """Remove duplicate articles/posts based on content similarity"""
    seen_hashes = set()
    unique_articles = []
    
    for article in articles:
        # Create content hash from title + summary/text
        content = ""
        if "headline" in article:
            content += article.get("headline", "")
        if "summary" in article:
            content += article.get("summary", "")
        if "text" in article:
            content += article.get("text", "")
        
        # Normalize and hash
        content_normalized = re.sub(r'\s+', ' ', content.lower().strip())
        content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_articles.append(article)
    
    logger.info(f"Deduplicated {len(articles)} -> {len(unique_articles)} articles")
    return unique_articles

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

# --- Enhanced Main Logic ---
def main(start_date=None, end_date=None, incremental=True, enhanced=True):
    """Enhanced main function with multi-source sentiment analysis"""
    
    now_utc = datetime.now(pytz.UTC).replace(second=0, microsecond=0)
    
    # Determine time range
    if incremental and os.path.exists(LAST_FETCH_FILE):
        with open(LAST_FETCH_FILE, "r") as f:
            last_fetch = int(f.read().strip())
        start_unix = last_fetch + 1
    else:
        start_unix = int((now_utc - timedelta(days=LOOKBACK_DAYS)).timestamp() - 1)

    if start_date:
        start_unix = int(datetime.fromisoformat(start_date).timestamp())
    end_unix = int(now_utc.timestamp()) if not end_date else int(datetime.fromisoformat(end_date).timestamp())
    
    start_dt = datetime.utcfromtimestamp(start_unix).replace(tzinfo=pytz.UTC)
    end_dt = datetime.utcfromtimestamp(end_unix).replace(tzinfo=pytz.UTC)

    logger.info(f"Fetching sentiment data for {TICKER} from {start_dt} to {end_dt}")
    
    # Collect data from all sources
    all_items = []
    
    # 1. Polygon News (existing)
    logger.info("Fetching from Polygon...")
    polygon_articles = fetch_polygon_news(TICKER, start_unix, end_unix, POLYGON_KEY)
    all_items.extend(polygon_articles)
    
    # 2. NewsAPI
    logger.info("Fetching from NewsAPI...")
    newsapi_articles = fetch_newsapi_articles(TICKER, start_dt, end_dt, NEWSAPI_KEY)
    all_items.extend(newsapi_articles)
    
    # 3. Alpha Vantage
    logger.info("Fetching from Alpha Vantage...")
    alpha_articles = fetch_alpha_vantage_news(TICKER, ALPHA_VANTAGE_KEY)
    all_items.extend(alpha_articles)
    
    # 4. Financial Modeling Prep
    logger.info("Fetching from FMP...")
    fmp_articles = fetch_fmp_news(TICKER, FMP_KEY)
    all_items.extend(fmp_articles)
    
    # 5. Twitter
    logger.info("Fetching from Twitter...")
    twitter_posts = fetch_twitter_sentiment(TICKER, start_dt, end_dt, TWITTER_BEARER_TOKEN)
    all_items.extend(twitter_posts)
    
    # 6. Reddit
    logger.info("Fetching from Reddit...")
    reddit_posts = fetch_reddit_sentiment(TICKER, start_unix, end_unix, 
                                         REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
    all_items.extend(reddit_posts)
    
    if not all_items:
        logger.info("No new content found across all sources.")
        return

    logger.info(f"Total items collected: {len(all_items)}")
    
    # Deduplicate content
    all_items = deduplicate_content(all_items)
    
    # Filter by time range and prepare for processing
    rows = []
    items_batch = []
    timestamps_batch = []

    for item in all_items:
        dt = datetime.utcfromtimestamp(item.get("datetime", 0)).replace(tzinfo=pytz.UTC)
        if dt.timestamp() < start_unix:
            continue

        items_batch.append(item)
        timestamps_batch.append(dt.isoformat())

        if len(items_batch) == BATCH_SIZE:
            if enhanced:
                process_enhanced_batch(items_batch, timestamps_batch, rows)
            else:
                # Legacy processing for backward compatibility
                texts = []
                timestamps_legacy = []
                for i, item in enumerate(items_batch):
                    if 'headline' in item:
                        text = clean_text(f"{item.get('headline', '')} {item.get('summary', '')}")
                    else:
                        text = clean_text(item.get('text', ''))
                    if text:
                        texts.append(text)
                        timestamps_legacy.append(timestamps_batch[i])
                
                if texts:
                    process_batch(texts, timestamps_legacy, rows)
            
            items_batch, timestamps_batch = [], []

    # Process remaining items
    if items_batch:
        if enhanced:
            process_enhanced_batch(items_batch, timestamps_batch, rows)
        else:
            texts = []
            timestamps_legacy = []
            for i, item in enumerate(items_batch):
                if 'headline' in item:
                    text = clean_text(f"{item.get('headline', '')} {item.get('summary', '')}")
                else:
                    text = clean_text(item.get('text', ''))
                if text:
                    texts.append(text)
                    timestamps_legacy.append(timestamps_batch[i])
            
            if texts:
                process_batch(texts, timestamps_legacy, rows)

    if rows:
        df_out = pd.DataFrame(rows)
        df_out["Datetime"] = pd.to_datetime(df_out["Datetime"])
        
        # Enhanced aggregation with momentum calculations
        if enhanced and 'Prob_Pos_Weighted' in df_out.columns:
            # Aggregate by minute with weighted averages
            numeric_cols = [col for col in df_out.columns if col.startswith('Prob_') or col in ['Engagement_Weight', 'Total_Weight', 'Sentiment_Confidence']]
            
            df_agg = df_out.groupby("Datetime", as_index=False).agg({
                **{col: 'mean' for col in numeric_cols},
                'Ticker': 'first',
                'Source': lambda x: '|'.join(set(x)),  # Combine sources
                'Source_Type': lambda x: '|'.join(set(x))  # Combine source types
            })
            
            # Calculate momentum and velocity metrics
            if len(df_agg) > 1:
                df_agg = calculate_sentiment_momentum(df_agg)
        else:
            # Legacy aggregation
            df_agg = df_out.groupby("Datetime", as_index=False).mean(numeric_only=True)
            df_agg['Ticker'] = TICKER

        os.makedirs(NEWS_DIR, exist_ok=True)
        
        # Merge with existing data
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV, parse_dates=["Datetime"])
            df_final = pd.concat([df_existing, df_agg]).drop_duplicates(subset=["Datetime"], keep='last').sort_values("Datetime")
        else:
            df_final = df_agg
        
        df_final.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved {len(df_final)} sentiment entries (including existing) to {OUTPUT_CSV}")
        
        # Archive the sentiment file we just created (ONE simple line added!)
        archive_file(OUTPUT_CSV)
        
        # Log summary statistics
        if enhanced and 'Prob_Pos_Weighted' in df_final.columns:
            avg_sentiment = (df_final['Prob_Pos_Weighted'] - df_final['Prob_Neg_Weighted']).mean()
            logger.info(f"Average sentiment score: {avg_sentiment:.3f}")
            
            if 'Sentiment_Velocity_1h' in df_final.columns:
                avg_velocity = df_final['Sentiment_Velocity_1h'].mean()
                logger.info(f"Average sentiment velocity (1h): {avg_velocity:.3f}")

        # Update last fetch timestamp
        with open(LAST_FETCH_FILE, "w") as f:
            f.write(str(end_unix))

# --- Enhanced Sentiment Analysis Functions ---

def calculate_weighted_sentiment(sentiment_scores, source_type, engagement_weight=1.0):
    """Calculate weighted sentiment score based on source type and engagement"""
    
    # Define weights for different models based on their financial relevance
    model_weights = {
        'finbert': 0.4,      # Financial domain-specific
        'roberta': 0.35,     # Good for social media
        'distilbert': 0.25   # General sentiment
    }
    
    # Source type weights
    source_weights = {
        'news': NEWS_WEIGHT,
        'social': SOCIAL_MEDIA_WEIGHT
    }
    
    # Calculate weighted average
    pos_score = (
        sentiment_scores['finbert'][2] * model_weights['finbert'] +
        sentiment_scores['roberta'][2] * model_weights['roberta'] +
        sentiment_scores['distilbert'][2] * model_weights['distilbert']
    )
    
    neu_score = (
        sentiment_scores['finbert'][1] * model_weights['finbert'] +
        sentiment_scores['roberta'][1] * model_weights['roberta'] +
        sentiment_scores['distilbert'][1] * model_weights['distilbert']
    )
    
    neg_score = (
        sentiment_scores['finbert'][0] * model_weights['finbert'] +
        sentiment_scores['roberta'][0] * model_weights['roberta'] +
        sentiment_scores['distilbert'][0] * model_weights['distilbert']
    )
    
    # Apply source weight and engagement weight
    base_weight = source_weights.get(source_type, 1.0)
    total_weight = base_weight * engagement_weight
    
    return {
        'pos': pos_score * total_weight,
        'neu': neu_score * total_weight,
        'neg': neg_score * total_weight,
        'confidence': max(pos_score, neu_score, neg_score),
        'weight': total_weight
    }

def calculate_sentiment_momentum(df, hours=24):
    """Calculate sentiment velocity and momentum over specified time window"""
    
    if len(df) < 2:
        return df
    
    df = df.sort_values('Datetime').copy()
    df['Sentiment_Score'] = df['Prob_Pos_Weighted'] - df['Prob_Neg_Weighted']
    
    # Calculate rolling averages
    df['Sentiment_MA_6h'] = df['Sentiment_Score'].rolling(window=360, min_periods=1).mean()  # 6 hours (360 minutes)
    df['Sentiment_MA_24h'] = df['Sentiment_Score'].rolling(window=1440, min_periods=1).mean()  # 24 hours
    
    # Calculate velocity (rate of change)
    df['Sentiment_Velocity_1h'] = df['Sentiment_Score'].diff(60)  # Change over 1 hour
    df['Sentiment_Velocity_6h'] = df['Sentiment_Score'].diff(360)  # Change over 6 hours
    
    # Calculate acceleration (second derivative)
    df['Sentiment_Acceleration'] = df['Sentiment_Velocity_1h'].diff()
    
    # Calculate momentum (velocity * magnitude)
    df['Sentiment_Momentum'] = df['Sentiment_Velocity_1h'] * abs(df['Sentiment_Score'])
    
    # Calculate volatility
    df['Sentiment_Volatility'] = df['Sentiment_Score'].rolling(window=180, min_periods=1).std()
    
    return df

def calculate_engagement_weight(item):
    """Calculate engagement weight for social media posts"""
    
    if item.get('source') == 'twitter':
        engagement = item.get('engagement', 0)
        # Normalize engagement score (log scale to prevent outliers from dominating)
        return min(1.0 + np.log1p(engagement) / 10, 3.0)
    
    elif 'reddit' in item.get('source', ''):
        score = item.get('score', 0)
        num_comments = item.get('num_comments', 0)
        # Reddit engagement weight
        engagement = score + num_comments * 2
        return min(1.0 + np.log1p(max(0, engagement)) / 5, 2.5)
    
    return 1.0  # Default weight for news

def process_enhanced_batch(items, timestamps, rows):
    """Enhanced batch processing with weighted sentiment and momentum"""
    try:
        # Separate text content for model processing
        texts = []
        item_metadata = []
        
        for i, item in enumerate(items):
            # Extract text content
            if 'headline' in item:
                text = clean_text(f"{item.get('headline', '')} {item.get('summary', '')}")
            else:
                text = clean_text(item.get('text', ''))
            
            if not text:
                continue
                
            texts.append(text)
            
            # Calculate engagement weight
            engagement_weight = calculate_engagement_weight(item)
            
            # Determine source type
            source = item.get('source', '')
            source_type = 'social' if any(s in source for s in ['twitter', 'reddit']) else 'news'
            
            item_metadata.append({
                'engagement_weight': engagement_weight,
                'source_type': source_type,
                'source': source,
                'timestamp': timestamps[i]
            })
        
        if not texts:
            return
        
        # Run sentiment analysis on all models
        probs_fin = classify_finbert(texts)
        probs_rob = classify_roberta(texts)
        probs_dis = classify_distilbert(texts)

        # Process each item with enhanced sentiment scoring
        for i in range(len(texts)):
            metadata = item_metadata[i]
            
            # Combine model outputs
            sentiment_scores = {
                'finbert': probs_fin[i],
                'roberta': probs_rob[i],
                'distilbert': probs_dis[i]
            }
            
            # Calculate weighted sentiment
            weighted_sentiment = calculate_weighted_sentiment(
                sentiment_scores,
                metadata['source_type'],
                metadata['engagement_weight']
            )
            
            # Only include high-confidence predictions
            if weighted_sentiment['confidence'] >= MIN_CONFIDENCE_THRESHOLD:
                rows.append({
                    "Datetime": metadata['timestamp'],
                    "Ticker": TICKER,
                    "Source": metadata['source'],
                    "Source_Type": metadata['source_type'],
                    "Engagement_Weight": metadata['engagement_weight'],
                    
                    # Original model outputs (for compatibility)
                    "Prob_Pos_finbert": probs_fin[i][2],
                    "Prob_Neu_finbert": probs_fin[i][1],
                    "Prob_Neg_finbert": probs_fin[i][0],
                    "Prob_Pos_roberta": probs_rob[i][2],
                    "Prob_Neu_roberta": probs_rob[i][1],
                    "Prob_Neg_roberta": probs_rob[i][0],
                    "Prob_Pos_distilbert": probs_dis[i][2],
                    "Prob_Neu_distilbert": probs_dis[i][1],
                    "Prob_Neg_distilbert": probs_dis[i][0],
                    
                    # Enhanced weighted scores
                    "Prob_Pos_Weighted": weighted_sentiment['pos'],
                    "Prob_Neu_Weighted": weighted_sentiment['neu'],
                    "Prob_Neg_Weighted": weighted_sentiment['neg'],
                    "Sentiment_Confidence": weighted_sentiment['confidence'],
                    "Total_Weight": weighted_sentiment['weight']
                })
                
    except Exception as e:
        logger.error(f"Enhanced batch processing error: {e}")

# Legacy function for backward compatibility
def process_batch(texts, timestamps, rows):
    """Legacy batch processing function for backward compatibility"""
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
    parser = argparse.ArgumentParser(description="Enhanced Multi-Source Sentiment Analysis")
    parser.add_argument("--start", type=str, help="Start date (ISO8601)")
    parser.add_argument("--end", type=str, help="End date (ISO8601)")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--no-incremental", action="store_false", dest="incremental", help="Disable incremental fetch")
    parser.add_argument("--legacy", action="store_true", help="Use legacy processing (backward compatibility)")
    parser.add_argument("--sources", type=str, nargs='+', 
                       choices=['polygon', 'newsapi', 'alphavantage', 'fmp', 'twitter', 'reddit', 'all'],
                       default=['all'], help="Specify which sources to use")
    args = parser.parse_args()
    
    # Update ticker if provided
    if args.ticker:
        from config import set_ticker
        set_ticker(args.ticker)
        # Update global variables
        TICKER = get_ticker()
        PATHS = get_paths()
        NEWS_DIR = PATHS["news_dir"]
        OUTPUT_CSV = PATHS["sentiment_combined"]
        LAST_FETCH_FILE = PATHS["last_fetch"]
    
    # Use enhanced processing unless legacy is specified
    enhanced_mode = not args.legacy
    
    logger.info(f"Starting sentiment analysis in {'Enhanced' if enhanced_mode else 'Legacy'} mode")
    logger.info(f"Sources: {args.sources}")
    
    main(args.start, args.end, args.incremental, enhanced_mode)