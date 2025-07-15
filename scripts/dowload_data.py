import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import argparse

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_ticker, get_paths, Config

# Import the simple archiver
from simple_archiver import archive_file

# --- CONFIGURE ---
TICKER = get_ticker()        # Get from config
TICKERS = [TICKER]           # list of ETFs or stocks
INTERVAL = Config.INTERVAL   # 1-minute bars
LOOKBACK_DAYS = Config.LOOKBACK_DAYS  # how many days back to fetch
PATHS = get_paths()          # Get all paths from config

def download_1min_bars(ticker, start, end):
    df = yf.download(
        tickers=ticker,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        interval=INTERVAL,
        progress=False
    )
    df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download stock price data")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Days of historical data to fetch")
    args = parser.parse_args()
    
    # Update ticker if provided
    if args.ticker:
        from config import set_ticker
        set_ticker(args.ticker)
        TICKER = get_ticker()
        TICKERS = [TICKER]
        PATHS = get_paths()
    
    # Create raw data directory
    raw_dir = os.path.dirname(PATHS["raw_csv"])
    os.makedirs(raw_dir, exist_ok=True)
    
    end_date = datetime.now() + timedelta(days=1)  # make end inclusive
    start_date = end_date - timedelta(days=args.days + 1)
    
    for ticker in TICKERS:
        df = download_1min_bars(ticker, start_date, end_date)
        df.index = df.index.tz_convert("UTC")  # ensure consistency with sentiment data
        
        # Use dynamic path from config
        out_path = get_paths(ticker)["raw_csv"]
        df.to_csv(out_path)
        print(f'Saved raw data for {ticker} → {out_path}')
        
        # Archive the raw data file we just created (ONE simple line added!)
        archive_file(out_path)