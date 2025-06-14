import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# --- CONFIGURE ---
TICKERS = ['TSLA']            # list of ETFs or stocks
INTERVAL = '1m'              # 1-minute bars
LOOKBACK_DAYS = 5            # how many days back to fetch

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
    os.makedirs("/Users/natwat/Desktop/CPSC_Projects/Trader/data/raw", exist_ok=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    for ticker in TICKERS:
        df = download_1min_bars(ticker, start_date, end_date)
        # ensure directory exists
        out_path = os.path.join("/Users/natwat/Desktop/CPSC_Projects/Trader/data/raw", f'{ticker}_raw.csv')
        df.to_csv(out_path)
        print(f'Saved raw data for {ticker} → {out_path}')