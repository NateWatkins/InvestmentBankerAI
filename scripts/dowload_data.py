import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# --- CONFIGURE ---
TICKERS = ['TSLA']            # list of ETFs or stocks
INTERVAL = '1m'              # 1-minute bars
LOOKBACK_DAYS = 7        # how many days back to fetch

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
    os.makedirs("/Users/natwat/Desktop/CPSC_Projects/INVBANKAI/data/raw", exist_ok=True)
    end_date = datetime.now() + timedelta(days=1)  # make end inclusive
    start_date = end_date - timedelta(days=LOOKBACK_DAYS + 1)
    for ticker in TICKERS:
        df = download_1min_bars(ticker, start_date, end_date)
        df.index = df.index.tz_convert("UTC")  # ensure consistency with sentiment data
        # ensure directory exists
        out_path = os.path.join("/Users/natwat/Desktop/CPSC_Projects/INVBANKAI/data/raw", f'{ticker}_raw.csv')
        df.to_csv(out_path)
        print(f'Saved raw data for {ticker} → {out_path}')