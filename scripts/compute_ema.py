import pandas as pd
import os
import sys
import argparse

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_ticker, get_paths, Config

# --- CONFIGURE ---
TICKER = get_ticker()  # Get from config
TICKERS = [TICKER]
PATHS = get_paths()    # Get all paths from config
RAW_DIR = os.path.dirname(PATHS["raw_csv"])
FEATURE_DIR = os.path.dirname(PATHS["features_csv"])
EMA_PERIOD = 20

def compute_ema(df, period):
    # Ensure 'Close' is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    ema_col = f'EMA{period}'
    df[ema_col] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute technical indicators")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--ema-period", type=int, default=EMA_PERIOD, help="EMA period")
    args = parser.parse_args()
    
    # Update ticker if provided
    if args.ticker:
        from config import set_ticker
        set_ticker(args.ticker)
        TICKER = get_ticker()
        TICKERS = [TICKER]
        PATHS = get_paths()
        RAW_DIR = os.path.dirname(PATHS["raw_csv"])
        FEATURE_DIR = os.path.dirname(PATHS["features_csv"])
    
    os.makedirs(FEATURE_DIR, exist_ok=True)

    for ticker in TICKERS:
        # Use dynamic path from config
        raw_path = get_paths(ticker)["raw_csv"]

        # Read CSV by skipping the first two rows, then assigning our own column names:
        df = pd.read_csv(
            raw_path,
            skiprows=2,
            names=['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume'],
            parse_dates=['Datetime']
        )

        if "Datetime" not in df.columns:
            raise KeyError("Column 'Datetime' not found in file after parsing.")

        # Now compute the EMA on the 'Close' column
        df_feat = compute_ema(df, args.ema_period)
        df_feat = df_feat.reset_index() if df_feat.index.name == "Datetime" else df_feat  # Ensures 'Datetime' is a column

        # Use dynamic path from config
        out_path = get_paths(ticker)["features_csv"]
        df_feat.to_csv(out_path, index=False)  # index=False so 'Datetime' is a column, not the index
        print(f'Saved features for {ticker} → {out_path}')

