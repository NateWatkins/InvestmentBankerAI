import pandas as pd
import os

# --- CONFIGURE ---
TICKERS = ['TSLA']
RAW_DIR = "/Users/natwat/Desktop/CPSC_Projects/Trader/data/raw"
FEATURE_DIR = '/Users/natwat/Desktop/CPSC_Projects/Trader/data/features'
EMA_PERIOD = 20

def compute_ema(df, period):
    # Ensure 'Close' is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    ema_col = f'EMA{period}'
    df[ema_col] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

if __name__ == '__main__':
    os.makedirs(FEATURE_DIR, exist_ok=True)

    for ticker in TICKERS:
        raw_path = os.path.join(RAW_DIR, f'{ticker}_raw.csv')

        # Read CSV by skipping the first two rows, then assigning our own column names:
        df = pd.read_csv(
            raw_path,
            skiprows=2,
            names=['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume'],
            parse_dates=['Datetime']
        )

        if "Datetime" not in df.columns:
            raise KeyError("Column 'Datetime' not found in file after parsing.")

        df.set_index("Datetime", inplace=True)


        # Now compute the EMA on the 'Close' column
        df_feat = compute_ema(df, EMA_PERIOD)

        out_path = os.path.join(FEATURE_DIR, f'{ticker}_features.csv')
        df_feat.to_csv(out_path)
        print(f'Saved features for {ticker} → {out_path}')
