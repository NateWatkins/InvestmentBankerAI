# src/data_pipeline.py
import os
import pandas as pd

def load_trading_dataframe(
    ticker: str = "TSLA",
    raw_dir: str = "data/raw",
    feature_dir: str = "data/features",
    news_dir: str = "data/news",
    include_raw: bool = False,
) -> pd.DataFrame:
    """
    Load all available data for `ticker` and return a merged DataFrame.
    The function is designed so you can easily plug in more feature files later.

    Parameters
    ----------
    ticker : str
        Symbol to load (e.g., "SPY").
    raw_dir, feature_dir, news_dir : str
        Directories where raw bars, technical features, and sentiment CSVs live.
    include_raw : bool
        If True, include the original OHLCV columns from the raw file.

    Returns
    -------
    pd.DataFrame
        Combined data indexed by Datetime.
    """
    frames = []

    # Feature file (e.g. EMA columns)
    feat_path = "/Users/natwat/Desktop/CPSC_Projects/Trader/data/features"
    df_feat = pd.read_csv(feat_path, parse_dates=["Datetime"], index_col="Datetime")
    frames.append(df_feat)

    # Optional raw OHLCV data
    if include_raw:
        raw_path = os.path.join(raw_dir, f"{ticker}_raw.csv")
        df_raw = pd.read_csv(
            raw_path,
            skiprows=2,
            names=["Datetime", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Datetime"],
            index_col="Datetime",
            infer_datetime_format=True,
        )
        frames.insert(0, df_raw)

    # Optional sentiment data (if it exists)
    sent_path = os.path.join(news_dir, f"{ticker}_sentiment.csv")
    if os.path.exists(sent_path):
        df_sent = pd.read_csv(sent_path, parse_dates=["Datetime"], index_col="Datetime")
        df_sent.rename(columns={"Sentiment": "NewsSentiment"}, inplace=True)
        frames.append(df_sent)

    # Combine all available pieces
    df = pd.concat(frames, axis=1).sort_index()
    df.dropna(inplace=True)
    return df