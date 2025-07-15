# scripts/combineSingleDB.py

import os
import pandas as pd

def load_and_aggregate_sentiment(csv_path, resample_interval='1min'):
    """
    Loads sentiment CSV, resamples to 1-min, forward-fills, computes features.
    Returns aggregated sentiment DataFrame with continuous 1-min UTC 'Datetime'.
    """
    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("Sentiment file is empty.")
    if "Datetime" not in df.columns:
        raise ValueError("Missing 'Datetime' column.")

    df = df.groupby("Datetime", as_index=True).mean()
    df = df.resample(resample_interval).ffill()

    sentiment_cols = [col for col in df.columns if col.startswith("Prob_Pos_") or col.startswith("Prob_Neu_") or col.startswith("Prob_Neg_")]
    df = df[df[sentiment_cols].sum(axis=1) > 0]

    prob_pos_cols = [col for col in df.columns if "Prob_Pos_" in col]
    df["Sentiment_Avg"] = df[prob_pos_cols].mean(axis=1)
    df["Sentiment_Disagreement"] = df[prob_pos_cols].std(axis=1)

    return df.reset_index()

def full_merge(sentiment_df, price_path, output_path=None):
    """
    Merges full price and sentiment datasets, matching on 1-min UTC 'Datetime'.
    Forward-fills sentiment (even if sparse) to all price bars.
    """
    price_df = pd.read_csv(price_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'], utc=True, errors='coerce')
    price_df = price_df.dropna(subset=['Datetime'])
    price_df = price_df.sort_values('Datetime').drop_duplicates('Datetime')

    sentiment_df['Datetime'] = pd.to_datetime(sentiment_df['Datetime'], utc=True, errors='coerce')
    sentiment_df = sentiment_df.dropna(subset=['Datetime'])
    sentiment_df = sentiment_df.sort_values('Datetime').drop_duplicates('Datetime')

    # Robust forward fill of sentiment onto ALL price bars
    sentiment_ff = sentiment_df.set_index('Datetime').reindex(price_df['Datetime']).ffill().reset_index()
    sentiment_ff.rename(columns={'index': 'Datetime'}, inplace=True)

    merged = price_df.merge(sentiment_ff, on='Datetime', how='left', suffixes=('', '_sent'))

    merged['Return'] = merged['Close'].pct_change().shift(-1)
    merged['Label'] = (merged['Return'] > 0).astype(int)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✅ Saved merged data to {output_path}")

    return merged

def full_incremental_merge(sentiment_df, price_path, output_path):
    """
    Incrementally append new price/sentiment rows to an existing merged file.
    Only rows with price['Datetime'] > last in output_path are processed.
    Sentiment is robustly forward-filled even for irregular intervals.
    """
    price_df = pd.read_csv(price_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'], utc=True, errors='coerce')
    price_df = price_df.dropna(subset=['Datetime'])
    price_df = price_df.sort_values('Datetime').drop_duplicates('Datetime')

    sentiment_df['Datetime'] = pd.to_datetime(sentiment_df['Datetime'], utc=True, errors='coerce')
    sentiment_df = sentiment_df.dropna(subset=['Datetime'])
    sentiment_df = sentiment_df.sort_values('Datetime').drop_duplicates('Datetime')

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df['Datetime'] = pd.to_datetime(existing_df['Datetime'], utc=True, errors='coerce')
        last_ts = existing_df['Datetime'].max()
        new_price_df = price_df[price_df['Datetime'] > last_ts]
    else:
        existing_df = None
        new_price_df = price_df.copy()

    if new_price_df.empty:
        print("⏸️ No new price data to merge.")
        return existing_df if existing_df is not None else None

    # Forward-fill sentiment for all new price times
    sentiment_ff = sentiment_df.set_index('Datetime').reindex(new_price_df['Datetime']).ffill().reset_index()
    sentiment_ff.rename(columns={'index': 'Datetime'}, inplace=True)

    merged = new_price_df.merge(sentiment_ff, on='Datetime', how='left', suffixes=('', '_sent'))
    merged['Return'] = merged['Close'].pct_change().shift(-1)
    merged['Label'] = (merged['Return'] > 0).astype(int)

    if existing_df is not None:
        final_df = pd.concat([existing_df, merged], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['Datetime'], keep='last').sort_values('Datetime')
    else:
        final_df = merged

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Appended {len(merged)} new rows → {output_path}")
    return final_df
