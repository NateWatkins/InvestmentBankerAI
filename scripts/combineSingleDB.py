# scripts/combineSingleDB.py

import os
import pandas as pd

def load_and_aggregate_sentiment(csv_path, resample_interval='1min'):

    df = pd.read_csv(csv_path, parse_dates=["Datetime"])
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("Sentiment file is empty.")
    if "Datetime" not in df.columns:
        raise ValueError("Missing 'Datetime' column.")

    # 🧠 Group by timestamp and average duplicates
    df = df.groupby("Datetime", as_index=True).mean()

    # 🕒 Resample to 1-min intervals
    df = df.resample(resample_interval).ffill()

    # 📉 Drop rows where all sentiment scores are 0
    sentiment_cols = [col for col in df.columns if col.startswith("Prob_Pos_") or col.startswith("Prob_Neu_") or col.startswith("Prob_Neg_")]
    df = df[df[sentiment_cols].sum(axis=1) > 0]

    # 🧠 Compute aggregate features
    prob_pos_cols = [col for col in df.columns if "Prob_Pos_" in col]
    df["Sentiment_Avg"] = df[prob_pos_cols].mean(axis=1)
    df["Sentiment_Disagreement"] = df[prob_pos_cols].std(axis=1)

    return df.reset_index()


    
import pandas as pd
import os

def full_merge(sentiment_df, price_path, output_path=None):
    """
    Merge full price and sentiment datasets, matching on 1-min UTC 'Datetime'.
    Forward-fills sentiment. Price is always the anchor.
    """
    # --- Load and standardize price data ---
    price_df = pd.read_csv(price_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'], utc=True, errors='coerce')
    price_df = price_df.dropna(subset=['Datetime'])
    price_df = price_df.sort_values('Datetime').drop_duplicates('Datetime')

    # --- Standardize sentiment data ---
    sentiment_df['Datetime'] = pd.to_datetime(sentiment_df['Datetime'], utc=True, errors='coerce')
    sentiment_df = sentiment_df.dropna(subset=['Datetime'])
    sentiment_df = sentiment_df.sort_values('Datetime').drop_duplicates('Datetime')

    # --- Forward-fill sentiment for all price times ---
    sentiment_ff = sentiment_df.set_index('Datetime').reindex(price_df['Datetime']).ffill().reset_index()
    sentiment_ff.rename(columns={'index': 'Datetime'}, inplace=True)  # Defensive; rarely needed

    # --- Merge price and sentiment ---
    merged = price_df.merge(sentiment_ff, on='Datetime', how='left', suffixes=('', '_sent'))

    # --- Add label columns ---
    merged['Return'] = merged['Close'].pct_change().shift(-1)
    merged['Label'] = (merged['Return'] > 0).astype(int)

    # --- Save and return ---
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"✅ Saved merged data to {output_path}")

    return merged



def full_incremental_merge(sentiment_df, price_path, output_path):
    """
    Incrementally append new price/sentiment rows to an existing merged file.
    Only rows with price['Datetime'] > last in output_path are processed.
    """
    # --- Load and standardize price data ---
    price_df = pd.read_csv(price_path)
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'], utc=True, errors='coerce')
    price_df = price_df.dropna(subset=['Datetime'])
    price_df = price_df.sort_values('Datetime').drop_duplicates('Datetime')

    # --- Standardize sentiment data ---
    sentiment_df['Datetime'] = pd.to_datetime(sentiment_df['Datetime'], utc=True, errors='coerce')
    sentiment_df = sentiment_df.dropna(subset=['Datetime'])
    sentiment_df = sentiment_df.sort_values('Datetime').drop_duplicates('Datetime')

    # --- Load existing output, if any ---
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df['Datetime'] = pd.to_datetime(existing_df['Datetime'], utc=True, errors='coerce')
        last_ts = existing_df['Datetime'].max()
        # Only keep new price data
        new_price_df = price_df[price_df['Datetime'] > last_ts]
    else:
        existing_df = None
        new_price_df = price_df.copy()

    if new_price_df.empty:
        print("⏸️ No new price data to merge.")
        return existing_df if existing_df is not None else None

    # --- Forward-fill sentiment for new price times ---
    sentiment_ff = sentiment_df.set_index('Datetime').reindex(new_price_df['Datetime']).ffill().reset_index()
    sentiment_ff.rename(columns={'index': 'Datetime'}, inplace=True)

    # --- Merge new price and sentiment ---
    merged = new_price_df.merge(sentiment_ff, on='Datetime', how='left', suffixes=('', '_sent'))

    # --- Add label columns ---
    merged['Return'] = merged['Close'].pct_change().shift(-1)
    merged['Label'] = (merged['Return'] > 0).astype(int)

    # --- Concatenate with previous, dedupe, sort ---
    if existing_df is not None:
        final_df = pd.concat([existing_df, merged], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['Datetime']).sort_values('Datetime')
    else:
        final_df = merged

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Appended {len(merged)} new rows → {output_path}")
    return final_df
