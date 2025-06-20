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


    
def merge_with_price(sentiment_df, price_path, output_path):
    """
    Merges upsampled sentiment data (1-minute resolution) with minute-level price data.

    Args:
        sentiment_df (pd.DataFrame): Aggregated sentiment with datetime column.
        price_path (str): Path to price/technical indicator CSV.
        output_path (str): File to write merged dataset to.
    """
    # --- Load and prep price data ---
    price_df = pd.read_csv(price_path)
    date_col = next((col for col in price_df.columns if "date" in col.lower() or "time" in col.lower()), None)
    if not date_col:
        raise ValueError("No datetime column found in price CSV.")

    price_df[date_col] = pd.to_datetime(price_df[date_col])
    price_df.rename(columns={date_col: "Datetime"}, inplace=True)
    price_df["Datetime"] = price_df["Datetime"].dt.tz_localize(None)

    # --- Prep and upsample sentiment ---
    sentiment_df["Datetime"] = pd.to_datetime(sentiment_df["Datetime"])
    sentiment_df["Datetime"] = sentiment_df["Datetime"].dt.tz_localize(None)
    sentiment_df = sentiment_df.set_index("Datetime").resample("1min").ffill().reset_index()

    # --- Merge ---
    df = pd.merge(price_df, sentiment_df, on="Datetime", how="left")

    # --- Add return + label ---
    df["Return"] = df["Close"].pct_change().shift(-1)
    df["Label"] = (df["Return"] > 0).astype(int)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Merged minute-level dataset saved to {output_path}")
def merge_with_price_incremental(sentiment_df, price_path, output_path, last_timestamp=None):
    """
    Incrementally merges 1min price and sentiment data, forward-filling sentiment.
    Appends only new rows to `output_path`.
    """

    # --- Load price data ---
    price_df = pd.read_csv(price_path)
    date_col = next((c for c in price_df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if not date_col:
        raise ValueError("No datetime column found in price CSV.")
    price_df[date_col] = pd.to_datetime(price_df[date_col], utc=True)
    price_df.rename(columns={date_col: "Datetime"}, inplace=True)

    # --- Load sentiment and standardize timezone ---
    sentiment_df["Datetime"] = pd.to_datetime(sentiment_df["Datetime"], utc=True)
    sentiment_df = sentiment_df.set_index("Datetime")

    # --- If previous full dataset exists, get last timestamp ---
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, parse_dates=["Datetime"])
        existing_df["Datetime"] = pd.to_datetime(existing_df["Datetime"], utc=True)
        if last_timestamp is None:
            last_timestamp = existing_df["Datetime"].max()
        price_df = price_df[price_df["Datetime"] > last_timestamp]
    else:
        existing_df = None

    if price_df.empty:
        print("⏸️ No new price data to merge.")

        # Create full timestamp range based on price_df
    full_range = pd.date_range(start=price_df["Datetime"].min(),
                            end=price_df["Datetime"].max(),
                            freq="1min",
                            tz="UTC")

    # Reindex sentiment_df to include all timestamps — then forward-fill
    sentiment_filled = sentiment_df.set_index("Datetime").reindex(full_range).ffill()

    # Only keep timestamps that exist in price_df (ensures clean merge)
    sentiment_filled = sentiment_filled.loc[price_df["Datetime"]].reset_index()
    sentiment_filled.rename(columns={"index": "Datetime"}, inplace=True)



    # --- Merge and compute labels ---
    merged = pd.merge(price_df, sentiment_filled, on="Datetime", how="inner")
    merged["Return"] = merged["Close"].pct_change().shift(-1)
    merged["Label"] = (merged["Return"] > 0).astype(int)

    # --- Append to full dataset ---
    if existing_df is not None:
        final_df = pd.concat([existing_df, merged]).drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
    else:
        final_df = merged

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Appended {len(merged)} new rows → {output_path}")

