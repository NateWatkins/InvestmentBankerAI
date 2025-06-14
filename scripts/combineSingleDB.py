# scripts/combineSingleDB.py

import os
import pandas as pd

def load_and_aggregate_sentiment(news_dir, resample_interval='1min'):
    """
    Aggregates sentiment CSVs from multiple models.

    Returns:
        pd.DataFrame: Aggregated sentiment features.
    """
    files = [f for f in os.listdir(news_dir) if f.endswith('_sentiment.csv')]
    model_dfs = []

    for file in files:
        model_name = file.replace("TSLA_", "").replace("_sentiment.csv", "")
        path = os.path.join(news_dir, file)

        df = pd.read_csv(path, parse_dates=["Datetime"])
        df.columns = df.columns.str.strip()
        if df.empty:
            print(f"⚠️ Skipping empty file: {file}")
            continue
        if not set(["Prob_Positive", "Prob_Neutral", "Prob_Negative"]).issubset(df.columns):
            print(f"⚠️ Missing sentiment columns in {file}, skipping.")
            continue

        df.set_index("Datetime", inplace=True)

        agg = df.resample(resample_interval).agg({
            "Prob_Positive": "mean",
            "Prob_Neutral": "mean",
            "Prob_Negative": "mean"
        }).rename(columns={
            "Prob_Positive": f"Prob_Pos_{model_name}",
            "Prob_Neutral": f"Prob_Neu_{model_name}",
            "Prob_Negative": f"Prob_Neg_{model_name}"
        })

        model_dfs.append(agg)

    if not model_dfs:
        raise ValueError("No sentiment files found.")

    merged = model_dfs[0]
    for df in model_dfs[1:]:
        merged = merged.join(df, how="outer")

    merged.fillna(0, inplace=True)

    prob_cols = [col for col in merged.columns if "Prob_Pos_" in col]
    merged["Sentiment_Avg"] = merged[prob_cols].mean(axis=1)
    merged["Sentiment_Disagreement"] = merged[prob_cols].std(axis=1)

    return merged.reset_index()

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
    df = pd.merge(price_df, sentiment_df, on="Datetime", how="inner")

    # --- Add return + label ---
    df["Return"] = df["Close"].pct_change().shift(-1)
    df["Label"] = (df["Return"] > 0).astype(int)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Merged minute-level dataset saved to {output_path}")