import pandas as pd

def aggregate_sentiment_by_minute(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, format="ISO8601").dt.floor("min")



    agg = df.groupby("Datetime").agg({
        "Prob_Positive": "mean",
        "Prob_Neutral": "mean",
        "Prob_Negative": "mean"
    }).reset_index()

    def label_sentiment(row):
        probs = [row["Prob_Negative"], row["Prob_Neutral"], row["Prob_Positive"]]
        return ["negative", "neutral", "positive"][probs.index(max(probs))]

    agg["Sentiment"] = agg.apply(label_sentiment, axis=1)
    agg["Ticker"] = "TSLA"

    agg = agg.rename(columns={
        "Prob_Positive": f"Prob_Pos_{model_name}",
        "Prob_Neutral": f"Prob_Neu_{model_name}",
        "Prob_Negative": f"Prob_Neg_{model_name}"
    })

    return agg[["Datetime", "Ticker", "Sentiment",
                f"Prob_Pos_{model_name}", f"Prob_Neu_{model_name}", f"Prob_Neg_{model_name}"]]

