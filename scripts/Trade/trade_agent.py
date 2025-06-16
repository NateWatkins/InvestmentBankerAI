import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO

# --- Load .env ---
load_dotenv(dotenv_path="/Users/natwat/Desktop/CPSC_Projects/Trader/env/.env")

# --- Alpaca Setup ---
api = tradeapi.REST(
    key_id=os.getenv("APCA_API_KEY_ID"),
    secret_key=os.getenv("APCA_API_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets"
)

# --- Config ---
MODEL_PATH = "model/ppo_tsla_agent.zip"
FEATURE_CSV = "data/features/TSLA_features_full.csv"
WINDOW_SIZE = 10
SYMBOL = "TSLA"
SLEEP_INTERVAL = 60  # seconds
DROP_COLS = ["Datetime", "Return_1min"]

# --- Load Model ---
model = PPO.load(MODEL_PATH)

# --- Track position and last seen data ---
position = 0  # 0 = no position, 1 = long
last_timestamp = None

print(f"[{datetime.now()}] ✅ Trading agent started for {SYMBOL}")

while True:
    try:
        df = pd.read_csv(FEATURE_CSV, parse_dates=["Datetime"]).dropna()
        df = df.sort_values("Datetime")

        if len(df) < WINDOW_SIZE:
            print("⏳ Not enough data yet...")
            time.sleep(SLEEP_INTERVAL)
            continue

        latest_time = df["Datetime"].iloc[-1]

        if latest_time == last_timestamp:
            print("🕒 Waiting for new data...")
            time.sleep(SLEEP_INTERVAL)
            continue

        # Prepare observation
        feature_df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
        obs = feature_df.iloc[-WINDOW_SIZE:].values  # shape: (10, N_features)
        obs = np.expand_dims(obs, axis=0)  # shape: (1, 10, N_features)

        print(f"[{datetime.now()}] 🔍 Obs shape: {obs.shape}")

        # Predict action
        action, _ = model.predict(obs)
        print(f"[{datetime.now()}] 🧠 Action: {action}")

        # Execute trade
        if action == 1 and position == 0:
            api.submit_order(symbol=SYMBOL, qty=1, side="buy", type="market", time_in_force="gtc")
            position = 1
            print("🚀 BUY executed")
        elif action == 2 and position == 1:
            api.submit_order(symbol=SYMBOL, qty=1, side="sell", type="market", time_in_force="gtc")
            position = 0
            print("💰 SELL executed")

        last_timestamp = latest_time
        time.sleep(SLEEP_INTERVAL)

    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error: {e}")
        time.sleep(SLEEP_INTERVAL)
