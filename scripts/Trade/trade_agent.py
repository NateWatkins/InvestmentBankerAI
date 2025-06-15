import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from dotenv import load_dotenv

# --- LOAD ENV ---
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/natwat/Desktop/CPSC_Projects/Trader/env/.env")

# --- CONFIG ---
BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL = "TSLA"
FEATURES_PATH = "data/features/TSLA_features_full.csv"
WINDOW_SIZE = 10
SLEEP_INTERVAL = 60  # in seconds

# --- Alpaca Auth ---
api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    BASE_URL
)

# --- Load model ---
model = PPO.load("model/ppo_tsla_agent")

# --- Track position and last timestamp processed ---
position = 0  # 0 = no position, 1 = long
last_timestamp = None

print(f"[{datetime.now()}] ✅ Trading agent started for {SYMBOL}")

while True:
    try:
        # Load latest feature file
        df = pd.read_csv(FEATURES_PATH, parse_dates=["Datetime"])
        df = df.sort_values("Datetime")

        if df.empty or len(df) < WINDOW_SIZE:
            print(f"[{datetime.now()}] ⚠️ Not enough data, waiting...")
            time.sleep(SLEEP_INTERVAL)
            continue

        latest_time = df["Datetime"].iloc[-1]

        # Skip if we already processed this timestamp
        if latest_time == last_timestamp:
            print(f"[{datetime.now()}] ⏸ Waiting for new data...")
            time.sleep(SLEEP_INTERVAL)
            continue

        # Prepare features
        obs = df.iloc[-WINDOW_SIZE:].values
        obs = np.expand_dims(obs, axis=0)

        # Predict action
        action, _ = model.predict(obs)
        print(f"[{datetime.now()}] 🧠 Action: {action}, Timestamp: {latest_time}")

        # Execute action
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
