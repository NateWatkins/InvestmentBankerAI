import pandas as pd
import ta  # technical analysis indicators

import os
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from scripts.realtime_features import compute_features
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()


# --- CONFIGURE ---
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL = "TSLA"
WINDOW_SIZE = 10

# Initialize Alpaca
api = tradeapi.REST(ALPACA_KEY_ID, ALPACA_SECRET_KEY, BASE_URL)

# Load PPO model
model = PPO.load("model/ppo_tsla_agent")

# Track position
position = 0  # 0 = no position, 1 = long

while True:
    try:
        # Get latest 50 1-min bars
        bars = api.get_bars(SYMBOL, timeframe="1Min", limit=50).df
        if bars.empty:
            time.sleep(60)
            continue

        # Compute features
        df = compute_features(bars)
        obs = df.iloc[-WINDOW_SIZE:].values  # Shape (window_size, n_features)
        obs = np.expand_dims(obs, axis=0)    # Shape (1, window_size, n_features)

        # Predict action
        action, _ = model.predict(obs)
        print(f"[{datetime.now()}] Action: {action}")

        # Execute action
        if action == 1 and position == 0:
            api.submit_order(symbol=SYMBOL, qty=1, side="buy", type="market", time_in_force="gtc")
            position = 1
            print("BUY executed")

        elif action == 2 and position == 1:
            api.submit_order(symbol=SYMBOL, qty=1, side="sell", type="market", time_in_force="gtc")
            position = 0
            print("SELL executed")

        time.sleep(60)

    except Exception as e:
        print("Error:", e)
        time.sleep(60)

