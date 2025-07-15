import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
import argparse

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_ticker, get_paths, Config

# --- Load .env ---
load_dotenv(dotenv_path=Config.ENV_PATH)

# --- Alpaca Setup ---
api = tradeapi.REST(
    key_id=os.getenv("APCA_API_KEY_ID"),
    secret_key=os.getenv("APCA_API_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets"
)

# --- Config ---
TICKER = get_ticker()  # Get from config
PATHS = get_paths()    # Get all paths from config
MODEL_PATH = PATHS["model_zip"]
FEATURE_CSV = PATHS["features_full_csv"]
WINDOW_SIZE = Config.WINDOW_SIZE
SYMBOL = TICKER
SLEEP_INTERVAL = Config.SLEEP_INTERVAL
DROP_COLS = Config.DROP_COLS

# --- Load Model ---
model = PPO.load(MODEL_PATH)

# --- Track position and last seen data ---
position = 0  # 0 = no position, 1 = long
last_timestamp = None
notebook_path = os.path.join(Config.PROJECT_ROOT, "scripts", "Data_manager.ipynb")
print(f"[{datetime.now()}] ✅ Trading agent started for {SYMBOL}")
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Live trading agent")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--data", type=str, help="Path to features CSV")
    args = parser.parse_args()
    
    # Update configuration if ticker provided
    if args.ticker:
        from config import set_ticker
        set_ticker(args.ticker)
        global TICKER, PATHS, MODEL_PATH, FEATURE_CSV, SYMBOL
        TICKER = get_ticker()
        PATHS = get_paths()
        MODEL_PATH = args.model or PATHS["model_zip"]
        FEATURE_CSV = args.data or PATHS["features_full_csv"]
        SYMBOL = TICKER
    
    # Load Model
    model = PPO.load(MODEL_PATH)
    
    # Track position and last seen data
    position = 0  # 0 = no position, 1 = long
    last_timestamp = None
    
    print(f"[{datetime.now()}] ✓ Trading agent started for {SYMBOL}")
    
    while True:
        
# --- Run Data_manager notebook --
        
        subprocess.run([
            sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", notebook_path
        ], check=True)
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

if __name__ == "__main__":
    main()
