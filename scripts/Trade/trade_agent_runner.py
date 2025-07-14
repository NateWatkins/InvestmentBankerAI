import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
sys.path.append("scripts")
from envs.train_ppo_agent_logged import TradingEnv

# --- Load Trading Data ---
def load_trading_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df.dropna().reset_index(drop=True)

# --- Create Environment ---
def create_environment(df: pd.DataFrame) -> DummyVecEnv:
    return DummyVecEnv([lambda: TradingEnv(df)])

# --- Load Model ---
def load_model(model_path: str, env: DummyVecEnv) -> PPO:
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return PPO.load(model_path, env=env)

# --- Run Inference ---
def run_inference(model: PPO, env: DummyVecEnv) -> float:
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    return float(np.squeeze(total_reward))

# --- Plot Performance (optional) ---
def plot_performance(env):
    try:
        net_worth = env.get_attr("net_worth")[0]
        plt.plot(net_worth)
        plt.title("Net Worth Over Time")
        plt.xlabel("Step")
        plt.ylabel("Net Worth")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/Users/natwat/Desktop/CPSC_Projects/InvBankAI/data/features/TSLA_features_full.csv")
    parser.add_argument("--model", type=str, default="/Users/natwat/Desktop/CPSC_Projects/InvBankAI/model/ppo_tsla_agent.zip")
    parser.add_argument("--plot", action="store_true", help="Plot net worth over time")
    args = parser.parse_args()

    print("📊 Loading data...")
    df = load_trading_data(args.data)
    env = create_environment(df)

    print("📥 Loading model...")
    model = load_model(args.model, env)

    print("🤖 Running inference...")
    total_reward = run_inference(model, env)
    print(f"✅ Total Reward: {total_reward:.2f}")

    if args.plot:
        print("📈 Plotting performance...")
        plot_performance(env)

if __name__ == "__main__":
    main()
