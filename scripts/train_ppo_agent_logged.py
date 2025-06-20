import os
import argparse
import pandas as pd
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces

# --- Custom Trading Environment ---

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0

        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

        # Example: 18 features in your dataset
        self.feature_columns = df.drop(columns=["Return", "Label", "Datetime"], errors="ignore").columns.tolist()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32,
        )


        # Discrete action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Let gymnasium handle seeding
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0

        # Any other reset logic...
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.loc[self.current_step, self.feature_columns].values
        return obs.astype(np.float32)



    def step(self, action):
        price = self.df.loc[self.current_step, "Close"]
        reward = 0

        if action == 1:  # Buy
            self.shares_held = self.balance / price
            self.balance = 0
        elif action == 2 and self.shares_held > 0:  # Sell
            self.balance = self.shares_held * price
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * price

        # Reward logic
        reward = self.df.loc[self.current_step, "Return"] * (1 if action == 1 else -1 if action == 2 else 0)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        pass

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/features/TSLA_features_full.csv", help="Path to training dataset")
parser.add_argument("--model", type=str, default="model/ppo_tsla_agent", help="Path to save/load model")
parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training")
args = parser.parse_args()

# --- Load Data ---
df = pd.read_csv(args.data, parse_dates=["Datetime"])
df = df.dropna().reset_index(drop=True)

# --- Setup Env ---
env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

# --- Load or Initialize Model ---
if os.path.exists(args.model + ".zip"):
    print("🔁 Continuing training from previous model...")
    model = PPO.load(args.model, env=env)
else:
    print("🆕 Training new model from scratch...")
    model = PPO("MlpPolicy", env, verbose=1)

# --- Train Model ---
model.learn(total_timesteps=args.timesteps)

# --- Save Model ---
os.makedirs(os.path.dirname(args.model), exist_ok=True)
model.save(args.model)
print(f"✅ Model saved to {args.model}")
