import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.envs.trading_env import TradingEnv



# --- Load data ---
data_path = "data/features/TSLA_features_full.csv"
df = pd.read_csv(data_path, parse_dates=["Datetime"]).dropna().reset_index(drop=True)

# --- Config ---
FEATURE_COLUMNS = [col for col in df.columns if col not in ["Datetime", "Return_1min"]]
WINDOW_SIZE = 10
TOTAL_TIMESTEPS = 100_000

# --- Create environment ---
env = DummyVecEnv([
    lambda: TradingEnv(df=df, window_size=WINDOW_SIZE, feature_columns=FEATURE_COLUMNS)
])

# --- Train PPO model ---
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# --- Save model ---
os.makedirs("model", exist_ok=True)
model.save("model/ppo_tsla_agent")
print("✅ Model saved to model/ppo_tsla_agent.zip")

# --- Plot rewards ---
rewards = env.get_attr("episode_rewards")[0]

plt.figure(figsize=(10, 4))
plt.plot(rewards, label="Episode Reward")
plt.title("PPO Training Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("model/training_reward_plot.png")
plt.show()
