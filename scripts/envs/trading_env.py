import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, window_size, feature_columns):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.feature_columns = feature_columns
        self.current_step = window_size
        self.position = 0  # 0 = no position, 1 = long

        self.episode_rewards = []
        self.total_reward = 0

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(feature_columns)),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0
        self.total_reward = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        obs = self.df[self.feature_columns].iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        return obs.astype(np.float32)

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        price_now = self.df["Close"].iloc[self.current_step]
        price_prev = self.df["Close"].iloc[self.current_step - 1]

        if action == 1 and self.position == 0:
            self.position = 1  # Buy
        elif action == 2 and self.position == 1:
            reward = price_now - price_prev
            self.position = 0  # Sell
        elif self.position == 1:
            reward = (price_now - price_prev) * 0.1  # Hold reward

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= len(self.df):
            done = True
            truncated = True
            self.episode_rewards.append(self.total_reward)

        obs = self._get_obs()
        return obs, reward, done, truncated, {}
