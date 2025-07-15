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
        """Execute one trading step"""
        reward = 0
        episode_done = False
        truncated = False
        action_taken = "NONE"
        
        try:
            # Validate action
            if action not in [0, 1, 2]:
                print(f"⚠️ Invalid action {action}, defaulting to HOLD")
                action = 0
            
            # Check if we have enough data
            if self.current_step >= len(self.df):
                print("📊 Reached end of data")
                episode_done = True
                truncated = True
                observation = self._get_obs()
                return observation, reward, episode_done, truncated, {"action": "END"}
            
            # Get current and previous prices safely
            current_price = self.df["Close"].iloc[self.current_step]
            prev_price = self.df["Close"].iloc[max(0, self.current_step - 1)]
            price_change = current_price - prev_price
            
            # Execute trading logic
            if action == 1 and self.position == 0:  # BUY
                self.position = 1
                action_taken = "BUY"
                reward = -0.001  # Small transaction cost
                
            elif action == 2 and self.position == 1:  # SELL
                reward = price_change - 0.001  # Price change minus transaction cost
                self.position = 0
                action_taken = "SELL"
                
            else:  # HOLD or invalid action
                action_taken = "HOLD"
                if self.position == 1:
                    # Give small reward for holding when in position during upward movement
                    reward = price_change * 0.1
                else:
                    reward = 0
            
            self.total_reward += reward
            self.current_step += 1
            
            # Check if episode is complete
            if self.current_step >= len(self.df):
                episode_done = True
                truncated = True
                self.episode_rewards.append(self.total_reward)
                print(f"📈 Episode complete - Total reward: {self.total_reward:.4f}")
            
            observation = self._get_obs()
            
            return observation, reward, episode_done, truncated, {
                "action": action_taken,
                "position": self.position,
                "price": current_price,
                "reward": reward
            }
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            # Return safe values on error
            safe_obs = np.zeros((self.window_size, len(self.feature_columns)), dtype=np.float32)
            return safe_obs, 0, True, True, {"error": str(e)}
