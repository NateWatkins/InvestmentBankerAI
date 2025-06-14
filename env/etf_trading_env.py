import gym
from gym import spaces
import numpy as np
import pandas as pd

class ETFTradingEnv(gym.Env):
    """
    Custom trading environment for RL agents using 1-minute ETF price + sentiment data.
    Implements position holding, profit/loss reward, slippage, and a context window.
    """

    def __init__(self, df: pd.DataFrame, starting_cash=10_000, window_size=10):
        super(ETFTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.starting_cash = starting_cash
        self.fee = 0.005  # flat fee per trade
        self.slippage_pct = 0.0005  # 0.05% slippage
        self.window_size = window_size

        # State features (exclude Datetime, Label, Return)
        self.features = df.columns.difference(["Datetime", "Label", "Return"]).tolist()
        self.n_features = len(self.features)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        # Initialize trade state
        self.reset()

    def reset(self):
        self.cash = self.starting_cash
        self.position = 0  # 0=no position, 1=long
        self.buy_price = 0.0
        self.total_profit = 0.0
        self.current_step = np.random.randint(self.window_size, len(self.df) - 1)
        return self._get_obs()

    def _get_obs(self):
        window = self.df.loc[self.current_step - self.window_size:self.current_step - 1, self.features]
        return window.values.astype(np.float32)

    def step(self, action):
        done = False
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        reward = 0.0

        # --- Action Logic ---
        if action == 1:  # Buy
            if self.position == 0:
                self.buy_price = price * (1 + self.slippage_pct)
                self.position = 1
                self.cash -= self.buy_price + self.fee

        elif action == 2:  # Sell
            if self.position == 1:
                sell_price = price * (1 - self.slippage_pct)
                pnl = sell_price - self.buy_price - self.fee
                reward = pnl
                self.cash += sell_price - self.fee
                self.total_profit += pnl
                self.position = 0

                # Penalty for large losses
                if pnl < -1:
                    reward -= abs(pnl) * 0.5

        # Holding position reward (optional)
        if self.position == 1:
            unrealized_pnl = price * (1 - self.slippage_pct) - self.buy_price
            reward += 0.1 * unrealized_pnl

        # Advance step
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._get_obs(), reward, done, {
            "step": self.current_step,
            "cash": self.cash,
            "position": self.position,
            "total_profit": self.total_profit
        }

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step} | Position: {self.position} | Cash: {self.cash:.2f} | Total PnL: {self.total_profit:.2f}"
        )
