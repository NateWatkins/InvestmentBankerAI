import os
import sys
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_ticker, get_paths, Config

def validate_training_data(data_file_path):
    """Validate the training data has required columns and sufficient rows"""
    print(f"Validating training data: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        print(f"❌ ERROR: Training data file not found: {data_file_path}")
        return None
        
    try:
        training_data = pd.read_csv(data_file_path, parse_dates=["Datetime"])
        
        # Handle NaN values intelligently like the trade agent
        all_nan_cols = [col for col in training_data.columns if training_data[col].isnull().all()]
        if all_nan_cols:
            print(f"⚠️  Dropping columns with all NaN values: {all_nan_cols}")
            training_data = training_data.drop(columns=all_nan_cols)
        
        # Only require essential columns to be non-null
        essential_cols = ["Datetime", "Close", "Return", "Label"]
        training_data = training_data.dropna(subset=essential_cols)
        training_data = training_data.ffill().bfill().reset_index(drop=True)
        
        print(f"✅ Loaded {len(training_data)} rows of training data")
        
        # Check required columns
        required_columns = ["Datetime", "Close", "Return", "Label"]
        missing_columns = [col for col in required_columns if col not in training_data.columns]
        
        if missing_columns:
            print(f"❌ ERROR: Missing required columns: {missing_columns}")
            return None
            
        # Check minimum data requirements
        if len(training_data) < 1000:
            print(f"⚠️ WARNING: Very little training data ({len(training_data)} rows). Recommend at least 1000 rows.")
            
        if len(training_data) < 100:
            print(f"❌ ERROR: Insufficient training data ({len(training_data)} rows). Need at least 100 rows.")
            return None
            
        print(f"✅ Training data validation passed")
        print(f"   Date range: {training_data['Datetime'].min()} to {training_data['Datetime'].max()}")
        print(f"   Features: {len(training_data.columns) - 3} (excluding Datetime, Return, Label)")
        
        return training_data
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load training data: {e}")
        return None

# --- Custom Trading Environment ---

class TrainingTradingEnv(gym.Env):
    """Simplified trading environment for training with better error handling"""
    
    def __init__(self, training_dataframe):
        super().__init__()
        
        if training_dataframe is None or len(training_dataframe) == 0:
            raise ValueError("Training dataframe cannot be None or empty")
            
        self.df = training_dataframe.reset_index(drop=True)
        self.current_step = 0
        
        # Trading state
        self.initial_balance = 100000
        self.current_balance = self.initial_balance
        self.shares_owned = 0
        self.total_net_worth = self.initial_balance
        self.trade_count = 0
        
        # Get feature columns (exclude non-feature columns and non-numeric columns)
        exclude_columns = ["Return", "Label", "Datetime", "Ticker", "Source", "Source_Type"]
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if not self.feature_columns:
            raise ValueError("No feature columns found in dataframe")
            
        print(f"✅ Trading environment created with {len(self.feature_columns)} features")
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32,
        )
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

 
    def reset(self, *, seed=None, options=None):
        """Reset environment to start new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.shares_owned = 0
        self.total_net_worth = self.initial_balance
        self.trade_count = 0
        
        try:
            return self._get_current_observation(), {}
        except Exception as e:
            print(f"❌ Error during reset: {e}")
            # Return safe fallback observation
            return np.zeros(len(self.feature_columns), dtype=np.float32), {}

    def _get_current_observation(self):
        """Get current state observation for the model"""
        try:
            if self.current_step >= len(self.df):
                # Return last valid observation if we're at the end
                obs_data = self.df.loc[len(self.df) - 1, self.feature_columns].values
            else:
                obs_data = self.df.loc[self.current_step, self.feature_columns].values
                
            # Handle any NaN values
            obs_data = np.nan_to_num(obs_data, nan=0.0)
            return obs_data.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error getting observation: {e}")
            return np.zeros(len(self.feature_columns), dtype=np.float32)



    def step(self, action):
        """Execute one step in the environment"""
        reward = 0
        episode_terminated = False
        truncated = False
        
        try:
            # Validate action
            if action not in [0, 1, 2]:
                action = 0  # Default to hold
                
            # Check bounds
            if self.current_step >= len(self.df):
                episode_terminated = True
                return self._get_current_observation(), 0, episode_terminated, truncated, {}
                
            current_price = self.df.loc[self.current_step, "Close"]
            
            # Execute trading action
            if action == 1 and self.shares_owned == 0:  # BUY
                self.shares_owned = self.current_balance / current_price
                self.current_balance = 0
                self.trade_count += 1
                reward = -0.001  # Small transaction cost
                
            elif action == 2 and self.shares_owned > 0:  # SELL
                self.current_balance = self.shares_owned * current_price
                self.shares_owned = 0
                self.trade_count += 1
                # Reward based on profit/loss
                reward = (self.current_balance - self.initial_balance) / self.initial_balance
                
            else:  # HOLD
                reward = 0
                
            # Update net worth
            self.total_net_worth = self.current_balance + self.shares_owned * current_price
            
            self.current_step += 1
            
            # Check if episode is done
            if self.current_step >= len(self.df) - 1:
                episode_terminated = True
                
            next_observation = self._get_current_observation()
            
            return next_observation, reward, episode_terminated, truncated, {
                "net_worth": self.total_net_worth,
                "trades": self.trade_count
            }
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            return np.zeros(len(self.feature_columns), dtype=np.float32), 0, True, True, {"error": str(e)}

    def render(self, mode="human"):
        """Render current state (optional)"""
        if mode == "human":
            print(f"Step: {self.current_step}, Balance: ${self.current_balance:.2f}, "
                  f"Shares: {self.shares_owned:.2f}, Net Worth: ${self.total_net_worth:.2f}")
        return None

def main():
    print("="*50)
    print(f"🤖 PPO Trading Model Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PPO trading agent with reliable error handling")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--data", type=str, help="Path to training dataset")
    parser.add_argument("--model", type=str, help="Path to save/load model")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training")
    args = parser.parse_args()
    
    # Update ticker if provided
    if args.ticker:
        print(f"Using custom ticker: {args.ticker}")
        from config import set_ticker
        set_ticker(args.ticker)
    
    # Get configuration
    current_ticker = get_ticker()
    current_paths = get_paths()
    
    training_data_path = args.data or current_paths["ready_csv"]
    model_save_path = args.model or current_paths["model_path"]
    
    print(f"📊 Ticker: {current_ticker}")
    print(f"📁 Training data: {training_data_path}")
    print(f"🤖 Model path: {model_save_path}")
    print(f"⏰ Training timesteps: {args.timesteps:,}")
    
    # Validate and load training data
    training_dataframe = validate_training_data(training_data_path)
    if training_dataframe is None:
        print("❌ Cannot continue without valid training data")
        return
    
    # Create training environment
    print("\n🏗️ Setting up training environment...")
    try:
        trading_env = TrainingTradingEnv(training_dataframe)
        vectorized_env = DummyVecEnv([lambda: Monitor(trading_env)])
        print("✅ Training environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create training environment: {e}")
        return
    
    # Load or create model
    print("\n🤖 Setting up PPO model...")
    model_file_with_extension = model_save_path + ".zip"
    
    try:
        if os.path.exists(model_file_with_extension):
            print(f"🔄 Loading existing model from {model_file_with_extension}")
            ppo_model = PPO.load(model_file_with_extension, env=vectorized_env)
            print("✅ Existing model loaded - continuing training")
        else:
            print("🆕 Creating new model from scratch")
            ppo_model = PPO("MlpPolicy", vectorized_env, verbose=1)
            print("✅ New model created")
    except Exception as e:
        print(f"❌ Failed to setup model: {e}")
        return
    
    # Train the model
    print(f"\n🚀 Starting training for {args.timesteps:,} timesteps...")
    try:
        start_time = datetime.now()
        ppo_model.learn(total_timesteps=args.timesteps)
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"✅ Training completed in {training_duration}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Save the model
    print(f"\n💾 Saving model to {model_save_path}...")
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        ppo_model.save(model_save_path)
        print(f"✅ Model saved successfully to {model_save_path}.zip")
        print(f"📁 Model file size: {os.path.getsize(model_file_with_extension) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Trained on {len(training_dataframe)} data points")
    print(f"🎯 Ready for trading with: python scripts/Trade/trade_agent.py --ticker {current_ticker}")

if __name__ == "__main__":
    main()
