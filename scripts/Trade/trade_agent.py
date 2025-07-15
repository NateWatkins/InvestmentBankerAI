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
import subprocess

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import get_ticker, get_paths, Config

def setup_api_connection():
    """Setup and test Alpaca API connection"""
    print("Setting up Alpaca API connection...")
    
    load_dotenv(dotenv_path=Config.ENV_PATH)
    
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("❌ ERROR: Alpaca API keys not found in .env file")
        print("Please add APCA_API_KEY_ID and APCA_API_SECRET_KEY to your .env file")
        return None
        
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url="https://paper-api.alpaca.markets"
        )
        
        # Test connection
        account_info = api.get_account()
        print(f"✅ Connected to Alpaca - Account: {account_info.id}")
        print(f"✅ Buying power: ${account_info.buying_power}")
        return api
        
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Alpaca API: {e}")
        return None

def load_trading_model(model_file_path):
    """Load the PPO model with error handling"""
    print(f"Loading trading model from {model_file_path}...")
    
    if not os.path.exists(model_file_path):
        print(f"❌ ERROR: Model file not found: {model_file_path}")
        print("Run training first: python scripts/envs/train_ppo_agent_logged.py")
        return None
        
    try:
        trading_model = PPO.load(model_file_path)
        print("✅ Trading model loaded successfully")
        return trading_model
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return None

def update_market_data(quick_mode=True):
    """Update market data - quick mode skips slow sentiment analysis"""
    if quick_mode:
        print("⚡ Quick market data update (price + technical only)...")
        try:
            # Import here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from quick_data_update import quick_update
            
            return quick_update()
            
        except Exception as e:
            print(f"❌ Quick update failed: {e}")
            print("⚠️  Falling back to existing data...")
            return True  # Continue with existing data
    else:
        print("🔄 Full market data update (includes sentiment - slow)...")
        try:
            notebook_path = os.path.join(Config.PROJECT_ROOT, "scripts", "Data_manager.ipynb")
            result = subprocess.run([
                sys.executable, "-m", "nbconvert", 
                "--to", "notebook", 
                "--execute", "--inplace", 
                notebook_path
            ], check=True, capture_output=True, text=True)
            
            print("✅ Full market data updated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Failed to update market data: {e}")
            return False
        except Exception as e:
            print(f"❌ ERROR: Unexpected error updating data: {e}")
            return False

def load_market_features(csv_file_path, required_window_size):
    """Load and validate market features from CSV with smart NaN handling"""
    print(f"Loading market features from {csv_file_path}...")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ ERROR: Features file not found: {csv_file_path}")
        return None
        
    try:
        market_data = pd.read_csv(csv_file_path, parse_dates=["Datetime"])
        print(f"📊 Raw data loaded: {len(market_data)} rows")
        
        # Check for problematic columns with all NaN values
        all_nan_cols = [col for col in market_data.columns if market_data[col].isnull().all()]
        if all_nan_cols:
            print(f"⚠️  Dropping columns with all NaN values: {all_nan_cols}")
            market_data = market_data.drop(columns=all_nan_cols)
        
        # Only drop rows where essential columns (price data) are missing
        essential_cols = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
        market_data = market_data.dropna(subset=essential_cols)
        
        # Fill remaining NaN values with forward fill for continuity
        market_data = market_data.ffill().bfill()
        
        market_data = market_data.sort_values("Datetime")
        
        if len(market_data) < required_window_size:
            print(f"❌ ERROR: Not enough data. Need {required_window_size}, got {len(market_data)}")
            return None
            
        print(f"✅ Loaded {len(market_data)} rows of market data")
        print(f"   Date range: {market_data['Datetime'].min()} to {market_data['Datetime'].max()}")
        return market_data
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load features: {e}")
        return None

def prepare_model_observation(market_data):
    """Prepare the observation data for the model (single row, not windowed)"""
    try:
        # Use same column exclusions as training script - exclude non-numeric columns too
        exclude_columns = ["Return", "Label", "Datetime", "Ticker", "Source", "Source_Type"]
        
        # Only select numeric columns for model input
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Get the most recent row (latest market state)
        latest_observation = market_data[feature_columns].iloc[-1].values
        
        # Convert to numpy array with proper shape for model
        model_input = latest_observation.astype(np.float32)
        
        print(f"✅ Prepared observation with shape: {model_input.shape} (features: {len(model_input)})")
        print(f"   Numeric feature columns: {len(feature_columns)}")
        return model_input
        
    except Exception as e:
        print(f"❌ ERROR: Failed to prepare observation: {e}")
        print(f"Available columns: {list(market_data.columns)}")
        print(f"Numeric columns: {list(market_data.select_dtypes(include=[np.number]).columns)}")
        return None

def execute_trade_action(api_connection, action_number, current_position, stock_symbol):
    """Execute buy/sell/hold based on model prediction"""
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_name = action_names.get(action_number, "UNKNOWN")
    
    print(f"🤖 Model prediction: {action_name} (action {action_number})")
    
    try:
        # BUY: Only if we don't have a position
        if action_number == 1 and current_position == 0:
            print(f"🚀 Executing BUY order for {stock_symbol}")
            api_connection.submit_order(
                symbol=stock_symbol,
                qty=1,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            print("✅ BUY order submitted successfully")
            return 1  # New position
            
        # SELL: Only if we have a position
        elif action_number == 2 and current_position == 1:
            print(f"💰 Executing SELL order for {stock_symbol}")
            api_connection.submit_order(
                symbol=stock_symbol,
                qty=1,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            print("✅ SELL order submitted successfully")
            return 0  # No position
            
        # HOLD or invalid action
        else:
            if action_number == 0:
                print("⏸️ Holding current position")
            else:
                print(f"⏸️ No action taken (action {action_number} not valid for current position {current_position})")
            return current_position
            
    except Exception as e:
        print(f"❌ ERROR: Failed to execute trade: {e}")
        return current_position  # Keep current position on error

def main():
    parser = argparse.ArgumentParser(description="Live trading agent with reliable error handling")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., TSLA, AAPL, NVDA)")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--data", type=str, help="Path to features CSV")
    args = parser.parse_args()
    
    print("="*50)
    print("🤖 TSLA Trading System Starting Up")
    print("="*50)
    
    # Update configuration if ticker provided
    if args.ticker:
        print(f"Using custom ticker: {args.ticker}")
        from config import set_ticker
        set_ticker(args.ticker)
    
    # Get current configuration
    current_ticker = get_ticker()
    current_paths = get_paths()
    
    model_file_path = args.model or current_paths["model_zip"]
    features_csv_path = args.data or current_paths["ready_csv"]  # Use same data as training
    data_notebook_path = os.path.join(Config.PROJECT_ROOT, "scripts", "Data_manager.ipynb")
    
    print(f"📊 Trading ticker: {current_ticker}")
    print(f"🧠 Model file: {model_file_path}")
    print(f"📈 Features file: {features_csv_path}")
    
    # Setup components with error handling
    api_connection = setup_api_connection()
    if not api_connection:
        print("❌ Cannot continue without API connection")
        return
        
    trading_model = load_trading_model(model_file_path)
    if not trading_model:
        print("❌ Cannot continue without trading model")
        return
    
    # Trading state
    current_position = 0  # 0 = no position, 1 = long position
    last_data_timestamp = None
    loop_count = 0
    
    print("\n🚀 Trading loop starting...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            loop_count += 1
            print(f"\n--- Trading Loop #{loop_count} at {datetime.now().strftime('%H:%M:%S')} ---")
            
            # Step 1: Update market data (quick mode - only price/technical data)
            data_update_success = update_market_data(quick_mode=True)
            if not data_update_success:
                print("⚠️ Failed to update data, using existing data")
            
            # Step 2: Load latest market features
            market_data = load_market_features(features_csv_path, Config.WINDOW_SIZE)
            if market_data is None:
                print("⏳ Waiting for valid data...")
                time.sleep(Config.SLEEP_INTERVAL)
                continue
            
            # Step 3: Check if we have new data
            latest_data_time = market_data["Datetime"].iloc[-1]
            if latest_data_time == last_data_timestamp:
                print(f"🕒 No new data since {latest_data_time}")
                time.sleep(Config.SLEEP_INTERVAL)
                continue
            
            print(f"✅ New data available - latest: {latest_data_time}")
            
            # Step 4: Prepare model input
            model_observation = prepare_model_observation(market_data)
            if model_observation is None:
                print("⚠️ Failed to prepare model input")
                time.sleep(Config.SLEEP_INTERVAL)
                continue
            
            # Step 5: Get model prediction
            try:
                predicted_action, _ = trading_model.predict(model_observation)
                # Convert numpy array to scalar if needed
                if hasattr(predicted_action, 'item'):
                    predicted_action = predicted_action.item()
                elif hasattr(predicted_action, '__getitem__'):
                    predicted_action = predicted_action[0]
                print(f"🧠 Model prediction: {predicted_action}")
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
                time.sleep(Config.SLEEP_INTERVAL)
                continue
            
            # Step 6: Execute trading action
            new_position = execute_trade_action(
                api_connection, 
                predicted_action, 
                current_position, 
                current_ticker
            )
            
            # Update state
            current_position = new_position
            last_data_timestamp = latest_data_time
            
            print(f"📊 Current position: {current_position} ({'LONG' if current_position == 1 else 'NONE'})")
            print(f"⏰ Sleeping for {Config.SLEEP_INTERVAL} seconds...")
            time.sleep(Config.SLEEP_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
        print(f"Final position: {current_position}")
    except Exception as e:
        print(f"\n❌ Unexpected error in main loop: {e}")
        print("Trading system stopped")

if __name__ == "__main__":
    main()
