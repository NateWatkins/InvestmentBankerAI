#!/usr/bin/env python3
"""
Simple system validator for TSLA Trading System
Checks if everything is set up correctly before trading/training
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_api_keys():
    """Check if required API keys are available"""
    print("🔑 Checking API keys...")
    
    env_file_path = os.path.join("env", ".env")
    if not os.path.exists(env_file_path):
        print("❌ ERROR: .env file not found in env/ directory")
        print("   Create env/.env with your API keys")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file_path)
    
    required_keys = {
        "APCA_API_KEY_ID": "Alpaca API Key",
        "APCA_API_SECRET_KEY": "Alpaca Secret Key"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} ({description})")
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        return False
    
    print("✅ All required API keys found")
    return True

def check_project_structure():
    """Check if project directories and files exist"""
    print("📁 Checking project structure...")
    
    required_dirs = [
        "scripts",
        "scripts/Trade",
        "scripts/envs", 
        "data",
        "data/raw",
        "data/features",
        "data/final",
        "data/news",
        "model",
        "env"
    ]
    
    required_files = [
        "config.py",
        "scripts/Trade/trade_agent.py",
        "scripts/envs/train_ppo_agent_logged.py",
        "scripts/envs/trading_env.py",
        "scripts/FH_getSent.py",
        "scripts/dowload_data.py",
        "scripts/compute_ema.py"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for directory in missing_dirs:
            print(f"   - {directory}")
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    if missing_dirs or missing_files:
        return False
    
    print("✅ Project structure looks good")
    return True

def check_data_availability():
    """Check if we have the minimum required data for trading"""
    print("📊 Checking data availability...")
    
    try:
        from config import get_ticker, get_paths
        
        current_ticker = get_ticker()
        paths = get_paths()
        
        print(f"   Current ticker: {current_ticker}")
        
        # Check if we have training data
        ready_csv_path = paths.get("ready_csv")
        if ready_csv_path and os.path.exists(ready_csv_path):
            try:
                training_data = pd.read_csv(ready_csv_path)
                print(f"✅ Training data found: {len(training_data)} rows")
                
                if len(training_data) < 100:
                    print("⚠️  WARNING: Very little training data. Consider running data pipeline")
                    
            except Exception as e:
                print(f"❌ Error reading training data: {e}")
                return False
        else:
            print("❌ No training data found")
            print("   Run: python scripts/dowload_data.py")
            print("   Then: python scripts/compute_ema.py") 
            print("   Then: python scripts/FH_getSent.py")
            return False
        
        # Check if we have a trained model
        model_path = paths.get("model_zip")
        if model_path and os.path.exists(model_path):
            print("✅ Trained model found")
        else:
            print("⚠️  No trained model found")
            print("   Run: python scripts/envs/train_ppo_agent_logged.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("🐍 Checking Python dependencies...")
    
    required_packages = {
        "pandas": "Data manipulation",
        "numpy": "Numerical computing", 
        "torch": "PyTorch for ML models",
        "transformers": "Hugging Face transformers",
        "stable_baselines3": "Reinforcement learning",
        "gymnasium": "RL environments",
        "alpaca_trade_api": "Alpaca trading",
        "yfinance": "Yahoo Finance data",
        "python-dotenv": "Environment variables"
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(f"{package} ({description})")
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install [package_name]")
        return False
    
    print("✅ All required packages installed")
    return True

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    print("🔗 Testing Alpaca API connection...")
    
    try:
        from dotenv import load_dotenv
        import alpaca_trade_api as tradeapi
        
        load_dotenv(dotenv_path=os.path.join("env", ".env"))
        
        api = tradeapi.REST(
            key_id=os.getenv("APCA_API_KEY_ID"),
            secret_key=os.getenv("APCA_API_SECRET_KEY"),
            base_url="https://paper-api.alpaca.markets"
        )
        
        account = api.get_account()
        print(f"✅ Connected to Alpaca successfully")
        print(f"   Account ID: {account.id}")
        print(f"   Buying power: ${float(account.buying_power):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        print("   Check your API keys in env/.env")
        return False

def main():
    """Run all validation checks"""
    print("🤖 TSLA Trading System Validator")
    print("=" * 40)
    print(f"Validation run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checks = [
        check_project_structure,
        check_python_dependencies, 
        check_api_keys,
        test_alpaca_connection,
        check_data_availability
    ]
    
    all_passed = True
    
    for check_function in checks:
        try:
            if not check_function():
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {check_function.__name__}: {e}")
            all_passed = False
        print()
    
    print("=" * 40)
    if all_passed:
        print("🎉 All checks passed! Your system is ready for trading.")
        print("\nNext steps:")
        print("   1. Train model: python scripts/envs/train_ppo_agent_logged.py")
        print("   2. Start trading: python scripts/Trade/trade_agent.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()