#!/usr/bin/env python3
"""
Quick data update for trading - only updates price and technical data, skips sentiment
"""

import subprocess
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_ticker

def quick_update():
    """Update only price and technical data (skip slow sentiment analysis)"""
    ticker = get_ticker()
    print(f"🚀 Quick update for {ticker} (skipping sentiment)...")
    
    try:
        # 1. Update price data (fast - ~5 seconds)
        print("📈 Updating price data...")
        subprocess.run([
            sys.executable, "scripts/dowload_data.py", "--ticker", ticker
        ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # 2. Update technical indicators (fast - ~2 seconds)
        print("🔧 Computing technical indicators...")
        subprocess.run([
            sys.executable, "scripts/compute_ema.py", "--ticker", ticker  
        ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # 3. Combine data to create final ready dataset (fast - ~3 seconds)
        print("🔗 Creating final dataset...")
        
        # Import here to create final dataset
        from scripts.combineSingleDB import load_and_aggregate_sentiment, full_merge
        from config import get_paths
        
        paths = get_paths(ticker)
        try:
            sentiment_df = load_and_aggregate_sentiment(paths['sentiment_combined'])
            full_merge(sentiment_df, paths['features_csv'], paths['ready_csv'])
            print('✅ Final dataset created')
        except Exception as e:
            print(f'⚠️ Using existing final dataset: {e}')
        
        print("✅ Quick update complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Update failed: {e}")
        return False

if __name__ == "__main__":
    quick_update()