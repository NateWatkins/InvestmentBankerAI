"""
Configuration module for InvBankAI trading system.
Centralizes all configurable parameters including ticker symbols and file paths.
"""

import os
from typing import List, Dict, Any

class Config:
    """Central configuration class for the trading system."""
    
    # --- Core Trading Configuration ---
    TICKER: str = "AAPL"  # Primary ticker symbol to trade
    TICKERS: List[str] = ["AAPL"]  # List of tickers (for potential multi-asset support)
    
    # --- Project Paths ---
    PROJECT_ROOT: str = "/Users/natwat/Desktop/CPSC_Projects/InvBankAI"
    
    # --- Data Configuration ---
    INTERVAL: str = "1m"  # Price data interval
    LOOKBACK_DAYS: int = 7  # Days of historical data to fetch
    WINDOW_SIZE: int = 10  # Lookback window for model observations
    SLEEP_INTERVAL: int = 60  # Trading loop sleep interval (seconds)
    
    # --- Model Configuration ---
    DROP_COLS: List[str] = ["Datetime", "Return_1min"]
    
    # --- Environment Configuration ---
    ENV_PATH: str = os.path.join(PROJECT_ROOT, "env", ".env")
    
    @classmethod
    def get_data_paths(cls, ticker: str = None) -> Dict[str, str]:
        """
        Generate all data file paths for a given ticker.
        
        Args:
            ticker: Ticker symbol (defaults to cls.TICKER)
            
        Returns:
            Dictionary of path names to full file paths
        """
        if ticker is None:
            ticker = cls.TICKER
            
        return {
            # Raw data paths
            "raw_csv": os.path.join(cls.PROJECT_ROOT, "data", "raw", f"{ticker}_raw.csv"),
            
            # News/sentiment paths
            "news_dir": os.path.join(cls.PROJECT_ROOT, "data", "news"),
            "last_fetch": os.path.join(cls.PROJECT_ROOT, "data", "news", f"{ticker}_last_fetch.txt"),
            "sentiment_combined": os.path.join(cls.PROJECT_ROOT, "data", "news", f"{ticker}_sentiment_combined.csv"),
            "distilbert_sentiment": os.path.join(cls.PROJECT_ROOT, "data", "news", f"{ticker}_distilbert_sentiment.csv"),
            "finbert_sentiment": os.path.join(cls.PROJECT_ROOT, "data", "news", f"{ticker}_finbert_sentiment.csv"),
            "roberta_sentiment": os.path.join(cls.PROJECT_ROOT, "data", "news", f"{ticker}_roberta_sentiment.csv"),
            
            # Feature paths
            "features_csv": os.path.join(cls.PROJECT_ROOT, "data", "features", f"{ticker}_features.csv"),
            "features_full_csv": os.path.join(cls.PROJECT_ROOT, "data", "features", f"{ticker}_features_full.csv"),
            
            # Final data paths
            "ready_csv": os.path.join(cls.PROJECT_ROOT, "data", "final", f"{ticker}_ready.csv"),
            "archive_csv": os.path.join(cls.PROJECT_ROOT, "data", "final", f"{ticker}_Archive.csv"),
            
            # Model paths
            "model_path": os.path.join(cls.PROJECT_ROOT, "model", f"ppo_{ticker.lower()}_agent"),
            "model_zip": os.path.join(cls.PROJECT_ROOT, "model", f"ppo_{ticker.lower()}_agent.zip"),
        }
    
    @classmethod
    def set_ticker(cls, ticker: str) -> None:
        """
        Update the primary ticker symbol.
        
        Args:
            ticker: New ticker symbol (e.g., 'AAPL', 'NVDA', 'MSFT')
        """
        cls.TICKER = ticker.upper()
        cls.TICKERS = [cls.TICKER]
    
    @classmethod
    def get_current_paths(cls) -> Dict[str, str]:
        """Get all data paths for the currently configured ticker."""
        return cls.get_data_paths(cls.TICKER)

# Create a global config instance
config = Config()

# Convenience functions for easy imports
def get_ticker() -> str:
    """Get the current ticker symbol."""
    return config.TICKER

def set_ticker(ticker: str) -> None:
    """Set the current ticker symbol."""
    config.set_ticker(ticker)

def get_paths(ticker: str = None) -> Dict[str, str]:
    """Get data paths for the specified ticker (or current ticker if None)."""
    return config.get_data_paths(ticker)