#!/usr/bin/env python3
"""
Test script to verify the configuration system works correctly.
Tests ticker switching and path generation for different symbols.
"""

import os
import sys
from config import Config, get_ticker, set_ticker, get_paths

def test_default_configuration():
    """Test that default configuration loads correctly."""
    print("=== Testing Default Configuration ===")
    ticker = get_ticker()
    print(f"Default ticker: {ticker}")
    
    paths = get_paths()
    print(f"Default paths for {ticker}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    assert ticker == "TSLA", f"Expected default ticker 'TSLA', got '{ticker}'"
    assert "TSLA" in paths["raw_csv"], f"Expected TSLA in raw_csv path, got {paths['raw_csv']}"
    print("✅ Default configuration test passed\n")

def test_ticker_switching():
    """Test switching between different ticker symbols."""
    print("=== Testing Ticker Switching ===")
    test_tickers = ["AAPL", "NVDA", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"Testing ticker: {ticker}")
        set_ticker(ticker)
        
        current_ticker = get_ticker()
        assert current_ticker == ticker, f"Expected '{ticker}', got '{current_ticker}'"
        
        paths = get_paths()
        
        # Check that ticker-specific paths contain the ticker
        ticker_paths = ["raw_csv", "sentiment_combined", "features_csv", "features_full_csv", 
                       "ready_csv", "archive_csv", "last_fetch", "distilbert_sentiment", 
                       "finbert_sentiment", "roberta_sentiment"]
        
        for key in ticker_paths:
            if key in paths:
                assert ticker in paths[key], f"Expected {ticker} in {key} path: {paths[key]}"
        
        # Check model paths contain lowercase ticker
        if "model_path" in paths:
            assert ticker.lower() in paths["model_path"], f"Expected {ticker.lower()} in model_path: {paths['model_path']}"
        if "model_zip" in paths:
            assert ticker.lower() in paths["model_zip"], f"Expected {ticker.lower()} in model_zip: {paths['model_zip']}"
        
        print(f"  ✅ {ticker} configuration working")
    
    print("✅ Ticker switching test passed\n")

def test_path_generation():
    """Test that all required paths are generated correctly."""
    print("=== Testing Path Generation ===")
    
    set_ticker("TEST")
    paths = get_paths()
    
    required_keys = [
        "raw_csv", "sentiment_combined", "features_csv", "features_full_csv",
        "ready_csv", "model_path", "model_zip", "news_dir", "last_fetch"
    ]
    
    for key in required_keys:
        assert key in paths, f"Missing required path key: {key}"
        
        # Skip directory paths that don't contain ticker
        if key in ["news_dir"]:
            continue
            
        # Model paths use lowercase ticker
        if "model" in key:
            assert "test" in paths[key], f"Ticker 'test' not found in {key} path: {paths[key]}"
        else:
            assert "TEST" in paths[key], f"Ticker 'TEST' not found in {key} path: {paths[key]}"
    
    # Test specific path formats
    assert paths["raw_csv"].endswith("TEST_raw.csv")
    assert paths["model_zip"].endswith("ppo_test_agent.zip")
    assert paths["sentiment_combined"].endswith("TEST_sentiment_combined.csv")
    
    print("✅ Path generation test passed\n")

def test_config_class_methods():
    """Test Config class methods work correctly."""
    print("=== Testing Config Class Methods ===")
    
    # Test get_data_paths with specific ticker
    aapl_paths = Config.get_data_paths("AAPL")
    assert "AAPL" in aapl_paths["raw_csv"]
    
    # Test set_ticker method
    Config.set_ticker("AMZN")
    assert Config.TICKER == "AMZN"
    assert Config.TICKERS == ["AMZN"]
    
    # Test get_current_paths
    current_paths = Config.get_current_paths()
    assert "AMZN" in current_paths["raw_csv"]
    
    print("✅ Config class methods test passed\n")

def test_imports():
    """Test that updated scripts can import config successfully."""
    print("=== Testing Script Imports ===")
    
    # Test that we can import config from scripts directory
    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    
    try:
        # Simulate script import
        sys.path.insert(0, scripts_dir)
        sys.path.insert(0, os.path.dirname(__file__))
        
        from config import get_ticker, get_paths
        ticker = get_ticker()
        paths = get_paths()
        
        print(f"  Successfully imported config, ticker: {ticker}")
        print("✅ Script imports test passed\n")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)

def run_all_tests():
    """Run all configuration tests."""
    print("🧪 Starting Configuration System Tests\n")
    
    try:
        test_default_configuration()
        test_ticker_switching()
        test_path_generation()
        test_config_class_methods()
        test_imports()
        
        print("🎉 All configuration tests passed!")
        print("\n📋 Configuration System Summary:")
        print("✅ Default TSLA configuration works")
        print("✅ Ticker switching works for multiple symbols")
        print("✅ Dynamic path generation works")
        print("✅ All path keys are present")
        print("✅ Config class methods work")
        print("✅ Script imports work")
        
        print(f"\n🔧 Current configuration:")
        print(f"  Ticker: {get_ticker()}")
        print(f"  Project Root: {Config.PROJECT_ROOT}")
        
        # Reset to default
        set_ticker("TSLA")
        print(f"  Reset to default ticker: {get_ticker()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()