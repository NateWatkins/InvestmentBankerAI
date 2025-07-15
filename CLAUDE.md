# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvBankAI is an autonomous day trading AI system that uses reinforcement learning (PPO) to make buy/hold/sell decisions for configurable stock tickers (TSLA, AAPL, NVDA, etc.). The system combines real-time price data, technical indicators, and sentiment analysis from financial news to make trading decisions via the Alpaca API in paper-trading mode.

## System Reliability & Validation

**🔧 Before starting, validate your system:**
```bash
# Check if everything is set up correctly
python3 system_validator.py
```

The system now includes comprehensive error handling and reliability improvements:

### ✅ Reliability Features
- **Robust Error Handling**: All scripts handle API failures, missing files, and data issues gracefully
- **Clear Error Messages**: When something fails, you get simple explanations of what went wrong
- **Automatic Recovery**: System continues running even when individual components fail
- **Progress Visibility**: Clear print statements show exactly what's happening
- **Input Validation**: All data and configurations are validated before use
- **Safe Defaults**: System uses sensible fallbacks when configurations are missing
- **Smart NaN Handling**: Automatically handles missing sentiment data without breaking the pipeline
- **Real-time Data Pipeline**: Fixed data combination issues for current market conditions

### 🛠️ System Components Enhanced
- **`trade_agent.py`**: Complete rewrite with modular functions and bulletproof error handling
- **`trading_env.py`**: Added input validation and safe observation handling  
- **`train_ppo_agent_logged.py`**: Comprehensive data validation and training progress monitoring
- **`system_validator.py`**: New tool to check your entire setup before trading
- **`quick_data_update.py`**: Fast data updates (15 seconds vs 5+ minutes) for real-time trading
- **`combineSingleDB.py`**: Fixed data aggregation to handle enhanced sentiment columns

## Key Development Commands

### Enhanced Data Pipeline (Run in sequence)
```bash
# 1. Enhanced multi-source sentiment analysis
python scripts/FH_getSent.py --ticker AAPL
# Options:
#   --legacy              # Use original processing for compatibility
#   --sources polygon newsapi twitter  # Specify which sources to use
#   --start 2024-01-01    # Custom date range
#   --end 2024-01-02

# 2. Download latest price data  
python scripts/dowload_data.py --ticker AAPL

# 3. Compute technical indicators
python scripts/compute_ema.py --ticker AAPL

# 4. Execute Data_manager notebook to combine all data
python -m nbconvert --to notebook --execute --inplace scripts/Data_manager.ipynb

# Enhanced sentiment features include:
# - Multiple news sources (Polygon, NewsAPI, Alpha Vantage, FMP)
# - Social media (Twitter, Reddit financial communities)
# - Weighted sentiment scoring with engagement metrics
# - Momentum and velocity calculations
```

### Fast Data Updates (For Real-time Trading)
```bash
# Quick update - only price and technical data (15 seconds)
python quick_data_update.py

# This updates:
# - Raw price data from yfinance
# - Technical indicators (EMA, RSI, MACD, etc.)
# - Final combined dataset
# But skips slow sentiment analysis for speed
```

### Model Training
```bash
# Train PPO agent with default parameters (50k timesteps)
python scripts/envs/train_ppo_agent_logged.py

# Train with custom ticker
python scripts/envs/train_ppo_agent_logged.py --ticker AAPL --timesteps 100000

# Train with custom parameters (paths are auto-generated from ticker)
python scripts/envs/train_ppo_agent_logged.py --ticker NVDA --timesteps 100000
```

### Live Trading
```bash
# Start autonomous trading agent (requires .env setup)
python scripts/Trade/trade_agent.py

# Trade different ticker with custom model
python scripts/Trade/trade_agent.py --ticker AAPL

# Or use the runner script
python scripts/Trade/trade_agent_runner.py
```

## Architecture Overview

### Data Flow
1. **Enhanced News & Social Media Fetching** (`FH_getSent.py`) - Multi-source sentiment analysis:
   - **News Sources**: Polygon.io, NewsAPI, Alpha Vantage, Financial Modeling Prep
   - **Social Media**: Twitter, Reddit (financial subreddits)
   - **Models**: DistilBERT, FinBERT, RoBERTa with weighted scoring
   - **Advanced Features**: Sentiment momentum, velocity, engagement weighting
2. **Price Data** (`dowload_data.py`) - Downloads 1-minute price bars from yfinance for configurable tickers
3. **Feature Engineering** (`compute_ema.py`) - Computes technical indicators (EMA, SMA, RSI, MACD, VWAP, ATR)
4. **Data Combination** (`combineSingleDB.py`) - Merges price and sentiment data with forward-fill for missing sentiment
5. **Model Training** (`train_ppo_agent_logged.py`) - Trains PPO agent using Stable-Baselines3
6. **Live Trading** (`trade_agent.py`) - Executes trades via Alpaca API based on model predictions

### Key Components

- **TradingEnv** (`scripts/envs/trading_env.py`) - Custom Gym environment for RL training with discrete actions (buy/hold/sell)
- **Data Manager** (`scripts/Data_manager.ipynb`) - Orchestrates the entire data pipeline execution
- **Enhanced Sentiment Pipeline** - Multi-source, multi-model sentiment analysis:
  - **Weighted Model Scoring**: FinBERT (40%), RoBERTa (35%), DistilBERT (25%)
  - **Source Weighting**: News (70%), Social Media (30%)
  - **Engagement Weighting**: Twitter likes/retweets, Reddit upvotes/comments
  - **Momentum Calculations**: Velocity, acceleration, volatility metrics
- **Feature Engineering** - Technical indicators computed on 1-minute price bars with configurable windows

### Configuration System

The system uses a centralized configuration in `config.py` that supports multiple ticker symbols:

#### Changing Ticker Symbol
```python
# Method 1: Via command line (recommended)
python scripts/FH_getSent.py --ticker AAPL
python scripts/compute_ema.py --ticker NVDA

# Method 2: Programmatically
from config import set_ticker
set_ticker("MSFT")

# Method 3: Direct config modification
# Edit config.py and change Config.TICKER = "AAPL"
```

#### File Paths
All file paths are automatically generated based on the ticker symbol:
- Raw data: `data/raw/{TICKER}_raw.csv`
- Features: `data/features/{TICKER}_features.csv`  
- Models: `model/ppo_{ticker}_agent.zip`
- News: `data/news/{TICKER}_sentiment_combined.csv`

### Environment Setup

1. Create `.env` file in `env/` directory with Alpaca API credentials:
   ```
   APCA_API_KEY_ID=your_key_here
   APCA_API_SECRET_KEY=your_secret_here
   ```

2. All paths are now managed through `config.py` - no manual path updates needed when switching tickers.

### Enhanced Sentiment Analysis System

The sentiment analysis system has been significantly expanded with multiple data sources and advanced analytics:

#### Data Sources
1. **News Sources**:
   - **Polygon.io**: Real-time financial news
   - **NewsAPI**: Comprehensive news aggregation
   - **Alpha Vantage**: Financial news with sentiment scores
   - **Financial Modeling Prep**: Company-specific news

2. **Social Media Sources**:
   - **Twitter**: Company mentions, hashtags, engagement metrics
   - **Reddit**: Financial subreddits (/r/stocks, /r/investing, /r/wallstreetbets, etc.)

#### Advanced Features
- **Weighted Model Scoring**: Different weights for FinBERT (financial), RoBERTa (social), DistilBERT (general)
- **Engagement Weighting**: Social media posts weighted by likes, retweets, upvotes, comments
- **Source Weighting**: News (70%) vs Social Media (30%) weighting
- **Confidence Filtering**: Only high-confidence predictions included
- **Content Deduplication**: Removes duplicate articles/posts
- **Momentum Metrics**: Sentiment velocity, acceleration, volatility over time windows

#### Output Format
Enhanced mode adds these columns while maintaining backward compatibility:
- `Prob_Pos_Weighted`, `Prob_Neu_Weighted`, `Prob_Neg_Weighted`: Weighted sentiment scores
- `Sentiment_Confidence`: Model confidence level
- `Sentiment_Velocity_1h`, `Sentiment_Velocity_6h`: Rate of sentiment change
- `Sentiment_Acceleration`: Second derivative of sentiment
- `Sentiment_Momentum`: Velocity × magnitude
- `Sentiment_Volatility`: Rolling standard deviation

#### API Keys Required
Add these to your `.env` file for full functionality:
```bash
# News Sources
NEWSAPI_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here  
FMP_KEY=your_key_here

# Social Media
TWITTER_BEARER_TOKEN=your_token_here
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

### Data Structure

- `data/raw/` - Raw price data from yfinance
- `data/news/` - Sentiment analysis results
- `data/features/` - Technical indicators and engineered features  
- `data/final/` - Final merged datasets ready for training
- `model/` - Trained PPO models (.zip files)
- `archive/` - Automatic file archiving organized by ticker and date

### File Archiving System

The system now includes automatic file archiving for all important data files:

#### Archive Structure
```
archive/
├── AAPL/
│   ├── 2025-07-15/
│   │   ├── AAPL_features_09-15-30.csv
│   │   ├── AAPL_sentiment_combined_09-16-45.csv
│   │   └── AAPL_ready_09-18-22.csv
│   └── 2025-07-16/
│       └── AAPL_features_14-22-11.csv
└── TSLA/
    └── 2025-07-15/
        └── TSLA_features_10-30-15.csv
```

#### Automatic Archiving
- **Raw Data**: `{TICKER}_raw.csv` files from `dowload_data.py`
- **Features**: `{TICKER}_features.csv` files from `compute_ema.py`
- **Sentiment**: `{TICKER}_sentiment_combined.csv` files from `FH_getSent.py`
- **Final Data**: `{TICKER}_ready.csv` files from `combineSingleDB.py`

#### Archive Management
```bash
# Test the archiving system
python simple_archiver.py

# Clean old archives (optional)
from simple_archiver import clean_old_archives
clean_old_archives(30)  # Keep 30 days, delete older
```

The archiving system:
- ✅ Preserves file history automatically
- ✅ Organizes by ticker symbol and date
- ✅ Uses timestamps to prevent overwrites
- ✅ Includes error handling that won't break main workflows
- ✅ Provides detailed logging of all archive operations

### Trading System Reliability Improvements

#### Real-time Data Pipeline
- **Issue Fixed**: Data combination script now handles enhanced sentiment columns properly
- **Smart NaN Handling**: Automatically drops problematic columns while preserving essential data
- **Fast Updates**: `quick_data_update.py` provides 15-second updates vs 5+ minute full pipeline
- **Model Compatibility**: Training and trading use identical feature sets (27 numeric features)

#### Trading Agent Enhancements
- **Modular Functions**: Each function has a single, clear purpose
- **Error Recovery**: System continues running even when components fail
- **API Connection Handling**: Graceful handling of Alpaca API issues
- **Data Validation**: All inputs validated before processing
- **Progress Visibility**: Clear logging of all trading decisions

#### Current Model Structure
- **Features**: 27 numeric features (price, volume, technical indicators, sentiment scores)
- **Architecture**: PPO (Proximal Policy Optimization) with MLP policy
- **Actions**: Discrete (0=HOLD, 1=BUY, 2=SELL)
- **Training Data**: Real market data with sentiment analysis
- **Update Frequency**: Model retrains when data structure changes

### Dependencies

The project uses Python with key libraries:
- `yfinance` for market data
- `stable-baselines3` for reinforcement learning
- `alpaca-trade-api` for trading execution
- `transformers` for sentiment analysis models
- `gymnasium` for RL environment interface
- `pandas`, `numpy` for data processing

### Trading Configuration

- **Symbol**: Configurable (TSLA, AAPL, NVDA, MSFT, etc.) via config.py or --ticker parameter
- **Timeframe**: 1-minute bars
- **Actions**: Discrete (0=hold, 1=buy, 2=sell)
- **Position**: Single share transactions only
- **Window Size**: 10-minute lookback for model observations
- **Update Frequency**: 60-second intervals for live trading

### Troubleshooting Common Issues

#### "Not enough data" Error
- **Cause**: NaN values in enhanced sentiment columns
- **Fix**: System now automatically handles this with smart NaN processing

#### "String to float" Error  
- **Cause**: Non-numeric columns (ticker, source) included in model input
- **Fix**: System now uses only numeric columns for model predictions

#### "Model shape mismatch" Error
- **Cause**: Model trained on different number of features than current data
- **Fix**: Retrain model with current data structure using `train_ppo_agent_logged.py`

#### Slow Trading Updates
- **Cause**: Full sentiment analysis taking 5+ minutes per update
- **Fix**: Use `quick_data_update.py` for 15-second updates (skips sentiment)

#### Real-time Data Not Updating
- **Cause**: Data combination script failing on enhanced sentiment columns
- **Fix**: Updated `combineSingleDB.py` handles mixed data types properly