# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvBankAI is an autonomous day trading AI system that uses reinforcement learning (PPO) to make buy/hold/sell decisions for configurable stock tickers (TSLA, AAPL, NVDA, etc.). The system combines real-time price data, technical indicators, and sentiment analysis from financial news to make trading decisions via the Alpaca API in paper-trading mode.

## Key Development Commands

### Data Pipeline (Run in sequence)
```bash
# All scripts now support --ticker parameter for configurable symbols

# 1. Fetch news sentiment data
python scripts/FH_getSent.py --ticker AAPL

# 2. Download latest price data  
python scripts/dowload_data.py --ticker AAPL

# 3. Compute technical indicators (EMA, RSI, MACD, etc.)
python scripts/compute_ema.py --ticker AAPL

# 4. Execute Data_manager notebook to combine all data
python -m nbconvert --to notebook --execute --inplace scripts/Data_manager.ipynb

# Or run with default ticker (TSLA) - no --ticker needed
python scripts/FH_getSent.py
python scripts/dowload_data.py
python scripts/compute_ema.py
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
1. **News Fetching** (`FH_getSent.py`) - Fetches financial news and computes sentiment scores using DistilBERT, FinBERT, and RoBERTa models
2. **Price Data** (`dowload_data.py`) - Downloads 1-minute price bars from yfinance for configurable tickers
3. **Feature Engineering** (`compute_ema.py`) - Computes technical indicators (EMA, SMA, RSI, MACD, VWAP, ATR)
4. **Data Combination** (`combineSingleDB.py`) - Merges price and sentiment data with forward-fill for missing sentiment
5. **Model Training** (`train_ppo_agent_logged.py`) - Trains PPO agent using Stable-Baselines3
6. **Live Trading** (`trade_agent.py`) - Executes trades via Alpaca API based on model predictions

### Key Components

- **TradingEnv** (`scripts/envs/trading_env.py`) - Custom Gym environment for RL training with discrete actions (buy/hold/sell)
- **Data Manager** (`scripts/Data_manager.ipynb`) - Orchestrates the entire data pipeline execution
- **Sentiment Pipeline** - Multi-model sentiment analysis with aggregation and disagreement metrics
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

### Data Structure

- `data/raw/` - Raw price data from yfinance
- `data/news/` - Sentiment analysis results
- `data/features/` - Technical indicators and engineered features  
- `data/final/` - Final merged datasets ready for training
- `model/` - Trained PPO models (.zip files)

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