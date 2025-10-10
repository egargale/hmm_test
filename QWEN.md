# HMM Futures Analysis Project Context

## Project Overview
This is a Hidden Markov Model (HMM) futures trading analysis project that provides multiple implementations for analyzing financial market data. The project uses Gaussian HMMs to identify different market regimes (e.g., low volatility uptrend, high volatility downtrend) in futures market data.

## Core Components

### Main Implementation (`main.py`)
- **Purpose**: Streaming approach for medium-sized datasets
- **Features**: 
  - Complete feature set with log returns, ATR (14-period), and ROC (5-period)
  - Model persistence with both model and scaler saved
  - Backtesting functionality with performance metrics
  - Lookahead bias prevention via position shifting
  - Memory-efficient chunking for large CSV files
- **Memory Usage**: Moderate (processed data must fit in RAM)

### Dask Implementation (`hmm_futures_script.py`)
- **Purpose**: Memory-efficient processing for large datasets
- **Features**:
  - Uses Dask for chunked processing
  - Feature set: log returns and 20-period range (volatility proxy)
  - Model persistence using joblib
  - Lookahead bias prevention implemented
- **Memory Usage**: Low (chunked processing with Dask)

### Daft Implementation (`hmm_futures_daft.py`)
- **Purpose**: Out-of-core processing for very large datasets
- **Features**:
  - Uses Daft (Apache Arrow backend) for lazy evaluation
  - Feature set: log returns and 20-period EWMA of range
  - Model persistence with both model and scaler saved
  - Lookahead bias prevention implemented
- **Memory Usage**: Very low (Arrow-backed processing)

## Key Features

### Feature Engineering
- **Log Returns**: Calculated using `np.log(df["Close"]).diff()`
- **ATR/Volatility**: Average True Range for pandas implementation, range proxies for other implementations  
- **ROC/Momentum**: Rate of change indicator

### Model Training
- Uses GaussianHMM from hmmlearn
- Configurable number of states (default 3)
- StandardScaler for feature normalization
- Model persistence via pickle/joblib

### Backtesting
- Simple regime-based strategy:
  - State 0: Long position (low volatility uptrend)
  - State 1: Flat position (neutral market)
  - State 2: Short position (high volatility downtrend)
- Performance metrics: Sharpe ratio and maximum drawdown
- Annualization factor assumes intraday data (252 * 78)

### Lookahead Bias Prevention
- Implemented via `np.roll(states, 1)` to shift positions by one time step
- Critical for realistic backtesting results

## Project Structure
```
hmm_test/
├── main.py                 # Streaming pandas implementation (full feature set)
├── hmm_futures_script.py   # Dask implementation (memory efficient)
├── hmm_futures_daft.py     # Daft implementation (out-of-core processing)
├── pyproject.toml          # Dependencies and project config
├── README.md               # Main documentation
├── AGENTS.md               # Implementation guidance
├── test_*.py               # Unit and integration tests
├── run_all_tests.py        # Test runner
└── TESTING_SUMMARY.md      # Test results
```

## Dependencies
- Python 3.13+
- hmmlearn >= 0.3.3
- pandas >= 2.3.1
- dask (for script.py)
- daft >= 0.5.9 (for daft.py)
- scikit-learn >= 1.7.0
- ta >= 0.11.0 (technical analysis indicators)
- joblib >= 1.5.1
- matplotlib >= 3.10.3

## Usage
```bash
# Install dependencies (uses uv package manager)
uv sync

# Basic usage
python main.py data.csv
python main.py data.csv --n_states 4 --max_iter 200 --chunksize 50000

# With features
python main.py data.csv --backtest --prevent-lookahead --plot --model-out model.pkl

# Run tests
python run_all_tests.py
```

## Development Conventions
- Type hints using Python 3.13+ syntax
- Standard library imports first, then third-party, then local
- Snake_case for functions/variables, UPPER_CASE for constants
- Comprehensive error handling with logging
- Unit tests for core functions and integration tests for CLI
- Docstrings for all functions

## Critical Implementation Details
- State interpretation: State 0 = long, State 1 = flat, State 2 = short (consistent across implementations)
- Performance metrics assume intraday data with 252*78 annualization factor
- Index alignment after feature engineering (especially in Dask implementation)
- Dtype downcasting to float32 for memory efficiency
- Each implementation uses different feature sets optimized for their respective data processing backends