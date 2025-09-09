# HMM Futures Analysis

Train and use a Hidden Markov Model on HUGE futures CSV files.

This project provides a robust implementation of Hidden Markov Models for analyzing futures market data. It includes multiple approaches for handling large datasets, from streaming with pandas to out-of-core processing with Dask and Daft.

## Features

### Core Functionality
- **Hidden Markov Model Training**: Train Gaussian HMMs on financial time series data
- **Regime Detection**: Identify different market states (e.g., low volatility uptrend, high volatility downtrend)
- **Feature Engineering**: Automatically compute log returns, ATR, and ROC indicators
- **Memory Efficient Processing**: Handle multi-gigabyte CSV files through chunking

### Advanced Features
- **Model Persistence**: Save and load trained models and scalers
- **Backtesting**: Built-in simple backtesting with performance metrics
- **Lookahead Bias Prevention**: Position shifting to ensure realistic backtests
- **Multiple Implementations**: Three different approaches for different use cases

## Installation

```bash
# Install dependencies (uses uv package manager)
uv sync
```

## Usage

### Basic Usage

```bash
# Train a 3-state HMM on your futures data
python main.py data.csv

# Train with custom parameters
python main.py data.csv --n_states 4 --max_iter 200 --chunksize 50000

# Save the trained model for later use
python main.py data.csv --model-out my_model.pkl

# Load a pre-trained model
python main.py data.csv --model-path my_model.pkl
```

### Advanced Features

```bash
# Enable backtesting
python main.py data.csv --backtest

# Prevent lookahead bias
python main.py data.csv --prevent-lookahead

# Generate visualization
python main.py data.csv --plot

# Combine multiple options
python main.py data.csv --n_states 3 --backtest --prevent-lookahead --plot --model-out model.pkl
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `csv` | Path to futures OHLCV CSV (required) | N/A |
| `-n, --n_states` | Number of hidden states | 3 |
| `-i, --max_iter` | Max EM iterations | 100 |
| `-p, --plot` | Save quick sanity plot | False |
| `--model-path` | Path to pre-trained model and scaler | None |
| `--model-out` | Path to save trained model and scaler | None |
| `--chunksize` | Chunk size for reading CSV | 100000 |
| `--prevent-lookahead` | Prevent lookahead bias by shifting positions | False |
| `--backtest` | Run simple backtest after training | False |

## Output Files

The program generates several output files:

- `data.hmm_states.csv`: Input data with predicted HMM states
- `data.backtest.csv`: Backtesting results (if `--backtest` enabled)
- `data.png`: Visualization plot (if `--plot` enabled)

## Data Format

The input CSV file should contain the following columns:

```csv
DateTime,Open,High,Low,Close,Volume
2023-01-01 00:00:00,100.0,101.0,99.0,100.5,1000
...
```

## Implementation Details

This project offers three different implementations optimized for different scenarios:

### 1. Streaming Approach (`main.py`)
- **Best for**: Medium-sized datasets that can fit in memory after processing
- **Features**: All advanced features including model persistence, backtesting, and lookahead prevention
- **Memory usage**: Moderate (processed data must fit in RAM)

### 2. Dask Approach (`hmm_futures_script.py`)
- **Best for**: Large datasets requiring memory-efficient processing
- **Features**: Downcasting to float32, model persistence
- **Memory usage**: Low (chunked processing with Dask)

### 3. Daft Approach (`hmm_futures_daft.py`)
- **Best for**: Very large datasets requiring out-of-core processing
- **Features**: Lazy evaluation with Arrow backend, model persistence
- **Memory usage**: Very low (Arrow-backed processing)

See [AGENTS.md](AGENTS.md) for detailed guidance on when to use each implementation.

## Backtesting Strategy

The built-in backtesting uses a simple regime-based strategy:
- **State 0**: Long position (low volatility uptrend)
- **State 1**: Flat position (neutral market)
- **State 2**: Short position (high volatility downtrend)

Performance metrics include:
- Final equity
- Annualized Sharpe ratio
- Maximum drawdown

## Development

### Running Tests

```bash
# Run all tests
python run_all_tests.py

# Run unit tests
python test_main.py

# Run CLI tests
python test_cli.py
```

See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for detailed test results and verification.

## Requirements

- Python 3.13+
- Dependencies listed in [pyproject.toml](pyproject.toml)
- Uses `uv` for dependency management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.