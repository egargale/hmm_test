# HMM Futures Analysis

Train and use a Hidden Markov Model on HUGE futures CSV files.

This project provides a robust implementation of Hidden Markov Models for analyzing futures market data. It includes multiple approaches for handling large datasets, from streaming with pandas to out-of-core processing with Dask and Daft. The project has been enhanced with improved data handling, robust CSV parsing, and better performance metrics.

## Features

### Core Functionality
- **Hidden Markov Model Training**: Train Gaussian HMMs on financial time series data
- **Regime Detection**: Identify different market states (e.g., low volatility uptrend, high volatility downtrend)
- **Feature Engineering**: Automatically compute log returns, ATR, ROC, RSI, Bollinger Bands, MACD, ADX, and Stochastic Oscillator indicators
- **Memory Efficient Processing**: Handle multi-gigabyte CSV files through chunking
- **Robust CSV Parsing**: Handles various CSV formats including columns with whitespace and different naming conventions

### Advanced Features
- **Model Persistence**: Save and load trained models and scalers
- **Backtesting**: Built-in simple backtest with performance metrics (Sharpe ratio, max drawdown)
- **Lookahead Bias Prevention**: Position shifting to ensure realistic backtests
- **Multiple Implementations**: Three different approaches for different use cases (Streaming, Dask, Daft)
- **Enhanced Performance Metrics**: Improved annualized Sharpe ratio and maximum drawdown calculations
- **Visualization**: Generate plots showing HMM states overlaid on price data

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

# Run with all advanced features
python main.py data.csv --n_states 3 --backtest --prevent-lookahead --plot --model-out model.pkl
```

### Advanced Features

```bash
# Enable backtesting with performance metrics
python main.py data.csv --backtest

# Prevent lookahead bias for realistic backtesting
python main.py data.csv --prevent-lookahead

# Generate visualization plot of HMM states
python main.py data.csv --plot

# Combine multiple options for comprehensive analysis
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

## Enhanced Features

The project now includes improved data handling for various CSV formats, robust column name parsing (including handling whitespace), and enhanced backtesting with more accurate performance metrics.

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
- **Enhancements**: Improved CSV parsing, robust handling of column names with whitespace, enhanced feature engineering with additional technical indicators

### 2. Dask Approach (`hmm_futures_script.py`)
- **Best for**: Large datasets requiring memory-efficient processing
- **Features**: Downcasting to float32, model persistence
- **Memory usage**: Low (chunked processing with Dask)
- **Enhancements**: Optimized chunking strategy, improved error handling

### 3. Daft Approach (`hmm_futures_daft.py`)
- **Best for**: Very large datasets requiring out-of-core processing
- **Features**: Lazy evaluation with Arrow backend, model persistence
- **Memory usage**: Very low (Arrow-backed processing)
- **Enhancements**: Better integration with Arrow data types, improved performance metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

All tests are passing successfully. The project has been enhanced with improved error handling, robust CSV parsing, and better performance metrics calculation.

See [TESTING_SUMMARY.md](TESTING_SUMMARY.md) for detailed test results and verification.

## Requirements

- Python 3.13+
- Dependencies listed in [pyproject.toml](pyproject.toml)
- Uses `uv` for dependency management
- Compatible with various CSV formats including those with whitespace in column names
- Robust error handling for malformed data

## Recent Improvements

### Enhanced Data Handling
- **Robust CSV Parsing**: Improved handling of various CSV formats including columns with leading/trailing whitespace
- **Flexible Column Names**: Support for both "DateTime"/"Close" and "Date"+"Time"/"Last" column naming conventions
- **Data Validation**: Enhanced validation of input data format with clearer error messages

### Improved Feature Engineering
- **Expanded Technical Indicators**: Added RSI, Bollinger Bands, MACD, ADX, Stochastic Oscillator, and VWAP indicators
- **Better Normalization**: Improved feature scaling methods for more consistent model training
- **Memory Optimization**: More efficient data processing with reduced memory footprint

### Enhanced Backtesting
- **Accurate Metrics**: Improved calculation of performance metrics including Sharpe ratio and maximum drawdown
- **Bias Prevention**: Better handling of lookahead bias with improved position shifting techniques
- **Detailed Reporting**: More comprehensive backtest results with equity curves and trade analytics

### Visualization Improvements
- **Enhanced Plots**: Better visualization of HMM states overlaid on price data
- **Customizable Charts**: More options for chart customization and export formats
- **Performance Tracking**: Visual representation of strategy performance over time

### Code Quality
- **Error Handling**: Improved error handling and logging throughout the application
- **Documentation**: Enhanced inline documentation and code comments
- **Testing**: Expanded test coverage with additional unit and integration tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.