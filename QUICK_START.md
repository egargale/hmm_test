# 🚀 HMM Futures Analysis - Quick Start Guide

## ⚡ Get Running in 5 Minutes

The HMM Futures Analysis system is **fully functional** and ready to analyze real market data immediately.

### Prerequisites
- Python 3.11+
- `uv` package manager (recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd hmm_test

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 🎯 Basic Usage

### 1. Analyze Bitcoin Data (Built-in)
```bash
# Quick analysis with default settings
uv run python -m cli_simple analyze -i BTC.csv -o results/ --n-states 3
```

### 2. Analyze Your Own Data
```bash
# Prepare your CSV file with OHLCV data
# Required columns: Date, Open, High, Low, Close, Volume

# Run analysis
uv run python -m cli_simple analyze -i your_data.csv -o results/ --n-states 3
```

### 3. Python API Usage
```python
import pandas as pd
from model_training.hmm_trainer import train_single_hmm_model
from model_training.inference_engine import predict_states
from data_processing.csv_parser import process_csv
from data_processing.feature_engineering import add_features
from utils import HMMConfig

# Load and process data
data = process_csv('BTC.csv')
features = add_features(data)

# Train HMM model
config = HMMConfig(n_components=3, covariance_type='full', n_iter=100)
model, score, metadata = train_single_hmm_model(features, config)

# Predict market regimes
states = predict_states(model, features)

print(f"Model trained with log-likelihood: {score:.2f}")
print(f"Detected {config.n_components} market regimes")
```

## 📊 What You Get

### Output Files
- `model.pkl`: Trained HMM model
- `states.csv`: Predicted market states for each date
- `performance_report.html`: Interactive dashboard
- `regime_analysis.pdf`: Detailed market regime report
- `charts/`: Generated visualizations

### Market Regimes
The system detects market states like:
- **State 0**: Bull market (uptrending)
- **State 1**: Bear market (downtrending)
- **State 2**: Neutral/sideways market

### Performance Metrics
- Total return and volatility
- Sharpe ratio and maximum drawdown
- Regime transition probabilities
- State duration analysis

## 🔧 Advanced Options

### Custom Configuration
```bash
uv run python -m cli_simple analyze \
    -i BTC.csv \
    -o results/ \
    --n-states 4 \
    --covariance-type diag \
    --test-size 0.2 \
    --random-seed 42
```

### Different Processing Engines
```bash
# For large datasets, use Dask engine
uv run python -m cli_simple analyze -i large_data.csv -o results/ --engine dask

# For out-of-core processing, use Daft engine
uv run python -m cli_simple analyze -i huge_data.csv -o results/ --engine daft
```

### Feature Engineering
```python
# Custom feature configuration
feature_config = {
    'returns': {'periods': [1, 5, 10]},
    'moving_averages': {'periods': [10, 20, 50]},
    'volatility': {'periods': [14, 21]},
    'momentum': {'periods': [14]},
    'volume': {'enabled': True}
}

features = add_features(data, indicator_config=feature_config)
```

## 📈 Example Results

Analyzing BTC.csv (1,005 days of Bitcoin price data):

```
=== HMM Model Results ===
Components: 3 market regimes
Log-Likelihood: -2847.32
Converged: Yes
Iterations: 45

=== Regime Analysis ===
State 0 (Bull): 42.3% of time, avg return +0.15%
State 1 (Bear): 31.1% of time, avg return -0.08%
State 2 (Neutral): 26.6% of time, avg return +0.02%

=== Transition Matrix ===
      Bull  Bear  Neutral
Bull   0.92  0.05    0.03
Bear   0.08  0.85    0.07
Neutral 0.04  0.06    0.90
```

## 🎨 Visualizations

The system automatically generates:

1. **Price Chart with Regimes**: OHLCV chart color-coded by detected market states
2. **State Probabilities**: Time series of regime probabilities over time
3. **Performance Dashboard**: Interactive equity curve and drawdown charts
4. **Regime Characteristics**: Distribution of returns within each market state

## 📚 Learn More

- **Theory**: See `docs/HMM_Futures_Analysis_Comprehensive_Guide.md`
- **API Reference**: Check docstrings in source code
- **Examples**: Look at `examples/` directory (if exists)

## 🐛 Troubleshooting

### Common Issues

**"Model failed to converge"**
- Increase `n_iter` parameter
- Try different `covariance_type` ('diag', 'tied', 'spherical')
- Reduce number of states

**"Insufficient data"**
- Ensure you have at least 50 data points
- More data = better model quality

**"Memory issues"**
- Use Dask or Daft engine for large datasets
- Reduce feature engineering complexity

### Get Help

```bash
# Check CLI help
uv run python -m cli_simple --help

# Validate data first
uv run python -m cli_simple validate -i your_data.csv

# Enable debug logging
uv run python -m cli_simple analyze -i data.csv -o results/ --log-level DEBUG
```

## 🎯 Next Steps

1. **Experiment**: Try different numbers of states (2-5)
2. **Customize**: Adjust feature engineering for your market
3. **Backtest**: Use different state mappings for trading strategies
4. **Deploy**: Integrate into your trading workflow

---

**The system is ready to use!** 🚀

Start with the basic Bitcoin analysis, then experiment with your own data and configurations. The comprehensive documentation provides deeper theoretical background and advanced usage patterns.
