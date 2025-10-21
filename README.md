# HMM Futures Analysis

[![CI/CD](https://github.com/egargale/hmm_test/actions/workflows/ci.yml/badge.svg)](https://github.com/egargale/hmm_test/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egargale/hmm_test/branch/main/graph/badge.svg)](https://codecov.io/gh/egargale/hmm_test)
[![Documentation](https://readthedocs.org/projects/hmm-futures-analysis/badge/?version=latest)](https://hmm-futures-analysis.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/hmm-futures-analysis.svg)](https://badge.fury.io/py/hmm-futures-analysis)
[![Python versions](https://img.shields.io/pypi/pyversions/hmm-futures-analysis.svg)](https://pypi.org/project/hmm-futures-analysis/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A sophisticated Python toolkit for applying Hidden Markov Models (HMM) to futures market analysis and regime detection.

## Features

- **Multi-Engine Processing**: Choose from streaming, Dask, or Daft engines based on your data size
- **Advanced Feature Engineering**: Comprehensive technical indicators and custom feature support
- **Regime Detection**: Identify market states (trending, ranging, volatile) using HMMs
- **Professional Visualization**: Interactive charts and comprehensive dashboards
- **Backtesting Framework**: Test strategies across different market regimes
- **CLI Interface**: Command-line tools for automation and batch processing
- **Production Ready**: Type hints, comprehensive tests, and CI/CD pipeline

## Quick Start

### Installation

```bash
# Install from PyPI
pip install hmm-futures-analysis

# Install with all optional dependencies
pip install hmm-futures-analysis[all]

# Install from source
git clone https://github.com/egargale/hmm_test.git
cd hmm_test
uv sync  # or pip install -e .
```

### Basic Usage

```python
from src.data_processing.csv_parser import process_csv
from src.data_processing.feature_engineering import add_features
from src.model_training.hmm_trainer import train_model
from src.model_training.inference_engine import StateInference

# Load and prepare data
data = process_csv('your_futures_data.csv')
features = add_features(data)

# Train HMM model
X = features['close'].values.reshape(-1, 1)
model, metadata = train_model(X, config={'n_components': 3})

# Infer states
inference = StateInference(model)
states = inference.infer_states(X)

print(f"Identified {len(np.unique(states))} market regimes")
```

### Command Line Interface

```bash
# Analyze futures data with default settings
hmm-analyze analyze -i data.csv -o results/

# Use different processing engine for large datasets
hmm-analyze analyze -i large_data.csv -o results/ --engine dask

# Validate data format
hmm-analyze validate -i data.csv

# Generate visualization dashboard
hmm-analyze analyze -i data.csv -o results/ --generate-dashboard
```

## Documentation

- **User Guide**: [Complete documentation](https://hmm-futures-analysis.readthedocs.io)
- **Examples**: [Jupyter notebooks and tutorials](docs/examples/)
- **API Reference**: [Detailed API documentation](https://hmm-futures-analysis.readthedocs.io/en/latest/api/)

## Requirements

- Python 3.8+
- See `pyproject.toml` for complete dependency list

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/egargale/hmm_test.git
cd hmm_test

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m "not slow"    # Skip slow tests
```

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/

# Security checks
uv run bandit -r src/
uv run safety check
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
uv run sphinx-build -b html . _build/html

# Serve documentation locally
uv run sphinx-autobuild . _build/html
```

## Project Structure

```
hmm_test/
├── src/                          # Source code
│   ├── cli_simple.py            # Command-line interface
│   ├── data_processing/         # Data loading and feature engineering
│   ├── model_training/          # HMM training and inference
│   ├── processing_engines/      # Data processing engines
│   ├── backtesting/             # Backtesting framework
│   ├── visualization/           # Charts and dashboards
│   └── utils/                   # Utilities and configuration
├── tests/                       # Test suite
├── docs/                        # Documentation
├── examples/                    # Example scripts and notebooks
├── .github/workflows/           # CI/CD pipelines
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Examples

### Basic Market Regime Analysis

```python
import pandas as pd
from src.cli_simple import main

# Using the CLI
main(['analyze', '-i', 'es_data.csv', '-o', 'results/', '--n-states', 4])
```

### Custom Feature Engineering

```python
from src.data_processing.feature_engineering import FeatureEngineer

def custom_indicator(data):
    """Add your custom technical indicator"""
    return data['close'].pct_change(5).rolling(20).mean()

engineer = FeatureEngineer()
engineer.add_feature('custom_momentum', custom_indicator)
features = engineer.process(data)
```

### Strategy Development

```python
from examples.trading_strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(
    lookback_periods=[5, 10, 20],
    volatility_threshold=0.02,
    regime_filter=True
)

signals = strategy.generate_signals(data, hmm_states)
performance = strategy.calculate_performance_metrics(signals)
```

## Performance

- **Speed**: Processes 1M+ rows in seconds using Dask engine
- **Memory**: Efficient processing with configurable engines
- **Scalability**: From streaming (small data) to Daft (massive datasets)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `uv sync --dev`
4. Make your changes
5. Run tests: `uv run pytest`
6. Check code quality: `make quality`
7. Commit changes: `git commit -m 'Add amazing feature'`
8. Push to branch: `git push origin feature/amazing-feature`
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{hmm_futures_analysis,
  title={HMM Futures Analysis: A Python Toolkit for Market Regime Detection},
  author={HMM Futures Analysis Team},
  year={2024},
  url={https://github.com/egargale/hmm_test}
}
```

## Support

- **Documentation**: [https://hmm-futures-analysis.readthedocs.io](https://hmm-futures-analysis.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/egargale/hmm_test/issues)
- **Discussions**: [GitHub Discussions](https://github.com/egargale/hmm_test/discussions)

## Acknowledgments

- Built with [Click](https://click.palletsprojects.com/) for the CLI
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- HMM implementation from [hmmlearn](https://hmmlearn.readthedocs.io/)
- Distributed processing with [Dask](https://dask.org/)
- Documentation with [Sphinx](https://www.sphinx-doc.org/) and [Furo](https://pradyunsg.me/furo/)

---

**Disclaimer**: This software is for educational and research purposes only. Not financial advice.