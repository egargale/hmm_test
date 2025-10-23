# Testing Framework Design

## Executive Summary

**Purpose**: Design comprehensive testing framework for enhanced src directory architecture
**Scope**: All 12 src modules, CLI interface, and integration points
**Testing Types**: Unit tests, integration tests, performance tests, compatibility tests
**Coverage Target**: >90% code coverage with comprehensive test scenarios
**Complexity**: Medium to High (comprehensive framework with multiple test types)
**Priority**: High (Foundation for quality assurance and migration validation)

This document outlines a comprehensive testing framework design that will ensure the quality, reliability, and performance of the enhanced src directory architecture throughout the migration process and beyond.

---

## Current Testing State Analysis

### Existing Test Coverage Assessment

**Current Test Files in Main Directory**:
- **test_main.py**: Basic main.py functionality testing
- **test_cli.py**: CLI interface testing
- **test_hmm_models.py**: HMM model testing
- **test_backtesting.py**: Backtesting engine testing
- **test_visualization.py**: Visualization testing
- **test_processing_engines/**: Processing engine testing (Dask, Daft)
- **Other specialized tests**: 15+ additional test files

**Current Testing Characteristics**:
- **Fragmented Testing**: Tests scattered across multiple files without unified framework
- **Limited Coverage**: Estimated 60-70% coverage of main directory functionality
- **No Integration Testing**: Limited end-to-end workflow validation
- **No Performance Testing**: No systematic performance benchmarking
- **No Compatibility Testing**: No migration compatibility validation

**Testing Framework Gaps**:
- **No Centralized Test Runner**: Each test file run independently
- **Limited Test Data Management**: No systematic test data generation
- **No Test Utilities**: Limited shared testing utilities and fixtures
- **No Mocking Strategy**: No consistent approach to external dependencies
- **No CI/CD Integration**: No automated testing pipeline

---

## Enhanced Testing Framework Architecture

### Testing Framework Structure

```
src/testing/
├── __init__.py                    # Testing framework initialization
├── conftest.py                  # Pytest configuration
├── requirements.txt               # Testing dependencies
├── pytest.ini                   # Pytest configuration file
├── tox.ini                      # Tox configuration for multi-environment testing
├── Makefile                     # Testing commands and workflows
├── fixtures/                    # Test data and fixtures
│   ├── __init__.py
│   ├── sample_data.py          # Sample financial data generation
│   ├── mock_models.py          # Mock model objects
│   ├── test_configs.py         # Test configuration objects
│   └── data_generators.py      # Data generation utilities
├── unit/                        # Unit tests
│   ├── __init__.py
│   ├── test_data_processing/    # Data processing unit tests
│   │   ├── __init__.py
│   │   ├── test_csv_parser.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_data_validation.py
│   │   └── test_data_cleaner.py
│   ├── test_hmm_models/        # HMM model unit tests
│   │   ├── __init__.py
│   │   ├── test_gaussian_hmm.py
│   │   ├── test_gmm_hmm.py
│   │   ├── test_factory.py
│   │   └── test_ensemble.py
│   ├── test_backtesting/       # Backtesting unit tests
│   │   ├── __init__.py
│   │   ├── test_strategy_engine.py
│   │   ├── test_performance_analyzer.py
│   │   ├── test_transaction_costs.py
│   │   └── test_portfolio_manager.py
│   ├── test_processing_engines/ # Processing engine unit tests
│   │   ├── __init__.py
│   │   ├── test_streaming_engine.py
│   │   ├── test_dask_engine.py
│   │   ├── test_daft_engine.py
│   │   └── test_batch_engine.py
│   ├── test_algorithms/        # Algorithm unit tests
│   │   ├── __init__.py
│   │   ├── test_hmm_algorithm.py
│   │   ├── test_lstm_algorithm.py
│   │   └── test_hybrid_algorithm.py
│   ├── test_deep_learning/     # Deep learning unit tests
│   │   ├── __init__.py
│   │   ├── test_lstm_model.py
│   │   ├── test_training.py
│   │   └── test_inference.py
│   ├── test_configuration/     # Configuration unit tests
│   │   ├── __init__.py
│   │   ├── test_manager.py
│   │   ├── test_loaders.py
│   │   ├── test_validators.py
│   │   └── test_schemas.py
│   ├── test_cli/              # CLI unit tests
│   │   ├── __init__.py
│   │   ├── test_commands/
│   │   │   ├── __init__.py
│   │   │   ├── test_analyze.py
│   │   │   ├── test_validate.py
│   │   │   ├── test_infer.py
│   │   │   └── test_version.py
│   │   ├── test_config.py
│   │   └── test_progress.py
│   ├── test_visualization/     # Visualization unit tests
│   │   ├── __init__.py
│   │   ├── test_chart_generator.py
│   │   ├── test_dashboard_builder.py
│   │   ├── test_report_generator.py
│   │   └── test_interactive_plots.py
│   ├── test_monitoring/       # Monitoring unit tests
│   │   ├── __init__.py
│   │   ├── test_performance_monitor.py
│   │   ├── test_memory_monitor.py
│   │   └── test_metrics_collector.py
│   └── test_utils/             # Utility unit tests
│       ├── __init__.py
│       ├── test_validators.py
│       ├── test_serializers.py
│       └── test_math_utils.py
├── integration/                 # Integration tests
│   ├── __init__.py
│   ├── test_pipelines.py         # End-to-end pipeline tests
│   ├── test_workflows.py        # Complete workflow tests
│   ├── test_cli_workflows.py    # CLI integration tests
│   ├── test_engine_integration.py # Processing engine integration
│   └── test_algorithm_integration.py # Algorithm integration tests
├── performance/                # Performance tests
│   ├── __init__.py
│   ├── test_benchmarks.py       # Performance benchmarks
│   ├── test_memory_usage.py     # Memory usage tests
│   ├── test_scalability.py       # Scalability tests
│   └── test_profiling.py         # Code profiling tests
├── compatibility/               # Compatibility tests
│   ├── __init__.py
│   ├── test_main_directory.py   # Main directory compatibility
│   ├── test_cli_compatibility.py # CLI compatibility tests
│   ├── test_api_compatibility.py # API compatibility tests
│   └── test_data_compatibility.py # Data format compatibility
├── utils/                       # Testing utilities
│   ├── __init__.py
│   ├── assertions.py          # Custom assertion helpers
│   ├── fixtures.py           # Fixture management
│   ├── mocks.py             # Mock object utilities
│   ├── generators.py         # Test data generation
│   ├── comparators.py       # Result comparison utilities
│   └── reporters.py          # Test reporting utilities
└── reports/                     # Test reports and artifacts
    ├── __init__.py
    ├── coverage/              # Coverage reports
    ├── performance/          # Performance benchmark reports
    ├── integration/          # Integration test reports
    └── html/                 # HTML test reports
```

---

## Testing Framework Components

### 1. Test Configuration and Setup

**Pytest Configuration (conftest.py)**:
```python
# src/testing/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Pytest fixtures and configuration
def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--coverage",
        action="store_true",
        default=False,
        help="Generate coverage report"
    )

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Custom markers
    config.addinivalue_line(
        "markers",
        "unit: Unit tests",
        "integration: Integration tests",
        "performance: Performance tests",
        "slow: Slow running tests",
        "compatibility: Compatibility tests"
    )

    # Test discovery
    test_paths = [
        "src/testing/unit",
        "src/testing/integration",
        "src/testing/performance",
        "src/testing/compatibility"
    ]

    if config.getoption("--integration"):
        test_paths = ["src/testing/integration"]
    elif config.getoption("--performance"):
        test_paths = ["src/testing/performance"]
    elif config.getoption("--run-slow"):
        config.addini_value_line("addopts", "-m slow")

    config.testpaths = test_paths

    # Output configuration
    config.addini_value_line(
        "addopts",
        "--strict-markers",
        "--strict-config",
        "--tb=short"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_config():
    """Create test configuration object."""
    from src.testing.fixtures.test_configs import create_test_config
    return create_test_config()

@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    from src.testing.fixtures.sample_data import generate_sample_ohlcv_data
    return generate_sample_ohlcv_data(n_samples=1000)

@pytest.fixture
def mock_hmm_model():
    """Create mock HMM model for testing."""
    from src.testing.fixtures.mock_models import create_mock_hmm_model
    return create_mock_hmm_model()
```

**Tox Configuration (tox.ini)**:
```ini
# src/testing/tox.ini
[tox]
envlist = py38, py39, py310
skipsdist = .tox,.git,.mypy_cache,build,dist,*.egg-info
skipsinstall = true
requires = tox-conda

[testenv]
deps = pytest
       pytest-cov
       pytest-xdist
       pytest-mock
       pytest-benchmark
commands = pytest {posargs}
           python -m pytest_html_report src/testing/reports/html/index.html

[testenv:coverage]
deps = pytest
       pytest-cov
commands = pytest --cov=src --cov-report=html --cov-report=term-missing
           coverage report --fail-under=90

[testenv:integration]
deps = pytest
commands = pytest src/testing/integration

[testenv:performance]
deps = pytest
       pytest-benchmark
commands = pytest src/testing/performance --benchmark-only

[testenv:compatibility]
deps = pytest
commands = pytest src/testing/compatibility

[testenv:slow]
deps = pytest
commands = pytest -m slow src/testing/
```

### 2. Test Fixtures and Data Management

**Sample Data Generation (fixtures/sample_data.py)**:
```python
# src/testing/fixtures/sample_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import random

def generate_sample_ohlcv_data(
    n_samples: int = 1000,
    start_date: datetime = None,
    frequency: str = "1min",
    volatility: float = 0.02,
    trend: float = 0.0001,
    noise_level: float = 0.001
) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    if start_date is None:
        start_date = datetime(2023, 1, 1)

    # Generate timestamps
    if frequency == "1min":
        freq_delta = timedelta(minutes=1)
    elif frequency == "5min":
        freq_delta = timedelta(minutes=5)
    elif frequency == "1hour":
        freq_delta = timedelta(hours=1)
    elif frequency == "1day":
        freq_delta = timedelta(days=1)
    else:
        freq_delta = timedelta(minutes=1)

    timestamps = [start_date + i * freq_delta for i in range(n_samples)]

    # Generate price data with trend and volatility
    np.random.seed(42)

    # Starting price
    initial_price = 100.0

    # Random walk with trend
    returns = np.random.normal(trend, volatility, n_samples)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Add noise
    prices += np.random.normal(0, noise_level, n_samples)

    # Generate OHLC data from prices
    high_prices = prices + np.abs(np.random.normal(0, volatility * 0.5, n_samples))
    low_prices = prices - np.abs(np.random.normal(0, volatility * 0.5, n_samples))

    # Ensure high >= close >= low
    high_prices = np.maximum(high_prices, prices)
    low_prices = np.minimum(low_prices, prices)

    # Generate opening prices (previous close with noise)
    opening_prices = np.roll(prices, 1)
    opening_prices[0] = prices[0]
    opening_prices += np.random.normal(0, noise_level * 0.5, n_samples)

    # Generate volume data
    base_volume = 1000000
    volume_variation = np.random.lognormal(0, 0.5, n_samples)
    volumes = base_volume * volume_variation

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': timestamps,
        'open': opening_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes.astype(int)
    })

    # Set datetime as index
    df = df.set_index('datetime')

    return df

def generate_multi_symbol_data(
    symbols: list = ['AAPL', 'GOOGL', 'MSFT'],
    n_samples: int = 1000
) -> Dict[str, pd.DataFrame]:
    """Generate multi-symbol data for testing."""
    data = {}
    for symbol in symbols:
        data[symbol] = generate_sample_ohlcv_data(
            n_samples=n_samples,
            trend=np.random.uniform(-0.0001, 0.0001),
            volatility=np.random.uniform(0.01, 0.03)
        )
    return data

def generate_correlated_features(
    n_samples: int = 1000,
    correlation_matrix: np.ndarray = None
) -> pd.DataFrame:
    """Generate correlated features for testing."""
    if correlation_matrix is None:
        # Default 3x3 correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])

    # Generate correlated normal variables
    mean = np.zeros(len(correlation_matrix))
    cov = correlation_matrix * 0.01  # Scale covariance

    np.random.seed(42)
    data = np.random.multivariate_normal(mean, cov, n_samples)

    return pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
```

**Mock Objects (fixtures/mock_models.py)**:
```python
# src/testing/fixtures/mock_models.py
from unittest.mock import Mock, MagicMock
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

class MockGaussianHMM:
    """Mock HMM model for testing."""

    def __init__(self, n_components=3, covariance_type="full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.means_ = np.random.rand(n_components, 1)
        self.covars_ = np.random.rand(n_components, n_components)
        self.transmat_ = np.random.rand(n_components, n_components)
        self.startprob_ = np.random.rand(n_components)
        self.monitor_ = Mock()
        self.monitor_.converged = True

    def fit(self, X, lengths=None):
        # Mock fitting logic
        return self

    def predict(self, X):
        # Mock prediction logic
        return np.random.randint(0, self.n_components, size=len(X))

    def score(self, X):
        # Mock scoring logic
        return np.random.uniform(-1000, -100)

    def score_samples(self, X, lengths=None):
        # Mock sample scoring
        return np.random.uniform(-1000, -100, size=len(X))

class MockStandardScaler:
    """Mock StandardScaler for testing."""

    def __init__(self):
        self.mean_ = np.array([0.0])
        self.scale_ = np.array([1.0])
        self.fitted = False

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

class MockDataProcessor:
    """Mock data processor for testing."""

    def __init__(self):
        self.processed_data = None
        self.call_count = 0

    def process(self, data_path: str, config: dict = None):
        # Mock processing logic
        self.call_count += 1
        return generate_sample_ohlcv_data()

    def validate_data(self, data: pd.DataFrame):
        # Mock validation logic
        return {'is_valid': True, 'errors': [], 'warnings': []}

class MockProcessingEngine:
    """Mock processing engine for testing."""

    def __init__(self, engine_type="streaming"):
        self.engine_type = engine_type
        self.processed_count = 0

    def process(self, data_path: str, config: dict = None):
        # Mock processing logic
        self.processed_count += 1
        return generate_sample_ohlcv_data()
```

### 3. Test Utilities and Helpers

**Custom Assertions (utils/assertions.py)**:
```python
# src/testing/utils/assertions.py
import numpy as np
import pandas as pd
from typing import Any, Dict

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame,
                           check_dtype: bool = True, check_index: bool = True,
                           rtol: float = 1e-5, atol: float = 1e-8,
                           msg: str = None) -> None:
    """Assert that two DataFrames are equal."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype,
                                 check_index=check_index, rtol=rtol, atol=atol, msg=msg)

def assert_array_equal(arr1: np.ndarray, arr2: np.ndarray,
                        rtol: float = 1e-5, atol: float = 1e-8,
                        msg: str = None) -> None:
    """Assert that two arrays are equal."""
    np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol, err_msg=msg)

def assert_dict_structure(actual: Dict[str, Any], expected: Dict[str, Any],
                        msg: str = None) -> None:
    """Assert that two dictionaries have the same structure."""
    assert set(actual.keys()) == set(expected.keys()), \
        f"Dictionary keys differ. Expected: {set(expected.keys())}, Actual: {set(actual.keys())}"

    for key in expected.keys():
        assert type(actual[key]) == type(expected[key]), \
            f"Type mismatch for key '{key}'. Expected: {type(expected[key])}, Actual: {type(actual[key])}"
        if isinstance(expected[key], dict):
            assert_dict_structure(actual[key], expected[key])

def assert_valid_ohlcv_data(df: pd.DataFrame, msg: str = None) -> None:
    """Assert that DataFrame has valid OHLCV structure."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, \
        f"Missing required columns: {missing_columns}"

    # Check data types
    for col in ['open', 'high', 'low', 'close']:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"Column '{col}' must be numeric"

    assert df['volume'].dtype in ['int64', 'int32'], \
        f"Volume column must be integer type"

    # Check price relationships
    price_violations = (df['high'] < df['low']).sum()
    assert price_violations == 0, \
        f"High prices must be >= low prices. Found {price_violations} violations"

    # Check for negative values
    negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any().any()
    assert not negative_prices, \
        "Price columns must be non-negative"

    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in data")

def assert_hmm_states_valid(states: np.ndarray, n_states: int,
                          msg: str = None) -> None:
    """Assert that HMM states are valid."""
    assert states.min() >= 0, "States must be non-negative"
    assert states.max() < n_states, f"States must be < {n_states}"
    assert len(states) > 0, "States array must not be empty"
```

**Test Data Generators (utils/generators.py)**:
```python
# src/testing/utils/generators.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def generate_config_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Generate configuration overrides for testing."""
    base_config = {
        'analysis': {
            'n_states': 3,
            'covariance_type': 'full',
            'random_state': 42,
            'test_size': 0.2
        },
        'processing': {
            'engine': 'streaming',
            'chunk_size': 100000,
            'memory_limit': '8GB'
        },
        'features': {
            'indicators': [],
            'validation': True,
            'preprocessing': True
        },
        'backtesting': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0001
        },
        'visualization': {
            'chart_style': 'seaborn',
            'save_plots': True,
            'interactive': False
        },
        'logging': {
            'level': 'INFO',
            'console_output': True
        },
        'performance': {
            'monitoring': False,
            'memory_threshold': 0.8
        }
    }

    # Deep merge overrides
    return deep_merge(base_config, overrides)

def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base_config.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def generate_test_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for comprehensive testing."""
    scenarios = [
        # Basic configuration
        {
            'name': 'basic_default',
            'config': {
                'analysis': {'n_states': 3},
                'processing': {'engine': 'streaming'}
            }
        },

        # Large dataset configuration
        {
            'name': 'large_dataset',
            'config': {
                'analysis': {'n_states': 5},
                'processing': {
                    'engine': 'dask',
                    'chunk_size': 1000000,
                    'parallel_workers': 4
                }
            }
        },

        # Memory-constrained configuration
        {
            'name': 'memory_constrained',
            'config': {
                'analysis': {'n_states': 2},
                'processing': {
                    'engine': 'daft',
                    'chunk_size': 50000
                },
                'performance': {
                    'memory_threshold': 0.6
                }
            }
        },

        # Advanced features configuration
        {
            'name': 'advanced_features',
            'config': {
                'models': {
                    'hmm': {'n_components': 4, 'covariance_type': 'full'},
                    'lstm': {
                        'sequence_length': 60,
                        'hidden_units': [64, 64],
                        'dropout_rate': 0.3
                    }
                },
                'backtesting': {
                    'strategies': [
                        {
                            'name': 'aggressive',
                            'parameters': {'position_multiplier': 1.5}
                        },
                        {
                            'name': 'conservative',
                            'parameters': {'position_multiplier': 0.5}
                        }
                    ]
                }
            }
        },

        # Visualization configuration
        {
            'name': 'rich_visualization',
            'config': {
                'visualization': {
                    'interactive': True,
                    'save_plots': True,
                    'chart_style': 'plotly',
                    'color_scheme': 'viridis'
                }
            }
        },

        # Development configuration
        {
            'name': 'development',
            'config': {
                'logging': {
                    'level': 'DEBUG',
                    'file_path': './logs/debug.log'
                },
                'performance': {
                    'monitoring': True,
                    'profiling': True
                }
            }
        }
    ]

    return scenarios

def generate_test_config_files(temp_dir: Path) -> List[Path]:
    """Generate test configuration files."""
    config_files = []

    # YAML configuration file
    yaml_config = temp_dir / "test_config.yaml"
    yaml_content = {
        'analysis': {
            'n_states': 4,
            'covariance_type': 'diag'
        },
        'processing': {
            'engine': 'dask',
            'chunk_size': 50000
        }
    }

    import yaml
    with open(yaml_config, 'w') as f:
        yaml.dump(yaml_content, f)
    config_files.append(yaml_config)

    # JSON configuration file
    json_config = temp_dir / "test_config.json"
    json_content = {
        'analysis': {
            'n_states': 3,
            'covariance_type': 'full'
        },
        'features': {
            'indicators': [
                {'name': 'atr', 'enabled': True}
            ]
        }
    }

    import json
    with open(json_config, 'w') as f:
        json.dump(json_content, f, indent=2)
    config_files.append(json_config)

    return config_files
```

---

## Testing Types and Strategies

### 1. Unit Testing Framework

**Unit Testing Philosophy**:
- **Isolation**: Test individual components in isolation
- **Mocking**: Use mocks for external dependencies
- **Fast Execution**: Unit tests should run quickly (<1 second per test)
- **Comprehensive Coverage**: Test all public interfaces and edge cases

**Unit Test Structure**:
```python
# src/testing/unit/test_data_processing/test_csv_parser.py
import pytest
from src.testing.fixtures import sample_ohlcv_data, mock_data_processor
from src.data_processing.csv_parser import CSVParser

class TestCSVParser:
    """Test CSV parser functionality."""

    def test_parse_standard_format(self, sample_ohlcv_data):
        """Test parsing standard OHLCV format."""
        # Setup
        parser = CSVParser()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        # Execute
        result = parser.parse_csv(csv_path)

        # Verify
        assert_valid_ohlcv_data(result)
        assert len(result) == len(sample_ohlcv_data)

    def test_parse_alternative_format(self, sample_ohlcv_data):
        """Test parsing alternative CSV formats."""
        parser = CSVParser()

        # Convert to alternative format
        alt_data = sample_ohlcv_data.reset_index()
        alt_data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Last', 'Volume']

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            alt_data.to_csv(f, index=False)
            csv_path = f.name

        # Execute
        result = parser.parse_csv(csv_path)

        # Verify
        assert_valid_ohlcv_data(result)

    @pytest.mark.parametrize("file_format", ["standard", "alternative"])
    def test_csv_format_detection(self, file_format, sample_ohlcv_data):
        """Test CSV format detection."""
        parser = CSVParser()

        if file_format == "standard":
            csv_content = sample_ohlcv_data.to_csv()
        else:
            alt_data = sample_ohlcv_data.reset_index()
            alt_data.columns = ['Date', 'Time', 'Open', 'high', 'Low', 'Last', 'Volume']
            csv_content = alt_data.to_csv(index=False)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        detected_format = parser.detect_format(csv_path)
        assert detected_format == file_format

    def test_data_validation(self, sample_ohlcv_data):
        """Test data validation functionality."""
        parser = CSVParser()

        # Introduce validation errors
        invalid_data = sample_ohlcv_data.copy()
        invalid_data.loc[0, 'close'] = -1.0  # Negative price
        invalid_data.loc[1, 'volume'] = -100  # Negative volume

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f)
            csv_path = f.name

        # Execute validation
        result = parser.validate_data(invalid_data)

        # Verify validation errors
        assert not result['is_valid']
        assert len(result['errors']) > 0
        assert "negative price" in str(result['errors']).lower()

    def test_memory_efficiency(self, large_dataset_path):
        """Test memory efficiency with large datasets."""
        parser = CSVParser()

        # Monitor memory usage
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss

        # Process large dataset
        result = parser.parse_csv(large_dataset_path)

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Verify reasonable memory usage
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
        assert len(result) > 1000  # Processed sufficient data

    def test_error_handling(self, invalid_csv_path):
        """Test error handling for invalid files."""
        parser = CSVParser()

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            parser.parse_csv("nonexistent.csv")

        # Test corrupted file
        with pytest.raises(Exception):
            parser.parse_csv(invalid_csv_path)

    def test_chunk_processing(self, sample_ohlcv_data):
        """Test chunked CSV processing."""
        parser = CSVParser()

        # Create large CSV file
        large_data = pd.concat([sample_ohlcv_data] * 10)  # 10x larger dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f)
            csv_path = f.name

        # Test chunked processing
        chunks = list(parser.read_csv_chunks(csv_path, chunk_size=100))

        # Verify chunk processing
        assert len(chunks) > 1
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == len(large_data)

        # Verify data continuity
        for i in range(len(chunks) - 1):
            assert not chunks[i].empty
            # Verify timestamp continuity
            last_row_time = chunks[i].index[-1]
            first_row_time = chunks[i + 1].index[0]
            assert first_row_time > last_row_time
```

### 2. Integration Testing Framework

**Integration Testing Philosophy**:
- **End-to-End**: Test complete workflows from start to finish
- **Real Data**: Use realistic data where possible
- **Component Interaction**: Test integration between modules
- **Error Propagation**: Test error handling across component boundaries

**Integration Test Structure**:
```python
# src/testing/integration/test_pipelines.py
import pytest
from src.testing.fixtures import sample_ohlcv_data, test_config
from src.data_processing.csv_parser import CSVParser
from src.data_processing.feature_engineering import FeatureEngineer
from src.hmm_models.factory import HMMModelFactory
from src.model_training.hmm_trainer import HMMTrainer
from src.backtesting.strategy_engine import StrategyEngine

class TestAnalysisPipeline:
    """Test complete analysis pipeline integration."""

    def test_end_to_end_analysis(self, sample_ohlcv_data, test_config):
        """Test complete analysis from data to results."""
        # Setup
        config = test_config

        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        # Step 1: Data Loading
        parser = CSVParser()
        raw_data = parser.parse_csv(csv_path)
        assert len(raw_data) > 0

        # Step 2: Feature Engineering
        engineer = FeatureEngineer(config['features'])
        features = engineer.add_features(raw_data)
        assert len(features.columns) > len(raw_data.columns)
        assert not features.isnull().any().any()

        # Step 3: Model Creation
        factory = HMMModelFactory()
        hmm_config = config['models']['hmm']
        model = factory.create_model('gaussian', hmm_config)
        assert model is not None

        # Step 4: Model Training
        trainer = HMMTrainer()
        training_data = features[['log_ret', 'atr']].values
        trained_model = trainer.train_model(training_data, model)
        assert trained_model is not None

        # Step 5: State Prediction
        states = trained_model.predict(training_data)
        assert len(states) == len(training_data)
        assert states.min() >= 0
        assert states.max() < hmm_config['n_components']

        # Step 6: Backtesting
        strategy_engine = StrategyEngine(config['backtesting'])
        backtest_result = strategy_engine.backtest_strategy(
            raw_data.iloc[len(raw_data) - len(states):],
            states,
            {"0": 1, "1": 0, "2": -1}  # Simple state mapping
        )
        assert backtest_result is not None
        assert len(backtest_result.trades) >= 0

    def test_pipeline_with_different_engines(self, sample_ohlcv_data, test_config):
        """Test pipeline with different processing engines."""
        engines = ['streaming', 'dask', 'daft']

        for engine_type in engines:
            config['processing']['engine'] = engine_type

            # Run pipeline with specific engine
            self.test_end_to_end_analysis(sample_ohlcv_data, config)

    def test_pipeline_error_handling(self, test_config):
        """Test error handling throughout pipeline."""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'invalid_col': [1, 2, 3]
        })

        with pytest.raises(Exception):
            self.test_end_to_end_analysis(invalid_data, test_config)

    def test_pipeline_with_configuration_overrides(self, sample_ohlcv_data, test_config):
        """Test pipeline with configuration overrides."""
        # Override configuration
        overrides = {
            'analysis': {'n_states': 5},
            'features': {
                'indicators': [
                    {'name': 'custom_indicator', 'enabled': True, 'parameters': {'period': 20}}
                ]
            }
        }

        # Update configuration
        updated_config = deep_merge(test_config, overrides)

        # Run pipeline with overrides
        self.test_end_to_end_analysis(sample_ohlcv_data, updated_config)

class TestCLIWorkflows:
    """Test CLI workflow integration."""

    def test_cli_analyze_command(self, sample_ohlcv_data, test_config):
        """Test CLI analyze command integration."""
        from src.cli.main import cli
        from click.testing import CliRunner

        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        # Run CLI command
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--config-file', 'test_config.yaml',
            'analyze',
            '--input-csv', csv_path,
            '--n-states', '4',
            '--output-dir', 'test_output'
        ])

        # Verify CLI execution
        assert result.exit_code == 0
        assert 'Analysis completed' in result.output

        # Verify output files created
        output_dir = Path('test_output')
        assert output_dir.exists()
        assert (output_dir / 'states.csv').exists() or
                (output_dir / 'model.pkl').exists() or
                (output_dir / 'report.html').exists())

    def test_cli_with_different_modes(self, sample_ohlcv_data, test_config):
        """Test CLI with different operation modes."""
        modes = ['simple', 'standard', 'advanced']

        for mode in modes:
            # Update configuration for mode
            mode_config = test_config.copy()
            if mode == 'simple':
                mode_config['features']['indicators'] = [
                    {'name': 'log_returns', 'enabled': True}
                ]
            elif mode == 'advanced':
                mode_config['performance']['monitoring'] = True
                mode_config['visualization']['interactive'] = True

            # Run CLI with mode
            self.test_cli_analyze_command(sample_ohlcv_data, mode_config)
```

### 3. Performance Testing Framework

**Performance Testing Philosophy**:
- **Benchmarking**: Systematic performance measurement
- **Resource Monitoring**: Memory and CPU usage tracking
- **Scalability Testing**: Large dataset handling
- **Regression Prevention**: Performance regression detection

**Performance Test Structure**:
```python
# src/testing/performance/test_benchmarks.py
import pytest
import time
import psutil
from src.testing.fixtures import large_sample_data, test_config
from src.processing.engines.factory import ProcessingEngineFactory

class TestPerformanceBenchmarks:
    """Test performance benchmarks for different components."""

    @pytest.mark.performance
    def test_processing_engine_performance(self, large_sample_data, test_config):
        """Benchmark processing engine performance."""
        engines = ['streaming', 'dask', 'daft']
        results = {}

        for engine_type in engines:
            config = test_config.copy()
            config['processing']['engine'] = engine_type

            # Initialize engine
            factory = ProcessingEngineFactory()
            engine = factory.create_engine(engine_type)

            # Benchmark processing
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            result = engine.process(large_sample_data, config['processing'])

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            results[engine_type] = {
                'processing_time': end_time - start_time,
                'memory_usage': end_memory - start_memory,
                'rows_processed': len(result)
            }

        # Verify performance requirements
        for engine_type, metrics in results.items():
            assert metrics['processing_time'] < 30.0, f"{engine_type} processing too slow"
            assert metrics['memory_usage'] < 500 * 1024 * 1024, f"{engine_type} using too much memory"

    @pytest.mark.performance
    def test_hmm_training_performance(self, large_sample_data, test_config):
        """Benchmark HMM training performance."""
        n_states_values = [2, 3, 4, 5]
        results = {}

        for n_states in n_states_values:
            config = test_config.copy()
            config['models']['hmm']['n_components'] = n_states

            # Initialize model
            factory = HMMModelFactory()
            model = factory.create_model('gaussian', config['models']['hmm'])

            # Prepare training data
            features = large_sample_data[['log_ret', 'atr']].values

            # Benchmark training
            start_time = time.time()
            model.fit(features)
            end_time = time.time()

            results[n_states] = {
                'training_time': end_time - start_time,
                'convergence': model.monitor_.converged,
                'log_likelihood': model.score(features)
            }

        # Verify performance requirements
        for n_states, metrics in results.items():
            assert metrics['training_time'] < 60.0, f"HMM training with {n_states} states too slow"
            assert metrics['convergence'] == True, f"HMM with {n_states} states failed to converge"

    @pytest.mark.performance
    def test_memory_usage_with_different_chunk_sizes(self, test_config):
        """Test memory usage with different chunk sizes."""
        chunk_sizes = [1000, 10000, 100000, 1000000]
        results = {}

        for chunk_size in chunk_sizes:
            config = test_config.copy()
            config['processing']['chunk_size'] = chunk_size

            # Initialize engine
            factory = ProcessingEngineFactory()
            engine = factory.create_engine('streaming')

            # Monitor memory usage
            process = psutil.Process()
            start_memory = process.memory_info().rss
            max_memory = process.memory_info().rss

            # Process data with specific chunk size
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                # Generate large dataset
                large_data = generate_sample_ohlcv_data(n_samples=10000)
                large_data.to_csv(f)
                csv_path = f.name

            result = engine.process(csv_path, config['processing'])

            end_memory = process.memory_info().rss
            max_memory = max(max_memory, end_memory)

            results[chunk_size] = {
                'max_memory': max_memory,
                'peak_memory_increase': max_memory - start_memory,
                'rows_processed': len(result)
            }

        # Verify memory efficiency
        for chunk_size, metrics in results.items.items():
            assert metrics['peak_memory_increase'] < 200 * 1024 * 1024, \
                f"Chunk size {chunk_size} uses too much memory"

    @pytest.mark.performance
    def test_scalability_with_dataset_size(self, test_config):
        """Test system scalability with different dataset sizes."""
        dataset_sizes = [1000, 10000, 100000, 1000000]
        results = {}

        for size in dataset_sizes:
            config = test_config.copy()

            # Generate dataset
            data = generate_sample_ohlcv_data(n_samples=size)

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                data.to_csv(f)
                csv_path = f.name

            # Benchmark processing
            start_time = time.time()
            factory = ProcessingEngineFactory()
            engine = factory.create_engine('dask')

            result = engine.process(csv_path, config['processing'])

            end_time = time.time()

            results[size] = {
                'processing_time': end_time - start_time,
                'rows_processed': len(result)
            }

        # Verify linear scaling (approximately)
        time_per_row_1000 = results[1000]['processing_time'] / 1000
        for size, metrics in results.items():
            expected_time = time_per_row_1000 * size
            assert metrics['processing_time'] < expected_time * 2.0, \
                f"Processing {size} rows took too long: {metrics['processing_time']}s (expected < {expected_time:.2f}s)"
```

### 4. Compatibility Testing Framework

**Compatibility Testing Philosophy**:
- **Migration Validation**: Ensure migrated functionality works correctly
- **Backward Compatibility**: Preserve existing user workflows
- **API Compatibility**: Maintain consistent interfaces
- **Data Format Compatibility**: Support existing data formats

**Compatibility Test Structure**:
```python
# src/testing/compatibility/test_main_directory.py
import pytest
import subprocess
import tempfile
from src.testing.fixtures import sample_ohlcv_data
from src.cli.main import cli

class TestMainDirectoryCompatibility:
    """Test compatibility with main directory functionality."""

    def test_main_py_command_line_interface(self, sample_ohlcv_data):
        """Test main.py command-line interface compatibility."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        # Run main.py script
        result = subprocess.run([
            'python', 'main.py',
            csv_path,
            '--n-states', '3',
            '--max-iter', '50'
        ], capture_output=True, text=True)

        # Verify execution
        assert result.returncode == 0
        assert 'Training Gaussian HMM' in result.stdout
        assert 'Model converged' in result.stdout

    def test_main_py_output_format(self, sample_ohlcv_data):
        """Test main.py output format compatibility."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        result = subprocess.run([
            'python', 'main.py',
            csv_path,
            '--model-out', 'test_model.pkl',
            '--plot'
        ], capture_output=True, text=True)

        # Verify output files
        assert Path('test_model.pkl').exists()
        assert Path(csv_path.with_suffix('.png')).exists()

    def test_cli_vs_main_py_compatibility(self, sample_ollcv_data, test_config):
        """Test CLI vs main.py output compatibility."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_ohlcv_data.to_csv(f)
            csv_path = f.name

        # Run CLI analysis
        cli_result = subprocess.run([
            'python', '-m', 'src.cli.main:cli',
            '--config-file', 'test_config.yaml',
            'analyze',
            '--input-csv', csv_path,
            '--output-dir', 'cli_output'
        ], capture_output=True, text=True)

        # Run main.py analysis
        main_result = subprocess.run([
            'python', 'main.py',
            csv_path,
            '--n-states', '3',
            '--model-out', 'main_model.pkl'
        ], capture_output=True, text=True)

        # Both should complete successfully
        assert cli_result.returncode == 0
        assert main_result.returncode == 0

        # Both should create model files
        assert Path('cli_output').exists()
        assert Path('main_model.pkl').exists()

    def test_configuration_file_compatibility(self, test_config):
        """Test configuration file compatibility across CLI and main directory."""
        # Create configuration file
        config_content = {
            'analysis': {
                'n_states': 4,
                'covariance_type': 'diag'
            },
            'processing': {
                'engine': 'dask'
            }
        }

        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_content, f)
            config_path = f.name

        # Test CLI with config file
        cli_result = subprocess.run([
            'python', '-m', 'src.cli.main:cli',
            '--config-file', config_path,
            'analyze',
            '--input-csv', 'test_data.csv'
        ], capture_output=True, text=True)

        assert cli_result.returncode == 0
```

---

## Test Automation and CI/CD

### Continuous Integration Pipeline

**GitHub Actions Workflow (.github/workflows/test.yml)**:
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist pytest-benchmark

      - name: Run unit tests
        run: |
          pytest src/testing/unit --cov=src --cov-report=xml --cov-report=html --cov-report=term

      - name: Run integration tests
        run: |
          pytest src/testing/integration --tb=short

      - name: Run performance tests
        run: |
          pytest src/testing/performance --benchmark-only --benchmark-json=benchmarks.json

      - name: Run compatibility tests
        run: |
          pytest src/testing/compatibility --tb=short

      - name: Upload coverage reports
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: # optional flags
          name: codecov-umbrella
```

### Local Development Testing

**Makefile for Testing**:
```makefile
# src/testing/Makefile
.PHONY: test test-unit test-integration test-performance test-compatibility test-all

# Run all tests
test-all:
	pytest src/testing/ --cov=src --cov-report=html --cov-report=term

# Run unit tests only
test-unit:
	pytest src/testing/unit --cov=src --cov-report=html

# Run integration tests
test-integration:
	pytest src/testing/integration --tb=short

# Run performance tests
test-performance:
	pytest src/testing/performance --benchmark-only

# Run compatibility tests
test-compatibility:
	pytest src/testing/compatibility --tb=short

# Generate coverage report
coverage:
	pytest src/testing/ --cov=src --cov-report=html --cov-report=term

# Clean test artifacts
clean:
	rm -rf src/testing/reports/*
	rm -rf .pytest_cache/
	rm -rf .coverage
```

### Test Data Management

**Test Data Generation and Management**:
```python
# src/testing/fixtures/__init__.py
from .sample_data import *
from .mock_models import *
from .test_configs import *

# Auto-generate test data on demand
def get_test_data(data_type: str, **kwargs):
    """Get test data of specified type."""
    if data_type == "ohlcv_small":
        return generate_sample_ohlcv_data(n_samples=100)
    elif data_type == "ohlcv_large":
        return generate_sample_ohlcv_data(n_samples=10000)
    elif data_type == "multi_symbol":
        return generate_multi_symbol_data()
    elif data_type == "correlated_features":
        return generate_correlated_features()
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def get_mock_model(model_type: str, **kwargs):
    """Get mock model of specified type."""
    if model_type == "hmm":
        return MockGaussianHMM()
    elif model_type == "scaler":
        return MockStandardScaler()
    elif model_type == "data_processor":
        return MockDataProcessor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

---

## Testing Success Metrics

### Coverage Requirements

**Target Coverage Metrics**:
- **Overall Coverage**: >90% for all code
- **Unit Test Coverage**: >95% for core modules
- **Integration Test Coverage**: >85% for workflows
- **Critical Path Coverage**: 100% for main functionality

**Coverage Breakdown by Module**:
- **Data Processing**: >95%
- **HMM Models**: >95%
- **Backtesting**: >90%
- **CLI Interface**: >90%
- **Configuration**: >95%
- **Visualization**: >85%
- **Deep Learning**: >90%

### Performance Requirements

**Performance Benchmarks**:
- **Unit Test Execution**: <5 seconds total
- **Integration Test Execution**: <30 seconds total
- **Performance Test Execution**: <2 minutes total
- **Memory Usage**: <100MB peak during testing

**Resource Requirements**:
- **Test Execution Time**: <5 minutes total for all tests
- **Memory Usage**: <200MB peak during testing
- **Disk Space**: <500MB for test artifacts

### Quality Requirements

**Test Quality Standards**:
- **Clear Test Names**: Descriptive and consistent naming
- **Test Documentation**: Comprehensive docstrings
- **Error Messages**: Clear and informative
- **Test Organization**: Logical grouping and structure
- **Mock Strategy**: Comprehensive mocking of dependencies

**Test Validation Standards**:
- **Data Validation**: Verify test data integrity
- **Result Validation**: Verify expected vs actual results
- **Edge Case Coverage**: Test boundary conditions
- **Error Scenarios**: Test error handling and recovery
- **Performance Scenarios**: Test with realistic data sizes

---

## Risk Mitigation for Testing

### High-Risk Testing Items

**1. Test Data Management**:
- **Risk**: Test data may become outdated or inconsistent
- **Mitigation**: Auto-generation with deterministic seeds
- **Validation**: Data validation in test fixtures
- **Version Control**: Test data versioning and tracking

**2. Mock Object Maintenance**:
- **Risk**: Mock objects may not match real implementations
- **Mitigation**: Regular mock validation against real interfaces
- **Validation**: Mock behavior testing
- **Maintenance**: Automated mock synchronization

**3. Performance Test Stability**:
- **Risk**: Performance tests may be flaky due to system load
- **Mitigation**: Baseline establishment and variance thresholds
- **Validation**: Statistical analysis of test results
- **Environment Control**: Consistent testing environments

### Medium-Risk Testing Items

**1. External Dependencies**:
- **Risk**: External dependencies may change or fail
- **Mitigation**: Docker containers for dependency isolation
- **Validation**: Dependency availability testing
- **Fallback**: Mock implementations for critical dependencies

**2. File System Dependencies**:
- **Risk**: Test file operations may fail in some environments
- **Mitigation**: Temporary file management
- **Validation**: File permission testing
- **Cleanup**: Comprehensive cleanup procedures

**3. CI/CD Pipeline**:
- **Risk**: Pipeline failures may block development
- **Mitigation: Multiple environment testing
- **Validation**: Pipeline dry-run validation
- **Fallback**: Local testing fallback procedures

---

## Testing Timeline and Implementation

### Phase 1: Framework Setup (12 hours)

**Task 1.1: Create Testing Infrastructure (4 hours)**
- Set up pytest configuration and conftest.py
- Create directory structure for tests
- Implement Makefile for test automation
- Set up GitHub Actions workflow

**Task 1.2: Create Test Fixtures (4 hours)**
- Implement sample data generation
- Create mock object factories
- Create test configuration utilities
- Add test data validation

**Task 1.3: Implement Test Utilities (4 hours)**
- Create custom assertion helpers
- Implement comparison utilities
- Add reporting utilities
- Create mock management systems

### Phase 2: Unit Test Implementation (25 hours)

**Task 2.1: Data Processing Unit Tests (8 hours)**
- Test CSV parser functionality
- Test feature engineering
- Test data validation
- Test data cleaning utilities

**Task 2.2: HMM Model Unit Tests (8 hours)**
- Test HMM model implementations
- Test model factory patterns
- Test training functionality
- Test inference and prediction

**Task 2.3: Backtesting Unit Tests (6 hours)**
- Test strategy engine
- Test performance analysis
- Test transaction cost modeling
- Test portfolio management

**Task 2.4: CLI Unit Tests (3 hours)**
- Test CLI command implementations
- Test configuration integration
- Test progress tracking
- Test output formatting

### Phase 3: Integration Test Implementation (20 hours)

**Task 3.1: Pipeline Integration Tests (10 hours)**
- Test end-to-end analysis workflows
- Test component integration
- Test error propagation
- Test configuration integration

**Task 3.2: CLI Workflow Tests (5 hours)**
- Test CLI command workflows
- Test configuration file usage
- Test different CLI modes
- Test CLI error handling

**Task 3.3: Engine Integration Tests (5 hours)**
- Test processing engine integration
- Test engine-specific features
- Test engine error handling
- Test performance variations

### Phase 4: Advanced Testing (15 hours)

**Task 4.1: Performance Tests (8 hours)**
- Implement performance benchmarks
- Test memory usage optimization
- Test scalability with large datasets
- Test profiling and optimization

**Task 4.2: Compatibility Tests (7 hours)**
- Test main directory compatibility
- Test CLI compatibility
- Test API compatibility
- Test data format compatibility

### Phase 5: Test Automation (10 hours)

**Task 5.1: CI/CD Setup (5 hours)**
- Configure GitHub Actions
- Set up automated testing
- Configure coverage reporting
- Set up test artifact management

**Task 5.2: Local Development Setup (5 hours)**
- Create development testing workflows
- Set up test data generation
- Create test debugging utilities
- Create test documentation

---

## Conclusion

The testing framework design provides a comprehensive, multi-layered testing strategy that will ensure the quality and reliability of the enhanced src directory architecture throughout the migration process and beyond.

**Key Testing Framework Features**:
- **Comprehensive Coverage**: Unit, integration, performance, and compatibility testing
- **Professional Tools**: Pytest-based with advanced features (coverage, benchmarking, xdist)
- **Mock Strategy**: Comprehensive mocking for external dependencies
- **Automation**: Full CI/CD integration with GitHub Actions
- **Performance Monitoring**: Systematic benchmarking and regression prevention

**Testing Framework Success Criteria**:
- **Coverage**: >90% overall code coverage
- **Performance**: Consistent performance with <5% variance
- **Quality**: Clear test documentation and error messages
- **Reliability**: Stable test execution with minimal flakiness

This testing framework will serve as the quality assurance foundation for the entire migration project, ensuring that all migrated functionality works correctly and performs as expected, while providing early detection of any regressions or issues.

---

*Testing Framework Design Completed: October 23, 2025*
*Phase 1 Analysis and Planning: Complete*