"""
Pytest configuration and fixtures for HMM futures analysis testing.

This module provides common fixtures and configuration for all test modules,
including data generation, temporary directories, and test utilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Generator, Dict, Any
import warnings

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory path."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session")
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    close_prices = base_price * np.cumprod(1 + returns)

    high_spread = np.random.uniform(0.005, 0.02, 100)
    low_spread = np.random.uniform(0.005, 0.02, 100)

    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices * (1 + high_spread),
        'low': close_prices * (1 - low_spread),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)

    return data


@pytest.fixture(scope="session")
def sample_ohl_csv_file(sample_ohlcv_data: pd.DataFrame, test_data_dir: Path) -> Path:
    """Create a sample CSV file with OHLCV data."""
    csv_path = test_data_dir / "sample_ohlcv.csv"
    sample_ohlcv_data.to_csv(csv_path)
    return csv_path


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def random_seed() -> int:
    """Return a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def configure_warnings():
    """Configure warnings for test runs."""
    # Filter out common warnings that don't affect test functionality
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
    warnings.filterwarnings("ignore", message=".*PytestReturnNotNoneWarning.*")


@pytest.fixture
def hmm_config() -> Dict[str, Any]:
    """Return default HMM configuration for testing."""
    return {
        'n_components': 3,
        'covariance_type': 'full',
        'n_iter': 100,
        'tol': 1e-4,
        'random_state': 42
    }


@pytest.fixture
def backtest_config() -> Dict[str, Any]:
    """Return default backtest configuration for testing."""
    return {
        'initial_capital': 100000.0,
        'commission': 0.001,
        'slippage': 0.0001,
        'lookahead_bias_prevention': True,
        'lookahead_days': 1
    }


@pytest.fixture
def feature_config() -> Dict[str, Any]:
    """Return default feature engineering configuration."""
    return {
        'returns': {'periods': [1, 5, 10]},
        'moving_averages': {'periods': [5, 10, 20]},
        'volatility': {'periods': [14]},
        'momentum': {'periods': [14]},
        'volume': {'enabled': True}
    }


@pytest.fixture
def sample_states(sample_ohlcv_data: pd.DataFrame) -> np.ndarray:
    """Create sample HMM states for testing."""
    np.random.seed(42)
    n_samples = len(sample_ohlcv_data)
    return np.random.choice([0, 1, 2], size=n_samples)


@pytest.fixture
def sample_features(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample engineered features for testing."""
    data = sample_ohlcv_data.copy()

    # Add basic features
    data['returns'] = data['close'].pct_change()

    # Add moving averages
    for period in [5, 10, 20]:
        data[f'sma_{period}'] = data['close'].rolling(window=period).mean()

    # Add volatility
    data['volatility_14'] = data['returns'].rolling(window=14).std()

    # Add momentum
    data['momentum_14'] = data['close'].pct_change(14)

    # Drop NaN values
    data = data.dropna()

    return data


@pytest.fixture
def sample_trade_data() -> pd.DataFrame:
    """Create sample trade data for backtesting tests."""
    dates = pd.date_range('2020-01-01', periods=20, freq='D')

    return pd.DataFrame({
        'entry_time': dates[::2],
        'exit_time': dates[1::2],
        'entry_price': np.random.uniform(95, 105, 10),
        'exit_price': np.random.uniform(95, 105, 10),
        'size': np.random.uniform(0.5, 2.0, 10),
        'pnl': np.random.uniform(-5, 5, 10),
        'commission': np.full(10, 1.0),
        'slippage': np.full(10, 0.5)
    })


@pytest.fixture(scope="session")
def coverage_config():
    """Configure coverage reporting settings."""
    return {
        'fail_under': 95,
        'show_missing': True,
        'omit': [
            '*/tests/*',
            '*/test_*',
            'setup.py',
            '*/conftest.py'
        ]
    }