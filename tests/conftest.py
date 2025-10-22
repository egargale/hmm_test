"""
Pytest configuration and shared fixtures for HMM Futures Analysis tests.

This module provides common test utilities, fixtures, and configuration
for all test types (unit, integration, e2e).
"""

import shutil

# Add src to path for imports
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import project modules for testing
from utils import HMMConfig, ProcessingConfig


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory path."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session")
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    close_prices = base_price * np.cumprod(1 + returns)

    high_spread = np.random.uniform(0.005, 0.02, 100)
    low_spread = np.random.uniform(0.005, 0.02, 100)

    data = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices * (1 + high_spread),
            "low": close_prices * (1 - low_spread),
            "close": close_prices,
            "volume": np.random.uniform(1000, 5000, 100),
        },
        index=dates,
    )

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
        "n_components": 3,
        "covariance_type": "full",
        "n_iter": 100,
        "tol": 1e-4,
        "random_state": 42,
    }


@pytest.fixture
def backtest_config() -> Dict[str, Any]:
    """Return default backtest configuration for testing."""
    return {
        "initial_capital": 100000.0,
        "commission": 0.001,
        "slippage": 0.0001,
        "lookahead_bias_prevention": True,
        "lookahead_days": 1,
    }


@pytest.fixture
def feature_config() -> Dict[str, Any]:
    """Return default feature engineering configuration."""
    return {
        "returns": {"periods": [1, 5, 10]},
        "moving_averages": {"periods": [5, 10, 20]},
        "volatility": {"periods": [14]},
        "momentum": {"periods": [14]},
        "volume": {"enabled": True},
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
    data["returns"] = data["close"].pct_change()

    # Add moving averages
    for period in [5, 10, 20]:
        data[f"sma_{period}"] = data["close"].rolling(window=period).mean()

    # Add volatility
    data["volatility_14"] = data["returns"].rolling(window=14).std()

    # Add momentum
    data["momentum_14"] = data["close"].pct_change(14)

    # Drop NaN values
    data = data.dropna()

    return data


@pytest.fixture
def sample_trade_data() -> pd.DataFrame:
    """Create sample trade data for backtesting tests."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")

    return pd.DataFrame(
        {
            "entry_time": dates[::2],
            "exit_time": dates[1::2],
            "entry_price": np.random.uniform(95, 105, 10),
            "exit_price": np.random.uniform(95, 105, 10),
            "size": np.random.uniform(0.5, 2.0, 10),
            "pnl": np.random.uniform(-5, 5, 10),
            "commission": np.full(10, 1.0),
            "slippage": np.full(10, 0.5),
        }
    )


@pytest.fixture(scope="session")
def coverage_config():
    """Configure coverage reporting settings."""
    return {
        "fail_under": 95,
        "show_missing": True,
        "omit": ["*/tests/*", "*/test_*", "setup.py", "*/conftest.py"],
    }


@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Return a test ProcessingConfig instance."""
    return ProcessingConfig(
        engine_type="streaming",
        chunk_size=100,
        indicators={"sma_5": {"window": 5}, "sma_10": {"window": 10}, "returns": {}},
    )


@pytest.fixture
def hmm_config_object() -> HMMConfig:
    """Return a test HMMConfig instance."""
    return HMMConfig(
        n_states=3,
        covariance_type="full",
        n_iter=50,  # Reduced for test speed
        random_state=42,
        tol=1e-3,
        num_restarts=2,  # Reduced for test speed
    )


# Test utilities
def validate_dataframe(
    df: pd.DataFrame, min_rows: int = 1, required_columns: list = None
):
    """Utility to validate DataFrame structure in tests."""
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        assert not missing_cols, f"Missing required columns: {missing_cols}"

    # Check for NaN values in critical columns
    if required_columns:
        for col in required_columns:
            if col in df.columns:
                assert (
                    not df[col].isna().all()
                ), f"Column {col} contains only NaN values"


def validate_hmm_model(model, n_states: int):
    """Utility to validate HMM model in tests."""
    assert hasattr(model, "n_components"), "Model missing n_components attribute"
    assert (
        model.n_components == n_states
    ), f"Expected {n_states} states, got {model.n_components}"
    assert hasattr(model, "means_"), "Model appears not to be trained (missing means_)"
    assert hasattr(
        model, "covars_"
    ), "Model appears not to be trained (missing covars_)"


def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function for performance testing."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def check_memory_usage():
    """Get current memory usage for testing."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return None


# Skip conditions for different environments
def skip_if_no_internet():
    """Skip test if no internet connection is available."""
    try:
        import urllib.request

        urllib.request.urlopen("http://www.google.com", timeout=1)
        return False
    except:
        return True


def skip_if_windows():
    """Skip test on Windows platform."""
    return sys.platform.startswith("win")


# Test configuration constants
TEST_CONFIG = {
    "default_n_states": 3,
    "test_data_size": 1000,
    "max_test_duration": 300,  # 5 minutes
    "memory_limit_mb": 1000,  # 1GB
}


# Custom test data generator
class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_ohlcv_data(
        n_samples: int = 1000,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001,
    ) -> pd.DataFrame:
        """Create synthetic OHLCV data with specified parameters."""
        np.random.seed(42)

        # Generate price series with trend and volatility
        returns = np.random.normal(trend, volatility, n_samples)
        prices = start_price * np.cumprod(1 + returns)

        # Generate intraday variation
        intraday_range = np.random.uniform(0.01, 0.03, n_samples)

        data = {
            "open": prices,
            "high": prices * (1 + intraday_range),
            "low": prices * (1 - intraday_range),
            "close": prices * np.roll(1 + np.random.normal(0, 0.005, n_samples), 1),
            "volume": np.random.lognormal(10, 1, n_samples).astype(int),
        }

        # Fix relationships
        df = pd.DataFrame(data)
        df.iloc[0, df.columns.get_loc("open")] = start_price
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        df["close"] = df["close"].fillna(df["open"])

        # Add datetime index
        df.index = pd.date_range("2020-01-01", periods=n_samples, freq="1D")

        return df

    @staticmethod
    def create_features_from_ohlcv(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from OHLCV data using the feature engineering module."""
        from data_processing.feature_engineering import add_features

        return add_features(ohlcv_data.copy())
