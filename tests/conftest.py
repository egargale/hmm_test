"""Shared fixtures for hmm_test integration tests."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent


def run_regime(*args):
    """Run hmm_futures_analysis/cli.py with args, return CompletedProcess."""
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return result


@pytest.fixture
def btc_csv():
    """Path to BTC.csv test data."""
    p = ROOT / "test_data" / "BTC.csv"
    if not p.exists():
        pytest.skip("BTC.csv not available")
    return str(p)


@pytest.fixture
def futures_csv():
    """Path to test_futures.csv test data."""
    p = ROOT / "test_data" / "test_futures.csv"
    if not p.exists():
        pytest.skip("test_futures.csv not available")
    return str(p)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0.05, 1.5, n))
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.normal(0, 0.5, n),
        "high": close + abs(np.random.normal(1, 0.5, n)),
        "low": close - abs(np.random.normal(1, 0.5, n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })
