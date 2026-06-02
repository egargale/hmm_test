"""Shared fixtures for hmm_test integration tests."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default timeout (seconds) for CLI integration tests.
CLI_TIMEOUT = 60


def run_regime(*args, timeout=CLI_TIMEOUT):
    """Run hmm_futures_analysis/cli.py with args, return CompletedProcess.

    Raises AssertionError with a descriptive message if the subprocess
    exceeds *timeout* seconds.
    """
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"CLI command timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc
    return result


@pytest.fixture(scope="session")
def btc_csv():
    """Path to BTC.csv test data."""
    p = ROOT / "test_data" / "BTC.csv"
    if not p.exists():
        pytest.skip("BTC.csv not available")
    return str(p)


@pytest.fixture(scope="session")
def futures_csv():
    """Path to test_futures.csv test data."""
    p = ROOT / "test_data" / "test_futures.csv"
    if not p.exists():
        pytest.skip("test_futures.csv not available")
    return str(p)


@pytest.fixture(scope="session")
def sample_ohlcv():
    """Load SPY OHLCV sample for fast, realistic HMM tests."""
    p = ROOT / "test_data" / "SPY.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.astype(float)
