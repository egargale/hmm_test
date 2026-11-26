"""Shared fixtures for hmm_test integration tests."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Default timeout (seconds) for CLI integration tests. 60s fits a fast dev box but
# not shared CI or a cloud sandbox: fshmm alone takes ~92s on 0700_HK. Override
# with HMM_CLI_TIMEOUT instead of editing this constant per machine.
CLI_TIMEOUT = int(os.environ.get("HMM_CLI_TIMEOUT", "60"))


def run_regime(*args, timeout=CLI_TIMEOUT, env=None):
    """Run hmm_futures_analysis/cli.py with args, return CompletedProcess.

    Raises AssertionError with a descriptive message if the subprocess
    exceeds *timeout* seconds.  *env* is an optional mapping of extra
    environment variables merged over the current process environment for
    the subprocess (e.g. ``{"OMP_NUM_THREADS": "1"}``).
    """
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    run_env = {**os.environ, **env} if env else None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=timeout,
            env=run_env,
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
def spy_csv():
    """Path to SPY.csv test data."""
    p = ROOT / "test_data" / "SPY.csv"
    if not p.exists():
        pytest.skip("SPY.csv not available")
    return str(p)


@pytest.fixture(scope="session")
def nvda_csv():
    """Path to NVDA.csv test data."""
    p = ROOT / "test_data" / "NVDA.csv"
    if not p.exists():
        pytest.skip("NVDA.csv not available")
    return str(p)


@pytest.fixture(scope="session")
def ko_csv():
    """Path to KO.csv test data."""
    p = ROOT / "test_data" / "KO.csv"
    if not p.exists():
        pytest.skip("KO.csv not available")
    return str(p)


@pytest.fixture(scope="session")
def xle_csv():
    """Path to XLE.csv test data."""
    p = ROOT / "test_data" / "XLE.csv"
    if not p.exists():
        pytest.skip("XLE.csv not available")
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
