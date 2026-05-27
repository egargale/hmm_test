"""Integration tests: assert the three regime engines produce different outputs.

Issue #11: Validates that threshold, messina, and hmm engines are
genuinely independent analysis methods, not returning identical results.
"""

import pandas as pd
import pytest

from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
from hmm_futures_analysis.regime.pipeline import run as pipeline_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def btc_ohlcv(btc_csv):
    """Load BTC OHLCV DataFrame for HMM engine tests."""
    df = pd.read_csv(btc_csv, parse_dates=["Date"], index_col="Date")
    df.columns = [c.strip() for c in df.columns]
    return df[["Open", "High", "Low", "Last", "Volume"]].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Last": "close",
            "Volume": "volume",
        }
    )


@pytest.fixture
def btc_prices(btc_csv):
    """Load BTC close prices as pd.Series via the canonical loader."""
    return load_from_csv(btc_csv)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEngineIndependence:
    """Assert the three regime engines produce genuinely different outputs."""

    # -- Determinism --------------------------------------------------------

    def test_threshold_is_deterministic(self, btc_prices):
        """Same input → same output for threshold engine (deterministic)."""
        common = dict(source="test", engine="threshold", min_train=300)
        result_a = pipeline_run(btc_prices, **common)
        result_b = pipeline_run(btc_prices, **common)

        assert result_a["transition_matrix"] == result_b["transition_matrix"]
        assert result_a["signal"] == result_b["signal"]
        assert result_a["regime_counts"] == result_b["regime_counts"]

    # -- Transition matrix --------------------------------------------------

    def test_transition_matrix_differs_threshold_vs_hmm(
        self, btc_prices, btc_ohlcv
    ):
        """HMM engine produces a different transition matrix than threshold."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_hmm = pipeline_run(
            btc_prices, engine="hmm", ohlcv=btc_ohlcv, **common
        )
        assert (
            result_hmm["transition_matrix"]
            != result_threshold["transition_matrix"]
        )

    def test_transition_matrix_differs_threshold_vs_messina(
        self, btc_prices, btc_ohlcv
    ):
        """Messina engine produces a different transition matrix than threshold."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_messina = pipeline_run(
            btc_prices, engine="messina", ohlcv=btc_ohlcv, **common
        )
        assert (
            result_messina["transition_matrix"]
            != result_threshold["transition_matrix"]
        )

    # -- Signal -------------------------------------------------------------

    def test_signal_differs_threshold_vs_hmm(self, btc_prices, btc_ohlcv):
        """Signal from threshold must differ from hmm engine."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_hmm = pipeline_run(
            btc_prices, engine="hmm", ohlcv=btc_ohlcv, **common
        )
        assert result_threshold["signal"] != result_hmm["signal"]

    def test_signal_differs_threshold_vs_messina(self, btc_prices, btc_ohlcv):
        """Signal from threshold must differ from messina engine."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_messina = pipeline_run(
            btc_prices, engine="messina", ohlcv=btc_ohlcv, **common
        )
        assert result_threshold["signal"] != result_messina["signal"]

    # -- Regime counts ------------------------------------------------------

    def test_regime_counts_differs_threshold_vs_hmm(
        self, btc_prices, btc_ohlcv
    ):
        """Regime counts from threshold must differ from hmm engine."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_hmm = pipeline_run(
            btc_prices, engine="hmm", ohlcv=btc_ohlcv, **common
        )
        assert result_threshold["regime_counts"] != result_hmm["regime_counts"]

    def test_regime_counts_differs_threshold_vs_messina(
        self, btc_prices, btc_ohlcv
    ):
        """Regime counts from threshold must differ from messina engine."""
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(btc_prices, engine="threshold", **common)
        result_messina = pipeline_run(
            btc_prices, engine="messina", ohlcv=btc_ohlcv, **common
        )
        assert (
            result_threshold["regime_counts"]
            != result_messina["regime_counts"]
        )
