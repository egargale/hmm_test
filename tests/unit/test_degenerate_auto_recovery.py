"""Tests for degenerate-fit auto-downgrade in _hmm_classify_pipeline.

Issue #91: When an HMM engine produces a degenerate 3-state fit
(one state < 5% of bars), automatically downgrade to n_states=2
before entering the walk-forward loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hmm_futures_analysis.regime.engine_protocol import ClassifyOutput
from hmm_futures_analysis.regime.engines._hmm_pipeline import (
    _check_degenerate,
    _hmm_classify_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate a price series with enough bars for walk-forward."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0002, 0.01, n)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(prices, index=dates, name="close")


def _make_ohlcv(prices: pd.Series) -> pd.DataFrame:
    """Build synthetic OHLCV from a price series."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({"close": prices})
    df["open"] = prices * (1 + rng.uniform(-0.005, 0.005, len(prices)))
    df["high"] = prices * (1 + np.abs(rng.uniform(0, 0.01, len(prices))))
    df["low"] = prices * (1 - np.abs(rng.uniform(0, 0.01, len(prices))))
    df["volume"] = rng.uniform(1e6, 1e7, len(prices))
    return df


# ---------------------------------------------------------------------------
# Tracer bullet 1: Healthy 3-state fit stays at n_states=3
# ---------------------------------------------------------------------------


class TestHealthyFitNoDowngrade:
    """When a 3-state HMM produces balanced regimes, no auto-downgrade."""

    def test_n_states_unchanged_on_balanced_fit(self):
        """n_states remains 3 when all regimes have > 5% of bars."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        result = _hmm_classify_pipeline(
            engine, prices, ohlcv, returns, min_train=50, profile=False
        )

        assert isinstance(result, ClassifyOutput)
        assert result.n_states == 3, "Healthy 3-state fit should not be auto-downgraded"


# ---------------------------------------------------------------------------
# Tracer bullet 2: Degenerate fit triggers auto-downgrade to n_states=2
# ---------------------------------------------------------------------------


class TestDegenerateAutoDowngrade:
    """When a 3-state fit degenerates, auto-downgrade to n_states=2."""

    def test_check_degenerate_detects_collapsed_state(self):
        """_check_degenerate returns True when one state has < 5% of bars."""
        # 0 bear, 80 sideways, 20 bull → bear has 0% < 5%
        regimes = np.array([1] * 80 + [2] * 20)
        assert _check_degenerate(regimes, n_states=3) is True

    def test_check_degenerate_healthy_returns_false(self):
        """_check_degenerate returns False when all states have >= 5%."""
        # 30 bear, 40 sideways, 30 bull → all > 5%
        regimes = np.array([0] * 30 + [1] * 40 + [2] * 30)
        assert _check_degenerate(regimes, n_states=3) is False

    def test_check_degenerate_two_state_not_checked(self):
        """_check_degenerate returns False for n_states=2 (nothing to collapse)."""
        regimes = np.array([0] * 50 + [1] * 50)
        assert _check_degenerate(regimes, n_states=2) is False

    def test_pipeline_downgrades_on_degenerate_fit(self):
        """_hmm_classify_pipeline auto-downgrades to n_states=2 when degenerate."""
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        # Patch _check_degenerate to simulate degenerate detection
        # without relying on specific HMM convergence behavior
        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
            return_value=True,
        ):
            result = _hmm_classify_pipeline(
                engine, prices, ohlcv, returns, min_train=50, profile=False
            )

        assert result.n_states == 2, (
            "Degenerate 3-state fit should auto-downgrade to n_states=2"
        )


# ---------------------------------------------------------------------------
# Tracer bullet 3: Auto-recovery audit trail in engine_info
# ---------------------------------------------------------------------------


class TestAutoRecoveryAuditTrail:
    """engine_info records degenerate auto-recovery metadata."""

    def test_engine_info_has_auto_recovered_flag(self):
        """engine_info.degenerate_auto_recovered is True after downgrade."""
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
            return_value=True,
        ):
            result = _hmm_classify_pipeline(
                engine, prices, ohlcv, returns, min_train=50, profile=False
            )

        assert result.engine_info is not None
        assert result.engine_info.get("degenerate_auto_recovered") is True

    def test_engine_info_has_original_n_states(self):
        """engine_info.original_n_states records the user's requested n_states."""
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
            return_value=True,
        ):
            result = _hmm_classify_pipeline(
                engine, prices, ohlcv, returns, min_train=50, profile=False
            )

        assert result.engine_info["original_n_states"] == 3

    def test_healthy_fit_has_no_engine_info(self):
        """engine_info is None when no auto-recovery occurred."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        result = _hmm_classify_pipeline(
            engine, prices, ohlcv, returns, min_train=50, profile=False
        )

        # Healthy fit should not trigger auto-recovery
        if result.engine_info is not None:
            assert not result.engine_info.get("degenerate_auto_recovered", False)


# ---------------------------------------------------------------------------
# Tracer bullet 4: Stderr warning on auto-downgrade
# ---------------------------------------------------------------------------


class TestStderrWarning:
    """Stderr warning is emitted when auto-downgrade fires."""

    def test_stderr_contains_degenerate_warning(self):
        """Stderr contains '[degenerate]' when auto-downgrade fires."""
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        with (
            patch(
                "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
                return_value=True,
            ),
        ):
            import io
            import sys

            captured = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = captured
            try:
                _hmm_classify_pipeline(
                    engine, prices, ohlcv, returns, min_train=50, profile=False
                )
            finally:
                sys.stderr = old_stderr

        output = captured.getvalue()
        assert "[degenerate]" in output, (
            f"Expected '[degenerate]' in stderr, got: {output!r}"
        )

    def test_no_stderr_warning_on_healthy_fit(self):
        """No '[degenerate]' stderr on healthy 3-state fit."""
        import io
        import sys
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=3)

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch(
                "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
                return_value=False,
            ):
                _hmm_classify_pipeline(
                    engine, prices, ohlcv, returns, min_train=50, profile=False
                )
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "[degenerate]" not in output


# ---------------------------------------------------------------------------
# Tracer bullet 5: No pre-check when n_states=2 already
# ---------------------------------------------------------------------------


class TestNoPrecheckForTwoStates:
    """Pre-check is skipped when engine already has n_states=2."""

    def test_n_states_2_stays_2(self):
        """n_states=2 engine completes without degenerate pre-check logic."""
        from unittest.mock import patch

        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        engine = HMMGenericEngine(n_states=2)

        # _check_degenerate should never be called for n_states=2
        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._check_degenerate",
            side_effect=AssertionError("_check_degenerate should not be called"),
        ):
            result = _hmm_classify_pipeline(
                engine, prices, ohlcv, returns, min_train=50, profile=False
            )

        assert result.n_states == 2
        assert result.engine_info is None or not result.engine_info.get(
            "degenerate_auto_recovered", False
        )
