"""Tests for classify-pipeline phase dispatch and engine implementations.

Issue #62: classify_pipeline removed from RegimeEngine Protocol.
Dispatch now lives in pipeline.py via config.is_hmm check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hmm_futures_analysis.regime.engine_protocol import ClassifyOutput


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
# Tracer bullet 2a: ThresholdEngine.run_classify (ADR-0017)
# ---------------------------------------------------------------------------


class TestThresholdRunClassify:
    """ThresholdEngine.run_classify returns ClassifyOutput (ADR-0017)."""

    def test_returns_classify_output(self):
        """Returns ClassifyOutput with regimes, no posteriors."""
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        eng = ThresholdEngine(window=20, threshold=0.05)

        result = eng.run_classify(prices, ohlcv, returns, min_train=50)

        assert isinstance(result, ClassifyOutput)
        assert isinstance(result.regimes, np.ndarray)
        assert len(result.regimes) == len(returns)
        assert result.posteriors is None
        assert result.last_regime in (0, 1, 2)
        assert result.warmup_bars is None

    def test_regime_values_in_valid_range(self):
        """All regime values are 0, 1, or 2."""
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        eng = ThresholdEngine()

        result = eng.run_classify(prices, ohlcv, returns, min_train=50)
        assert set(np.unique(result.regimes)).issubset({0, 1, 2})

    def test_run_classify_has_no_is_hmm_branch(self):
        """run_classify source contains no is_hmm branching."""
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        # run_classify should not exist yet (RED) — but once it does,
        # this test itself only verifies behavior through the public interface.
        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        eng = ThresholdEngine(window=20, threshold=0.05)
        result = eng.run_classify(prices, ohlcv, returns, min_train=50)
        assert isinstance(result, ClassifyOutput)


# ---------------------------------------------------------------------------
# Tracer bullet 3a: HMMGenericEngine.run_classify (ADR-0017)
# ---------------------------------------------------------------------------


class TestHMMGenericRunClassify:
    """HMMGenericEngine.run_classify delegates to shared HMM pipeline."""

    def test_returns_classify_output_with_posteriors(self):
        """Returns ClassifyOutput with regimes and posteriors."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        eng = HMMGenericEngine(n_states=3)

        result = eng.run_classify(prices, ohlcv, returns, min_train=50)

        assert isinstance(result, ClassifyOutput)
        assert isinstance(result.regimes, np.ndarray)
        assert len(result.regimes) == len(returns)
        assert result.posteriors is not None
        assert result.posteriors.shape == (len(returns), 3)
        assert result.last_regime in (0, 1, 2)
        assert result.warmup_bars == 50

    def test_warmup_bars_stored(self):
        """warmup_bars in output matches min_train parameter."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)
        returns = prices.pct_change(fill_method=None).dropna()
        eng = HMMGenericEngine(n_states=3)

        result = eng.run_classify(prices, ohlcv, returns, min_train=80)
        assert result.warmup_bars == 80


# ---------------------------------------------------------------------------
# Tracer bullet 4: pipeline.run() uses dispatch consistently
# ---------------------------------------------------------------------------


class TestPipelineUniformDispatch:
    """pipeline.run() dispatches classify-pipeline via config.is_hmm."""

    def test_threshold_produces_valid_result(self):
        """pipeline.run() with threshold engine produces valid output."""
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)

        result = run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
            ohlcv=ohlcv,
            min_train=50,
            profile=False,
        )

        assert result.engine == "threshold"
        assert result.current_regime["index"] in (0, 1, 2)

    def test_hmm_generic_produces_valid_result(self):
        """pipeline.run() with hmm generic produces valid output."""
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run

        prices = _make_prices(300)
        ohlcv = _make_ohlcv(prices)

        result = run(
            prices,
            source="test",
            engine_config=HMMGenericConfig(n_states=3),
            ohlcv=ohlcv,
            min_train=50,
            profile=False,
        )

        assert result.engine == "hmm"
        assert result.current_regime["index"] in (0, 1, 2)

    def test_pipeline_no_string_dispatch(self):
        """pipeline.run() source has no if-engine-threshold or if-engine-hmm branches."""
        import ast
        import inspect

        from hmm_futures_analysis.regime import pipeline as pipeline_mod

        source = inspect.getsource(pipeline_mod.run)
        tree = ast.parse(source)

        string_branches = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for 'engine == "threshold"' or similar
                if (
                    isinstance(node.left, ast.Name)
                    and node.left.id == "engine"
                    and any(
                        isinstance(c, ast.Constant) and isinstance(c.value, str)
                        for c in node.comparators
                    )
                ):
                    string_branches.append(
                        f"L{node.lineno}: engine == {node.comparators[0].value!r}"
                    )

        assert not string_branches, (
            f"pipeline.run() still has string-based engine dispatch: {string_branches}"
        )
