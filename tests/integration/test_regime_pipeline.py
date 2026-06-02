"""Integration tests for the regime detection pipeline."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from hmm_futures_analysis.regime.markov_chain import (
    build_transition_matrix,
    classify_regimes,
    compute_persistence_diagonal,
    compute_signal,
    compute_stationary_distribution,
    forecast_n_steps,
)
from hmm_futures_analysis.regime.engine_configs import (  # noqa: F401
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)
from hmm_futures_analysis.regime.pipeline import (
    _nan_to_none,
    _probs_to_dict,
    run as pipeline_run,
)
from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest
from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv  # noqa: F401  (used by config tests below)


@pytest.mark.slow
class TestThresholdPipeline:
    """Test threshold-based regime classification pipeline."""

    def test_classify_returns_three_states(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        assert len(regimes) == len(returns)
        assert set(np.unique(regimes)).issubset({0, 1, 2})

    def test_transition_matrix_row_sums(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        assert transmat.shape == (3, 3)
        row_sums = transmat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_stationary_distribution_sums_to_one(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        stationary = compute_stationary_distribution(transmat)
        assert len(stationary) == 3
        np.testing.assert_allclose(sum(stationary), 1.0, atol=1e-8)

    def test_signal_in_range(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        current_regime = int(regimes[-1])
        signal = compute_signal(transmat[current_regime])
        assert -1.0 <= signal <= 1.0

    def test_walk_forward_returns_keys(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        result = walk_forward_backtest(prices, regimes=regimes)
        expected = {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_walk_forward_drawdown_negative_or_nan(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        result = walk_forward_backtest(prices, regimes=regimes)
        assert result.max_drawdown <= 0 or np.isnan(result.max_drawdown)

    def test_forecast_n_steps(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        current_regime = int(regimes[-1])
        probs = transmat[current_regime]
        forecast = forecast_n_steps(transmat, probs, 5)
        assert len(forecast) == 3
        np.testing.assert_allclose(sum(forecast), 1.0, atol=1e-8)

    def test_small_dataset_handles_gracefully(self, futures_csv):
        prices = load_from_csv(futures_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        # Should still return a valid array even for small data
        assert len(regimes) == len(returns)

    def test_persistence_diagonal_keys(self, btc_csv):
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change().dropna()
        regimes = classify_regimes(returns)
        transmat = build_transition_matrix(regimes)
        persistence = compute_persistence_diagonal(transmat)
        assert "bear" in persistence
        assert "sideways" in persistence
        assert "bull" in persistence


@pytest.mark.slow
class TestPipelineRunInputValidation:
    """Input validation tests for pipeline.run()."""

    @staticmethod
    def _make_series(n: int) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.Series([float(i) for i in range(1, n + 1)], index=dates, dtype=float)

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            pipeline_run(
                self._make_series(0), source="test", engine_config=ThresholdConfig()
            )

    def test_rejects_single_row(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            pipeline_run(
                self._make_series(1), source="test", engine_config=ThresholdConfig()
            )

    def test_rejects_zero_price_series(self):
        """[0.0, 0.0] produces only NaN returns after pct_change — empty after dropna."""
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        with pytest.raises(ValueError, match="at least 2 valid returns"):
            pipeline_run(
                pd.Series([0.0, 0.0], index=idx),
                source="test",
                engine_config=ThresholdConfig(),
            )

    def test_rejects_non_numeric_dtype(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        s = pd.Series(["10.0", "20.0", "30.0"], index=idx)
        with pytest.raises(ValueError, match="numeric"):
            pipeline_run(s, source="test", engine_config=ThresholdConfig())

    def test_rejects_non_datetime_index(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            pipeline_run(s, source="test", engine_config=ThresholdConfig())

    def test_rejects_dataframe(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match=r"pd\.Series"):
            pipeline_run(df, source="test", engine_config=ThresholdConfig())

    def test_run_returns_valid_structure(self, btc_csv):
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        # Top-level keys
        expected_keys = {
            "source",
            "engine",
            "dates",
            "current_regime",
            "next_state_probabilities",
            "signal",
            "transition_matrix",
            "persistence_diagonal",
            "stationary_distribution",
            "regime_counts",
            "walk_forward",
            "forecast",
            "engine_info",
            "framework",
            "disclaimer",
        }
        for key in expected_keys:
            assert key in result, f"Missing required key: {key}"
        # Engine
        assert result.engine == "threshold"
        # Dates
        assert "start" in result.dates
        assert "end" in result.dates
        # Current regime
        assert result.current_regime["name"] in ("bear", "sideways", "bull")
        assert isinstance(result.current_regime["index"], int)
        # Signal range
        assert -1.0 <= result.signal <= 1.0
        # Transition matrix
        tm = result.transition_matrix
        assert len(tm) == 3
        assert all(len(row) == 3 for row in tm)
        for row in tm:
            assert abs(sum(row) - 1.0) < 0.01
        # Persistence diagonal
        for name in ("bear", "sideways", "bull"):
            assert name in result.persistence_diagonal
            assert 0.0 <= result.persistence_diagonal[name] <= 1.0
        # Stationary distribution
        sd = result.stationary_distribution
        total = sd["bear"] + sd["sideways"] + sd["bull"]
        assert abs(total - 1.0) < 0.01
        # Walk-forward has 6 keys
        wf = result.walk_forward
        assert set(wf.keys()) == {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }
        assert isinstance(wf.n_trades, int)
        # Forecast
        for step in ("1_step", "5_step", "20_step"):
            assert step in result.forecast
            f = result.forecast[step]
            assert set(f.keys()) == {"bear", "sideways", "bull"}
            assert abs(sum(f.values()) - 1.0) < 0.01
        # Engine info
        ei = result.engine_info
        assert ei["method"] == "threshold"
        # Regime counts
        rc = result.regime_counts
        assert set(rc.keys()) == {"bear", "sideways", "bull"}
        assert all(isinstance(v, int) for v in rc.values())

    def test_run_with_hmm_engine_requires_ohlcv(self, btc_csv):
        """pipeline.run(engine='hmm') without OHLCV raises ValueError."""
        prices = load_from_csv(btc_csv)
        with pytest.raises(ValueError, match=r"OHLCV"):
            pipeline_run(prices, source="test", engine_config=HMMGenericConfig())

    def test_regime_transitions_present_in_result(self, btc_csv):
        """pipeline.run() → result.regime_transitions is a list."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
        )
        assert hasattr(result, "regime_transitions")
        transitions = result.regime_transitions
        assert isinstance(transitions, list)

    def test_regime_transitions_contents_valid(self, btc_csv):
        """pipeline.run() transitions have correct structure."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(
            prices,
            source="test",
            engine_config=ThresholdConfig(),
        )
        transitions = result.regime_transitions
        if len(transitions) > 0:
            event = transitions[0]
            assert isinstance(event["date"], str)
            assert event["from_regime"] in ("bear", "sideways", "bull")
            assert event["to_regime"] in ("bear", "sideways", "bull")
            assert event["from_regime"] != event["to_regime"]
            assert isinstance(event["bar_index"], int)


@pytest.mark.slow
class TestWalkForwardBacktest:
    """Tests for walk_forward_backtest with pre-computed regimes (ADR-0017)."""

    def test_regimes_arg_returns_rich_keys(self, btc_csv):
        """walk_forward_backtest returns 6 keys with regimes array."""
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        result = walk_forward_backtest(prices, regimes=regimes)
        expected = {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }
        for key in expected:
            assert key in result, f"Missing key: {key}"

    def test_win_rate_in_range(self, btc_csv):
        """Win rate should be in [0, 1] or NaN."""
        prices = load_from_csv(btc_csv)
        returns = prices.pct_change(fill_method=None).dropna()
        regimes = classify_regimes(returns, window=20, threshold=0.05)
        result = walk_forward_backtest(prices, regimes=regimes)
        wr = result.win_rate
        assert 0.0 <= wr <= 1.0 or np.isnan(wr)

    def test_missing_regimes_raises_type_error(self, btc_csv):
        """Missing required regimes kwarg raises TypeError."""
        prices = load_from_csv(btc_csv)
        with pytest.raises(TypeError):
            walk_forward_backtest(prices)  # noqa: missing regimes

    def test_insufficient_data_returns_nan(self, btc_csv):
        """Too few bars returns NaN-filled result."""
        prices = load_from_csv(btc_csv).iloc[:5]
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) == 0:
            returns = pd.Series([0.0], index=prices.index[:1])
        regimes = np.ones(len(returns), dtype=int)
        result = walk_forward_backtest(prices, regimes=regimes, min_train=252)
        assert np.isnan(result.sharpe)
        assert np.isnan(result.max_drawdown)
        assert result.n_trades == 0


@pytest.mark.slow
class TestHmmWalkForward:
    """Tests for HMM-based walk-forward backtest engines."""

    @pytest.fixture(scope="class")
    def ohlcv_small(self):
        """Small synthetic OHLCV dataset for fast HMM fitting."""
        np.random.seed(42)
        n = 400  # enough for min_train=252
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)  # ensure positive
        return pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )

    def test_hmm_engine_requires_ohlcv(self, btc_csv):
        """HMM engine with no OHLCV data raises during run_classify."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = load_from_csv(btc_csv)
        engine = HMMGenericEngine(n_states=3)
        returns = prices.pct_change(fill_method=None).dropna()
        # ADR-0017: engine.run_classify fails when OHLCV is None
        with pytest.raises((ValueError, TypeError)):
            engine.run_classify(prices, None, returns, min_train=50)

    def test_messina_engine_requires_ohlcv(self, btc_csv):
        """Messina engine with no OHLCV data raises during run_classify."""
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        prices = load_from_csv(btc_csv)
        engine = HMMMMessinaEngine(n_states=3)
        returns = prices.pct_change(fill_method=None).dropna()
        with pytest.raises((ValueError, TypeError)):
            engine.run_classify(prices, None, returns, min_train=50)

    def test_hmm_engine_returns_rich_keys(self, ohlcv_small):
        """HMM engine run_classify() → regimes → walk_forward_backtest."""
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        prices = ohlcv_small["close"]
        engine = HMMGenericEngine(n_states=3)
        returns = prices.pct_change(fill_method=None).dropna()
        min_train = 300

        # ADR-0017: engine owns the classify loop
        classify_out = engine.run_classify(
            prices, ohlcv_small, returns, min_train=min_train
        )
        wf_result = walk_forward_backtest(
            prices,
            regimes=classify_out.regimes,
            posteriors=classify_out.posteriors,
            min_train=min_train,
        )
        expected = {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }
        for key in expected:
            assert key in wf_result, f"Missing key: {key}"

    def test_messina_engine_returns_rich_keys(self, ohlcv_small):
        """Messina engine run_classify() → regimes → walk_forward_backtest."""
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        prices = ohlcv_small["close"]
        engine = HMMMMessinaEngine(n_states=3)
        returns = prices.pct_change(fill_method=None).dropna()
        min_train = 300

        # ADR-0017: engine owns the classify loop
        classify_out = engine.run_classify(
            prices, ohlcv_small, returns, min_train=min_train
        )
        wf_result = walk_forward_backtest(
            prices,
            regimes=classify_out.regimes,
            posteriors=classify_out.posteriors,
            min_train=min_train,
        )
        expected = {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }
        for key in expected:
            assert key in wf_result, f"Missing key: {key}"


@pytest.mark.slow
class TestPipelineRunEngine:
    """Tests for pipeline.run() with engine parameter."""

    def test_run_with_engine_threshold(self, btc_csv):
        """pipeline.run(engine_config=ThresholdConfig()) produces valid output."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        assert result.engine == "threshold"
        assert result.engine_info["method"] == "threshold"
        # Walk-forward has 6 keys
        wf = result.walk_forward
        assert set(wf.keys()) == {
            "sharpe",
            "max_drawdown",
            "n_trades",
            "win_rate",
            "profit_factor",
            "total_return",
        }

    def test_run_rejects_invalid_engine(self, btc_csv):
        """Invalid engine name in config raises ValueError."""
        prices = load_from_csv(btc_csv)

        @dataclasses.dataclass
        class BadConfig:
            name: str = "invalid"
            features: str = "returns"

        with pytest.raises(ValueError, match=r"engine"):
            pipeline_run(prices, source="test", engine_config=BadConfig())

    def test_run_engine_info_has_expected_keys(self, btc_csv):
        """engine_info block has required fields."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        ei = result.engine_info
        assert ei["method"] == "threshold"
        assert "features" in ei
        assert "n_states" in ei

    def test_run_output_no_hmm_key(self, btc_csv):
        """New contract: no top-level hmm key."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        assert "hmm" not in result
        assert "hmm_test_extras" not in result

    def test_run_walk_forward_sharpe_not_none(self, btc_csv):
        """With sufficient data, walk_forward produces numeric values."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        wf = result.walk_forward
        assert isinstance(wf.sharpe, float)
        assert not np.isnan(wf.sharpe)
        assert isinstance(wf.max_drawdown, float)
        assert wf.max_drawdown <= 0 or np.isnan(wf.max_drawdown)
        assert isinstance(wf.n_trades, int)
        assert wf.n_trades > 0


@pytest.mark.slow
class TestPipelineHelpers:
    """Unit tests for pipeline private helpers."""

    def test_nan_to_none_replaces_nan(self):
        assert _nan_to_none(float("nan")) is None
        assert _nan_to_none(np.nan) is None

    def test_nan_to_none_preserves_values(self):
        assert _nan_to_none(0.0) == 0.0
        assert _nan_to_none(-1.5) == -1.5
        assert _nan_to_none(1.5) == 1.5
        # isinstance(None, float) → False → returned as-is
        assert _nan_to_none(None) is None

    def test_nan_to_none_handles_int(self):
        # int values are not float → returned as-is
        assert _nan_to_none(42) == 42
        assert _nan_to_none(-1) == -1

    def test_probs_to_dict_maps_correctly(self):
        result = _probs_to_dict(np.array([0.1, 0.3, 0.6]))
        assert result == {"bear": 0.1, "sideways": 0.3, "bull": 0.6}
        assert all(isinstance(v, float) for v in result.values())

    def test_probs_to_dict_with_zeros(self):
        result = _probs_to_dict(np.array([0.0, 1.0, 0.0]))
        assert result["bear"] == 0.0
        assert result["sideways"] == 1.0
        assert result["bull"] == 0.0

    def test_nan_to_none_on_transition_matrix_nan(self):
        # Transition matrix may produce NaN in degenerate cases — verify
        assert _nan_to_none(math.nan) is None

    def test_nan_to_none_handles_infinity(self):
        """Infinity should serialize as None (JSON-safe)."""
        assert _nan_to_none(float("inf")) is None
        assert _nan_to_none(float("-inf")) is None

    def test_nan_to_none_preserves_valid_floats(self):
        """Normal finite floats pass through unchanged."""
        assert _nan_to_none(3.14) == 3.14
        assert _nan_to_none(-2.718) == -2.718
        assert _nan_to_none(0.0) == 0.0


@pytest.mark.slow
class TestPipelineEngineTopLevelStats:
    """Top-level stats must reflect the chosen engine, not always threshold."""

    @pytest.fixture(scope="class")
    def ohlcv_pipeline(self):
        """Synthetic OHLCV for pipeline top-level stats testing."""
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)
        return pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )

    def test_hmm_transition_matrix_differs_from_threshold(self, ohlcv_pipeline):
        """HMM engine must produce a different transition matrix than threshold."""
        prices = ohlcv_pipeline["close"]
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(
            prices, engine_config=ThresholdConfig(), **common
        )
        result_hmm = pipeline_run(
            prices, engine_config=HMMGenericConfig(), ohlcv=ohlcv_pipeline, **common
        )
        assert result_hmm.transition_matrix != result_threshold.transition_matrix

    def test_messina_transition_matrix_differs_from_threshold(self, ohlcv_pipeline):
        """Messina engine must produce a different transition matrix than threshold."""
        prices = ohlcv_pipeline["close"]
        common = dict(source="test", min_train=300)
        result_threshold = pipeline_run(
            prices, engine_config=ThresholdConfig(), **common
        )
        result_messina = pipeline_run(
            prices, engine_config=HMMMMessinaConfig(), ohlcv=ohlcv_pipeline, **common
        )
        assert result_messina.transition_matrix != result_threshold.transition_matrix

    def test_hmm_engine_info_contains_warmup_bars(self, ohlcv_pipeline):
        """HMM engine_info documents the warmup period."""
        prices = ohlcv_pipeline["close"]
        result = pipeline_run(
            prices,
            engine_config=HMMGenericConfig(),
            ohlcv=ohlcv_pipeline,
            source="test",
            min_train=300,
        )
        assert "warmup_bars" in result.engine_info
        assert result.engine_info["warmup_bars"] == 300

    def test_messina_engine_info_contains_warmup_bars(self, ohlcv_pipeline):
        """Messina engine_info documents the warmup period."""
        prices = ohlcv_pipeline["close"]
        result = pipeline_run(
            prices,
            engine_config=HMMMMessinaConfig(),
            ohlcv=ohlcv_pipeline,
            source="test",
            min_train=300,
        )
        assert "warmup_bars" in result.engine_info
        assert result.engine_info["warmup_bars"] == 300

    def test_threshold_engine_info_has_no_warmup_bars(self, ohlcv_pipeline):
        """Threshold engine does not report warmup_bars."""
        prices = ohlcv_pipeline["close"]
        result = pipeline_run(
            prices, engine_config=ThresholdConfig(), source="test", min_train=300
        )
        assert "warmup_bars" not in result.engine_info

    def test_hmm_pipeline_requires_ohlcv(self, ohlcv_pipeline):
        """engine=hmm without OHLCV raises ValueError at top-level."""
        prices = ohlcv_pipeline["close"]
        with pytest.raises(ValueError, match=r"OHLCV"):
            pipeline_run(prices, engine_config=HMMGenericConfig(), source="test")

    def test_messina_pipeline_requires_ohlcv(self, ohlcv_pipeline):
        """engine=messina without OHLCV raises ValueError at top-level."""
        prices = ohlcv_pipeline["close"]
        with pytest.raises(ValueError, match=r"OHLCV"):
            pipeline_run(prices, engine_config=HMMMMessinaConfig(), source="test")


@pytest.mark.slow
class TestPipelineRunEngineConfig:
    """Tests for pipeline.run() accepting engine_config parameter."""

    def test_threshold_config_accepted(self, btc_csv):
        """pipeline.run(engine_config=ThresholdConfig()) produces valid output."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        assert result.engine == "threshold"
        assert result.engine_info["method"] == "threshold"
        assert result.engine_info["features"] == "returns"
        assert result.walk_forward is not None
        wf = result.walk_forward
        assert isinstance(wf.sharpe, float)
        assert wf.n_trades > 0


class TestPipelineEnrichInfo:
    """Pipeline uses engine.enrich_info() for engine_info extras."""

    def test_hmm_engine_info_has_caveat(self, ohlcv_pipeline):
        """HMM engine_info gets caveat from enrich_info, not config."""
        prices = ohlcv_pipeline["close"]
        result = pipeline_run(
            prices,
            engine_config=HMMGenericConfig(),
            ohlcv=ohlcv_pipeline,
            source="test",
            min_train=300,
        )
        assert "caveat" in result.engine_info
        assert "HMM states sorted by mean return" in result.engine_info["caveat"]

    def test_robust_hmm_engine_info_has_method(self, ohlcv_pipeline):
        """RobustHMM engine_info gets robust_method from enrich_info."""
        prices = ohlcv_pipeline["close"]
        result = pipeline_run(
            prices,
            engine_config=RobustHMMConfig(robust_method="huber"),
            ohlcv=ohlcv_pipeline,
            source="test",
            min_train=300,
        )
        assert result.engine_info["robust_method"] == "huber"
        assert "caveat" in result.engine_info

    def test_threshold_engine_info_has_no_caveat(self, btc_csv):
        """Threshold engine_info has no caveat (no enrich_info method)."""
        prices = load_from_csv(btc_csv)
        result = pipeline_run(prices, source="test", engine_config=ThresholdConfig())
        assert "caveat" not in result.engine_info


class TestPipelineOHLCVFromEngine:
    """Pipeline OHLCV validation comes from engines, not pipeline guard."""

    def test_hmm_without_ohlcv_raises_engine_error(self, btc_csv):
        """HMM engine without OHLCV raises ValueError from the engine."""
        prices = load_from_csv(btc_csv)
        with pytest.raises(ValueError, match="HMMGenericEngine"):
            pipeline_run(prices, source="test", engine_config=HMMGenericConfig())

    def test_messina_without_ohlcv_raises_engine_error(self, btc_csv):
        """Messina engine without OHLCV raises ValueError from the engine."""
        prices = load_from_csv(btc_csv)
        with pytest.raises(ValueError, match="HMMMMessinaEngine"):
            pipeline_run(prices, source="test", engine_config=HMMMMessinaConfig())
