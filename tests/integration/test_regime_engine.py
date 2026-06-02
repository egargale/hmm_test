"""Tests for RegimeEngine protocol, engine registry, and config dataclasses."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_regime(*args):
    """Run hmm_futures_analysis/cli.py with args, return CompletedProcess."""
    cmd = [sys.executable, "-m", "hmm_futures_analysis.cli"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))


@pytest.mark.slow
class TestProtocolConformance:
    """Each engine class satisfies the RegimeEngine protocol."""

    def test_threshold_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        assert isinstance(engine, RegimeEngine)

    def test_hmm_generic_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        assert isinstance(engine, RegimeEngine)

    def test_hmm_messina_engine_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        assert isinstance(engine, RegimeEngine)


@pytest.mark.slow
class TestThresholdEngine:
    """ThresholdEngine classify returns expected regime indices."""

    @pytest.fixture(scope="class")
    def returns_series(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.Series(np.random.normal(0.001, 0.02, n), index=dates)

    def test_classify_returns_regime_in_valid_range(self, returns_series):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns_series)
        assert result.regime in {0, 1, 2}
        assert result.means is None

    def test_precompute_returns_none(self, returns_series):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        engine = ThresholdEngine(window=20, threshold=0.05)
        assert engine.precompute(returns_series) is None

    def test_classify_bull_when_strong_positive_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.01, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 2

    def test_classify_bear_when_strong_negative_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(-0.01, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 0

    def test_classify_sideways_when_flat_returns(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        returns = pd.Series(0.001, index=dates)
        engine = ThresholdEngine(window=20, threshold=0.05)
        result = engine.classify(returns)
        assert result.regime == 1


@pytest.mark.slow
class TestHMMGenericEngine:
    """HMMGenericEngine precompute and classify."""

    @pytest.fixture(scope="class")
    def ohlcv_data(self):
        np.random.seed(42)
        n = 300
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

    def test_precompute_returns_dataframe(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert features.shape[1] > 1

    def test_classify_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means is not None
        assert result.means.shape[0] == 3

    def test_classify_with_prev_means_preserves_labels(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        first = engine.classify(features)
        assert first.regime in {0, 1, 2}
        assert first.means is not None
        second = engine.classify(features, prev_means=first.means)
        assert second.regime in {0, 1, 2}
        assert second.means is not None

    def test_classify_with_2_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=2)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1}
        assert result.means is not None
        assert result.means.shape[0] == 2

    def test_classify_with_4_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        engine = HMMGenericEngine(n_states=4)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2, 3}
        assert result.means is not None
        assert result.means.shape[0] == 4


@pytest.mark.slow
class TestHMMMMessinaEngine:
    """HMMMMessinaEngine precompute and classify."""

    @pytest.fixture(scope="class")
    def ohlcv_data(self):
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

    def test_precompute_returns_dataframe(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0

    def test_classify_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1, 2}
        assert result.means is not None
        assert result.means.shape[0] == 3

    def test_classify_with_prev_means(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=3)
        features = engine.precompute(ohlcv_data)
        first = engine.classify(features)
        second = engine.classify(features, prev_means=first.means)
        assert second.regime in {0, 1, 2}

    def test_classify_with_2_states_returns_valid_regime(self, ohlcv_data):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        engine = HMMMMessinaEngine(n_states=2)
        features = engine.precompute(ohlcv_data)
        result = engine.classify(features)
        assert result.regime in {0, 1}
        assert result.means is not None
        assert result.means.shape[0] == 2


@pytest.mark.slow
class TestSelectNStates:
    """BIC-based state count selection (Issue #17)."""

    @pytest.fixture(scope="class")
    def synthetic_3state_features(self):
        """3-state synthetic feature data (each state draws from a different mean)."""
        np.random.seed(42)
        n_per = 100
        # Three Gaussians with well-separated means in 2D
        block_a = np.random.normal(loc=-5.0, scale=0.5, size=(n_per, 2))
        block_b = np.random.normal(loc=0.0, scale=0.5, size=(n_per, 2))
        block_c = np.random.normal(loc=5.0, scale=0.5, size=(n_per, 2))
        return np.vstack([block_a, block_b, block_c])

    def test_select_n_states_returns_int_in_range(self, synthetic_3state_features):
        from hmm_futures_analysis.regime.engines._hmm_pipeline import select_n_states

        result = select_n_states(synthetic_3state_features, max_states=5)
        assert isinstance(result, int)
        assert 2 <= result <= 5

    def test_select_n_states_picks_3_for_3state_data(self, synthetic_3state_features):
        """BIC should favor 3 states for data generated from 3 Gaussians."""
        from hmm_futures_analysis.regime.engines._hmm_pipeline import select_n_states

        result = select_n_states(synthetic_3state_features, max_states=6)
        assert result == 3

    def test_select_n_states_short_data_caps_max_states(self):
        """When data is short, max_states should be capped to avoid overfitting."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.engines._hmm_pipeline import select_n_states

        # Only 15 rows — should never attempt to fit more than 1 state
        # (15 // 10 = 1), which means max_states clamped to min(max_states, 1) = 1,
        # then floored to 2. So only k=2 should be tried.
        np.random.seed(42)
        tiny = np.random.randn(15, 2)

        original_fit = __import__(
            "hmm_futures_analysis.regime.engines._hmm_engine",
            fromlist=["_fit_hmm_on_slice"],
        )._fit_hmm_on_slice
        fit_calls = []

        def tracking_fit(features, n_states=3, random_state=42):
            fit_calls.append(n_states)
            return original_fit(features, n_states=n_states, random_state=random_state)

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_engine._fit_hmm_on_slice",
            side_effect=tracking_fit,
        ) as mock_fit:
            mock_fit.side_effect = lambda *a, **kw: (
                fit_calls.append(kw.get("n_states", 3)),
                original_fit(*a, **kw),
            )[1]
            result = select_n_states(tiny, max_states=6)

        # Should only try k=2, never k=3,4,5,6
        assert set(fit_calls) == {2}
        assert result == 2


@pytest.mark.slow
class TestEngineRegistry:
    """ENGINE_REGISTRY maps strings to correct classes."""

    def test_registry_has_five_engines(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

        assert set(ENGINE_REGISTRY.keys()) == {
            "threshold",
            "hmm",
            "messina",
            "robust_hmm",
            "fshmm",
        }

    def test_registry_threshold_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        assert ENGINE_REGISTRY["threshold"][0] is ThresholdEngine

    def test_registry_hmm_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        assert ENGINE_REGISTRY["hmm"][0] is HMMGenericEngine

    def test_registry_messina_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        assert ENGINE_REGISTRY["messina"][0] is HMMMMessinaEngine

    def test_registry_fshmm_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        assert ENGINE_REGISTRY["fshmm"][0] is FSHMMEngine

    def test_fshmm_satisfies_protocol(self):
        from hmm_futures_analysis.regime.engine_protocol import RegimeEngine
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        engine = FSHMMEngine(n_states=3)
        assert isinstance(engine, RegimeEngine)

    def test_invalid_engine_raises_key_error(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY

        with pytest.raises(KeyError):
            ENGINE_REGISTRY["invalid"]


@pytest.mark.slow
class TestWalkForwardWithProtocol:
    """Walk-forward backtest works with protocol-based engine dispatch."""

    @pytest.fixture(scope="class")
    def prices(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        return pd.Series(close, index=dates, name="close")

    def test_walk_forward_with_mock_engine(self, prices):
        from hmm_futures_analysis.regime.engine_protocol import ClassifyResult
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        # ADR-0017: walk_forward_backtest takes pre-computed regimes, not engine.
        # Generate regimes to pass.
        returns = prices.pct_change(fill_method=None).dropna()
        n = len(returns)
        regimes = np.ones(n, dtype=int)  # all sideways = 1

        result = walk_forward_backtest(prices, regimes=regimes, min_train=50)
        assert "sharpe" in result
        assert "n_trades" in result
        assert isinstance(result.n_trades, int)

    def test_walk_forward_rejects_string_engine(self, prices):
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        # ADR-0017: engine param removed. Missing regimes raises TypeError.
        with pytest.raises(TypeError):
            walk_forward_backtest(prices)  # noqa: missing required regimes

    def test_walk_forward_regimes_required(self, prices):
        """regimes is a required kwarg (ADR-0017)."""
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        with pytest.raises(TypeError):
            walk_forward_backtest(prices, min_train=50)  # noqa: missing regimes

    def test_walk_forward_threshold_engine_works(self, prices):
        """ThresholdEngine.run_classify() → regimes → walk_forward_backtest."""
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine
        from hmm_futures_analysis.regime.walk_forward import walk_forward_backtest

        engine = ThresholdEngine()
        returns = prices.pct_change(fill_method=None).dropna()
        # ADR-0017: engine owns classify, walk_forward replays regimes
        classify_out = engine.run_classify(prices, None, returns, min_train=50)
        result = walk_forward_backtest(
            prices, regimes=classify_out.regimes, min_train=50
        )
        assert "sharpe" in result
        assert isinstance(result.n_trades, int)


@pytest.mark.slow
class TestCLINStatesArg:
    """CLI --n-states accepts 'auto' and integers >= 2."""

    def test_n_states_auto_accepted(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "threshold", "--n-states", "auto", "--json"
        )
        assert proc.returncode == 0

    def test_n_states_3_backward_compat(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "threshold", "--n-states", "3", "--json"
        )
        assert proc.returncode == 0

    def test_n_states_1_rejected(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "hmm", "--n-states", "1", "--json"
        )
        assert proc.returncode != 0

    def test_n_states_invalid_string_rejected(self, btc_csv):
        proc = run_regime(
            "--csv", btc_csv, "--engine", "hmm", "--n-states", "foo", "--json"
        )
        assert proc.returncode != 0


@pytest.mark.slow
class TestPipelineAutoNStates:
    """Pipeline resolves n_states='auto' via BIC selection."""

    @pytest.fixture(scope="class")
    def prices_and_ohlcv(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100.0 + np.cumsum(np.random.normal(0.02, 1.0, n))
        close = np.maximum(close, 1.0)
        ohlcv = pd.DataFrame(
            {
                "open": close + np.random.normal(0, 0.3, n),
                "high": close + np.abs(np.random.normal(0.8, 0.4, n)),
                "low": close - np.abs(np.random.normal(0.8, 0.4, n)),
                "close": close,
                "volume": np.random.randint(100, 10000, n).astype(float),
            },
            index=dates,
        )
        prices = pd.Series(close, index=dates, name="close")
        return prices, ohlcv

    def test_auto_resolves_to_integer(self, prices_and_ohlcv):
        """pipeline_run with n_states='auto' resolves to an integer."""
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv
        output = pipeline_run(
            prices,
            source="test",
            engine_config=HMMGenericConfig(n_states="auto"),
            ohlcv=ohlcv,
        )
        # Should succeed and return a valid result
        assert "engine_info" in output
        n_used = output.engine_info["n_states"]
        assert isinstance(n_used, int)
        assert n_used >= 2

    def test_auto_uses_bic_selection(self, prices_and_ohlcv):
        """n_states='auto' should use select_n_states under the hood."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv

        with patch(
            "hmm_futures_analysis.regime.pipeline.select_n_states",
            return_value=3,
        ) as mock_bic:
            _ = pipeline_run(
                prices,
                source="test",
                engine_config=HMMGenericConfig(n_states="auto"),
                ohlcv=ohlcv,
            )
            mock_bic.assert_called_once()

    def test_auto_threshold_ignores_bic(self, prices_and_ohlcv):
        """n_states='auto' with threshold engine doesn't call select_n_states."""
        from unittest.mock import patch
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.pipeline import run as pipeline_run

        prices, ohlcv = prices_and_ohlcv

        with patch(
            "hmm_futures_analysis.regime.pipeline.select_n_states",
            return_value=3,
        ) as mock_bic:
            _ = pipeline_run(
                prices,
                source="test",
                engine_config=ThresholdConfig(),
            )
            mock_bic.assert_not_called()


class TestConfigDataclasses:
    """Config dataclasses and registry expansion (ADR-0004)."""

    def test_threshold_config_defaults(self):
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig

        cfg = ThresholdConfig()
        assert cfg.name == "threshold"
        assert cfg.features == "returns"
        assert cfg.window == 20
        assert cfg.threshold == 0.05

    def test_registry_returns_tuple_of_engine_and_config(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig

        entry = ENGINE_REGISTRY["threshold"]
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        assert entry[0] is ThresholdEngine
        assert entry[1] is ThresholdConfig

    def test_resolve_engine_threshold(self):
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        cfg = ThresholdConfig(window=10, threshold=0.1)
        engine = resolve_engine(cfg)
        assert isinstance(engine, ThresholdEngine)
        assert engine.window == 10
        assert engine.threshold == 0.1

    def test_hmm_generic_config_defaults(self):
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig

        cfg = HMMGenericConfig()
        assert cfg.name == "hmm"
        assert cfg.features == "generic"
        assert cfg.n_states == 3
        assert cfg.pca_variance is None

    def test_registry_hmm_config_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig

        assert ENGINE_REGISTRY["hmm"][1] is HMMGenericConfig

    def test_resolve_engine_hmm_generic(self):
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        cfg = HMMGenericConfig(n_states=4, pca_variance=0.95)
        engine = resolve_engine(cfg)
        assert isinstance(engine, HMMGenericEngine)
        assert engine.n_states == 4
        assert engine.pca_variance == 0.95

    def test_hmm_messina_config_defaults(self):
        from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig

        cfg = HMMMMessinaConfig()
        assert cfg.name == "messina"
        assert cfg.features == "messina"
        assert cfg.n_states == 3
        assert cfg.pca_variance is None

    def test_registry_messina_config_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig

        assert ENGINE_REGISTRY["messina"][1] is HMMMMessinaConfig

    def test_resolve_engine_messina(self):
        from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        cfg = HMMMMessinaConfig(n_states=5)
        engine = resolve_engine(cfg)
        assert isinstance(engine, HMMMMessinaEngine)
        assert engine.n_states == 5

    def test_robust_hmm_config_defaults(self):
        from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig

        cfg = RobustHMMConfig()
        assert cfg.name == "robust_hmm"
        assert cfg.features == "generic"
        assert cfg.n_states == 3
        assert cfg.pca_variance is None
        assert cfg.robust_method == "huber"

    def test_registry_robust_config_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig

        assert ENGINE_REGISTRY["robust_hmm"][1] is RobustHMMConfig

    def test_resolve_engine_robust_hmm(self):
        from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        cfg = RobustHMMConfig(n_states=4, robust_method="mcd")
        engine = resolve_engine(cfg)
        assert isinstance(engine, RobustHMMEngine)
        assert engine.n_states == 4
        assert engine.robust_method == "mcd"

    def test_fshmm_config_defaults(self):
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig

        cfg = FSHMMConfig()
        assert cfg.name == "fshmm"
        assert cfg.features == "generic"
        assert cfg.n_states == 3
        assert cfg.pca_variance is None
        assert cfg.saliency_threshold == 0.5

    def test_registry_fshmm_config_class(self):
        from hmm_futures_analysis.regime.engine_protocol import ENGINE_REGISTRY
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig

        assert ENGINE_REGISTRY["fshmm"][1] is FSHMMConfig

    def test_resolve_engine_fshmm(self):
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        cfg = FSHMMConfig(n_states=4, saliency_threshold=0.3)
        engine = resolve_engine(cfg)
        assert isinstance(engine, FSHMMEngine)
        assert engine.n_states == 4
        assert engine.saliency_threshold == 0.3

    def test_resolve_engine_rejects_unknown_name(self):
        import dataclasses
        from hmm_futures_analysis.regime.engine_protocol import resolve_engine

        @dataclasses.dataclass
        class BogusConfig:
            name: str = "nonexistent"
            features: str = "generic"

        with pytest.raises(ValueError, match="Unknown engine name"):
            resolve_engine(BogusConfig())


class TestConfigIsHmm:
    """Config dataclasses report correct is_hmm flag."""

    def test_threshold_is_hmm_false(self):
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig

        cfg = ThresholdConfig()
        assert cfg.is_hmm is False

    def test_hmm_generic_is_hmm_true(self):
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig

        assert HMMGenericConfig().is_hmm is True

    def test_messina_is_hmm_true(self):
        from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig

        assert HMMMMessinaConfig().is_hmm is True

    def test_robust_hmm_is_hmm_true(self):
        from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig

        assert RobustHMMConfig().is_hmm is True

    def test_fshmm_is_hmm_true(self):
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig

        assert FSHMMConfig().is_hmm is True


class TestConfigNoEngineInfoExtras:
    """engine_info_extras() has been removed from config dataclasses."""

    def test_threshold_has_no_engine_info_extras(self):
        from hmm_futures_analysis.regime.engine_configs import ThresholdConfig

        cfg = ThresholdConfig()
        assert not hasattr(cfg, "engine_info_extras")

    def test_hmm_generic_has_no_engine_info_extras(self):
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig

        cfg = HMMGenericConfig()
        assert not hasattr(cfg, "engine_info_extras")

    def test_messina_has_no_engine_info_extras(self):
        from hmm_futures_analysis.regime.engine_configs import HMMMMessinaConfig

        cfg = HMMMMessinaConfig()
        assert not hasattr(cfg, "engine_info_extras")

    def test_robust_hmm_has_no_engine_info_extras(self):
        from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig

        cfg = RobustHMMConfig()
        assert not hasattr(cfg, "engine_info_extras")

    def test_fshmm_has_no_engine_info_extras(self):
        from hmm_futures_analysis.regime.engine_configs import FSHMMConfig

        cfg = FSHMMConfig()
        assert not hasattr(cfg, "engine_info_extras")


class TestPrecomputeValidatesOHLCV:
    """Each HMM engine validates OHLCV input in precompute()."""

    def test_hmm_generic_precompute_none_raises(self):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        eng = HMMGenericEngine(n_states=3)
        with pytest.raises(ValueError, match="HMMGenericEngine"):
            eng.precompute(None)

    def test_hmm_messina_precompute_none_raises(self):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        eng = HMMMMessinaEngine(n_states=3)
        with pytest.raises(ValueError, match="HMMMMessinaEngine"):
            eng.precompute(None)

    def test_robust_hmm_precompute_none_raises(self):
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        eng = RobustHMMEngine(n_states=3)
        with pytest.raises(ValueError, match="RobustHMMEngine"):
            eng.precompute(None)

    def test_fshmm_precompute_none_raises(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        eng = FSHMMEngine(n_states=3)
        with pytest.raises(ValueError, match="FSHMMEngine"):
            eng.precompute(None)

    def test_threshold_precompute_none_returns_none(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        eng = ThresholdEngine()
        assert eng.precompute(None) is None


class TestEnrichInfo:
    """Engines that implement enrich_info() produce correct metadata."""

    def test_hmm_generic_enrich_info(self):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        eng = HMMGenericEngine(n_states=3)
        info = eng.enrich_info({})
        assert "caveat" in info
        assert "HMM states sorted by mean return" in info["caveat"]
        assert "warmup_bars" not in info

    def test_hmm_generic_enrich_info_with_warmup(self):
        from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine

        eng = HMMGenericEngine(n_states=3)
        info = eng.enrich_info({"warmup_bars": 252})
        assert info["warmup_bars"] == 252

    def test_hmm_messina_enrich_info(self):
        from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine

        eng = HMMMMessinaEngine(n_states=3)
        info = eng.enrich_info({})
        assert "caveat" in info
        assert "HMM states sorted by mean return" in info["caveat"]

    def test_robust_hmm_enrich_info_adds_method(self):
        from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine

        eng = RobustHMMEngine(n_states=3, robust_method="huber")
        info = eng.enrich_info({})
        assert info["robust_method"] == "huber"
        assert "caveat" in info

    def test_fshmm_enrich_info_without_saliency(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        eng = FSHMMEngine(n_states=3)
        info = eng.enrich_info({})
        assert "caveat" in info
        assert "feature_saliency" not in info

    def test_fshmm_enrich_info_with_saliency(self):
        from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

        eng = FSHMMEngine(n_states=3)
        eng._last_saliency = [0.1, 0.2, 0.3]
        eng._last_selected_features = ["feat_a", "feat_b"]
        info = eng.enrich_info({})
        assert info["feature_saliency"] == [0.1, 0.2, 0.3]
        assert info["selected_features"] == ["feat_a", "feat_b"]

    def test_threshold_has_no_enrich_info(self):
        from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine

        eng = ThresholdEngine()
        assert not hasattr(eng, "enrich_info")
