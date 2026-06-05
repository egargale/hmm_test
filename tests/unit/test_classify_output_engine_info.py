"""Tests for ClassifyOutput.engine_info field (issue #79).

Verifies that ClassifyOutput accepts an engine_info field and that
each engine populates it correctly during run_classify().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from hmm_futures_analysis.regime.engine_protocol import ClassifyOutput
from hmm_futures_analysis.regime.engines._hmm_engine import HMMEngineBase
from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine
from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine
from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine


class TestClassifyOutputEngineInfo:
    """ClassifyOutput accepts engine_info: dict | None field."""

    def test_engine_info_defaults_to_none(self):
        out = ClassifyOutput(regimes=np.array([0, 1, 2]))
        assert out.engine_info is None

    def test_engine_info_accepts_dict(self):
        info = {"caveat": "test", "robust_method": "huber"}
        out = ClassifyOutput(regimes=np.array([0, 1, 2]), engine_info=info)
        assert out.engine_info == info

    def test_engine_info_accepts_none_explicitly(self):
        out = ClassifyOutput(regimes=np.array([0, 1, 2]), engine_info=None)
        assert out.engine_info is None


def _make_base_output(**overrides):
    """Create a minimal ClassifyOutput for mocking _hmm_classify_pipeline."""
    defaults = dict(
        regimes=np.array([1, 1, 1]),
        posteriors=None,
        last_regime=1,
        warmup_bars=50,
        n_states=3,
    )
    defaults.update(overrides)
    return ClassifyOutput(**defaults)


class TestHMMEngineInfoPopulation:
    """HMM engines populate engine_info in ClassifyOutput during run_classify."""

    @pytest.mark.parametrize(
        "engine_cls",
        [HMMGenericEngine, HMMMMessinaEngine, RobustHMMEngine, FSHMMEngine],
        ids=lambda c: c.__name__,
    )
    def test_hmm_base_run_classify_populates_engine_info(self, engine_cls):
        """Each HMM engine returns ClassifyOutput with engine_info populated."""
        engine = engine_cls()
        base_output = _make_base_output()

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._hmm_classify_pipeline",
            return_value=base_output,
        ):
            prices = pd.Series([100.0] * 10)
            returns = pd.Series([0.01] * 10)
            result = engine.run_classify(prices, pd.DataFrame(), returns, min_train=5)

        assert isinstance(result, ClassifyOutput)
        assert result.engine_info is not None
        assert isinstance(result.engine_info, dict)
        assert "caveat" in result.engine_info
        assert "labels may swap" in result.engine_info["caveat"]

    def test_robust_hmm_includes_robust_method(self):
        """RobustHMMEngine includes robust_method in engine_info."""
        engine = RobustHMMEngine(robust_method="mcd")
        base_output = _make_base_output()

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._hmm_classify_pipeline",
            return_value=base_output,
        ):
            prices = pd.Series([100.0] * 10)
            returns = pd.Series([0.01] * 10)
            result = engine.run_classify(prices, pd.DataFrame(), returns, min_train=5)

        assert result.engine_info is not None
        assert result.engine_info["robust_method"] == "mcd"

    def test_fshmm_includes_saliency_when_available(self):
        """FSHMMEngine includes feature_saliency when _last_saliency is set."""
        engine = FSHMMEngine()
        engine._last_saliency = [0.1, 0.5, 0.3]
        engine._last_selected_features = ["feat_a", "feat_b"]
        base_output = _make_base_output()

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._hmm_classify_pipeline",
            return_value=base_output,
        ):
            prices = pd.Series([100.0] * 10)
            returns = pd.Series([0.01] * 10)
            result = engine.run_classify(prices, pd.DataFrame(), returns, min_train=5)

        assert result.engine_info is not None
        assert result.engine_info["feature_saliency"] == [0.1, 0.5, 0.3]
        assert result.engine_info["selected_features"] == ["feat_a", "feat_b"]

    def test_fshmm_without_saliency_omits_keys(self):
        """FSHMMEngine omits saliency keys when _last_saliency is not set."""
        engine = FSHMMEngine()
        base_output = _make_base_output()

        with patch(
            "hmm_futures_analysis.regime.engines._hmm_pipeline._hmm_classify_pipeline",
            return_value=base_output,
        ):
            prices = pd.Series([100.0] * 10)
            returns = pd.Series([0.01] * 10)
            result = engine.run_classify(prices, pd.DataFrame(), returns, min_train=5)

        assert result.engine_info is not None
        assert "feature_saliency" not in result.engine_info
        assert "selected_features" not in result.engine_info

    def test_threshold_returns_none_engine_info(self):
        """ThresholdEngine.run_classify returns engine_info=None."""
        engine = ThresholdEngine()
        prices = pd.Series([100.0 + i * 0.5 for i in range(100)])
        prices.index = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = prices.pct_change().dropna()

        result = engine.run_classify(prices, None, returns, min_train=50)

        assert isinstance(result, ClassifyOutput)
        assert result.engine_info is None
