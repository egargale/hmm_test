"""Unit tests for pipeline._build_engine_info helper."""

import numpy as np

from hmm_futures_analysis.regime.engine_configs import (
    HMMGenericConfig,
    ThresholdConfig,
)
from hmm_futures_analysis.regime.engine_protocol import ClassifyOutput
from hmm_futures_analysis.regime.pipeline import _build_engine_info


class TestBuildEngineInfo:
    """_build_engine_info builds engine_info dict from config and classify output."""

    def test_threshold_config_base_info(self):
        config = ThresholdConfig()
        out = ClassifyOutput(regimes=np.array([0, 1, 2]))
        result = _build_engine_info(config, resolved_n_states=3, classify_out=out)
        assert result["method"] == "threshold"
        assert result["features"] == "returns"
        assert result["n_states"] == 3

    def test_hmm_config_base_info(self):
        config = HMMGenericConfig()
        out = ClassifyOutput(regimes=np.array([0, 1, 2]))
        result = _build_engine_info(config, resolved_n_states=3, classify_out=out)
        assert result["method"] == "hmm"
        assert "features" in result
        assert result["n_states"] == 3

    def test_classify_out_with_no_engine_info(self):
        """classify_out.engine_info=None returns base dict only."""
        config = ThresholdConfig()
        out = ClassifyOutput(regimes=np.array([0, 1, 2]), engine_info=None)
        result = _build_engine_info(config, resolved_n_states=3, classify_out=out)
        assert result["method"] == "threshold"
        assert set(result.keys()) == {"method", "features", "n_states"}

    def test_classify_out_with_engine_info(self):
        """classify_out.engine_info dict merges into the base dict."""
        config = ThresholdConfig()
        info = {"warmup_bars": 252, "caveat": "test"}
        out = ClassifyOutput(regimes=np.array([0, 1, 2]), engine_info=info)
        result = _build_engine_info(config, resolved_n_states=3, classify_out=out)
        assert result["method"] == "threshold"
        assert result["warmup_bars"] == 252
        assert result["caveat"] == "test"

    def test_resolved_n_states_propagated(self):
        config = ThresholdConfig()
        out = ClassifyOutput(regimes=np.array([0, 1, 2]))
        result = _build_engine_info(config, resolved_n_states=5, classify_out=out)
        assert result["n_states"] == 5
