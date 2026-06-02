"""Unit tests for pipeline._build_engine_info helper."""

from hmm_futures_analysis.regime.engine_configs import ThresholdConfig
from hmm_futures_analysis.regime.pipeline import _build_engine_info


class TestBuildEngineInfo:
    """_build_engine_info builds engine_info dict from config and engine."""

    def test_threshold_config_base_info(self):
        config = ThresholdConfig()
        result = _build_engine_info(config, resolved_n_states=3, eng=None)
        assert result["method"] == "threshold"
        assert result["features"] == "returns"
        assert result["n_states"] == 3

    def test_hmm_config_base_info(self):
        from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig

        config = HMMGenericConfig()
        result = _build_engine_info(config, resolved_n_states=3, eng=None)
        assert result["method"] == "hmm"
        assert "features" in result
        assert result["n_states"] == 3

    def test_engine_without_enrich_info(self):
        """Engine without enrich_info method returns base dict only."""

        class BareEngine:
            pass

        config = ThresholdConfig()
        result = _build_engine_info(config, resolved_n_states=3, eng=BareEngine())
        assert result["method"] == "threshold"
        # No extra keys beyond the base three
        assert set(result.keys()) == {"method", "features", "n_states"}

    def test_engine_with_enrich_info(self):
        """Engine with enrich_info merges extras into the dict."""

        class RichEngine:
            def enrich_info(self, ctx: dict) -> dict:
                return {"warmup_bars": ctx.get("warmup_bars", 0), "caveat": "test"}

        config = ThresholdConfig()
        result = _build_engine_info(
            config, resolved_n_states=3, eng=RichEngine(), warmup_bars=252
        )
        assert result["method"] == "threshold"
        assert result["warmup_bars"] == 252
        assert result["caveat"] == "test"

    def test_resolved_n_states_propagated(self):
        config = ThresholdConfig()
        result = _build_engine_info(config, resolved_n_states=5, eng=None)
        assert result["n_states"] == 5
