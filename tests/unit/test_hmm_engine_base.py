"""Behavioral tests for HMMEngineBase abstract base class (issue #78).

Tests verify observable behavior through public interfaces:
inheritance, construction, precompute routing, enrich_info,
run_classify output shape, and registry integration.

ThresholdEngine is explicitly verified as NOT inheriting from HMMEngineBase.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmm_futures_analysis.regime.engine_configs import (
    FSHMMConfig,
    HMMGenericConfig,
    HMMMMessinaConfig,
    RobustHMMConfig,
    ThresholdConfig,
)
from hmm_futures_analysis.regime.engine_protocol import (
    ENGINE_REGISTRY,
    ClassifyOutput,
    resolve_engine,
)
from hmm_futures_analysis.regime.engines._hmm_engine import HMMEngineBase
from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine
from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine
from hmm_futures_analysis.regime.engines.threshold import ThresholdEngine


# ---- T1: Inheritance contract -------------------------------------------


HMM_ENGINES = [HMMGenericEngine, HMMMMessinaEngine, RobustHMMEngine, FSHMMEngine]


@pytest.mark.parametrize("engine_cls", HMM_ENGINES, ids=lambda c: c.__name__)
def test_hmm_engine_inherits_base(engine_cls):
    """Each HMM engine is a subclass of HMMEngineBase."""
    assert issubclass(engine_cls, HMMEngineBase)


def test_threshold_does_not_inherit_base():
    """ThresholdEngine must NOT inherit from HMMEngineBase."""
    assert not issubclass(ThresholdEngine, HMMEngineBase)


# ---- T2: Cannot instantiate abstract base --------------------------------


def test_base_class_is_abstract():
    """HMMEngineBase cannot be instantiated (classify is @abstractmethod)."""
    with pytest.raises(TypeError):
        HMMEngineBase()


# ---- T3: Concrete engines construct correctly ----------------------------


@pytest.mark.parametrize("engine_cls", HMM_ENGINES, ids=lambda c: c.__name__)
def test_default_construction(engine_cls):
    """Each HMM engine constructs with default parameters."""
    engine = engine_cls()
    assert engine.n_states == 3
    assert engine._pca_n_components is None


@pytest.mark.parametrize("engine_cls", HMM_ENGINES, ids=lambda c: c.__name__)
def test_custom_construction(engine_cls):
    """Each HMM engine accepts n_states and pca_variance."""
    engine = engine_cls(n_states=5, pca_variance=0.95)
    assert engine.n_states == 5
    assert engine.pca_variance == 0.95


def test_robust_hmm_has_robust_method():
    """RobustHMMEngine stores its unique robust_method parameter."""
    engine = RobustHMMEngine(robust_method="mcd")
    assert engine.robust_method == "mcd"


def test_fshmm_has_saliency_params():
    """FSHMMEngine stores its unique saliency parameters."""
    engine = FSHMMEngine(saliency_threshold=0.7, max_iter=100)
    assert engine.saliency_threshold == 0.7
    assert engine.max_iter == 100


# ---- T4: precompute routes through use_messina ---------------------------


def test_generic_and_messina_produce_different_features(sample_ohlcv):
    """Generic and Messina engines produce different feature column sets."""
    generic = HMMGenericEngine()
    messina = HMMMMessinaEngine()

    generic_feats = generic.precompute(sample_ohlcv)
    messina_feats = messina.precompute(sample_ohlcv)

    assert generic_feats is not None
    assert messina_feats is not None
    assert set(generic_feats.columns) != set(messina_feats.columns)


# ---- T5: precompute rejects None data -----------------------------------


@pytest.mark.parametrize("engine_cls", HMM_ENGINES, ids=lambda c: c.__name__)
def test_precompute_rejects_none(engine_cls):
    """All HMM engines raise ValueError when precompute(None) is called."""
    engine = engine_cls()
    with pytest.raises(ValueError, match="requires OHLCV data"):
        engine.precompute(None)


# ---- T6: enrich_info adds caveat ----------------------------------------


@pytest.mark.parametrize("engine_cls", HMM_ENGINES, ids=lambda c: c.__name__)
def test_enrich_info_adds_caveat(engine_cls):
    """enrich_info copies dict and adds the standard HMM caveat."""
    engine = engine_cls()
    info = {"engine": "test"}
    enriched = engine.enrich_info(info)
    assert "caveat" in enriched
    assert "labels may swap" in enriched["caveat"]
    # Original dict is not mutated
    assert "caveat" not in info


def test_robust_enrich_info_adds_method():
    """RobustHMMEngine.enrich_info adds robust_method to the output."""
    engine = RobustHMMEngine(robust_method="mcd")
    enriched = engine.enrich_info({"engine": "test"})
    assert enriched["robust_method"] == "mcd"


# ---- T7: run_classify produces ClassifyOutput ---------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        HMMGenericConfig(n_states=3),
        HMMMMessinaConfig(n_states=3, pca_variance=None),
    ],
    ids=["hmm", "messina"],
)
def test_run_classify_returns_classify_output(sample_ohlcv, config):
    """run_classify returns a ClassifyOutput with expected fields."""
    engine = resolve_engine(config)
    prices = sample_ohlcv["close"]
    returns = prices.pct_change().dropna()

    result = engine.run_classify(prices, sample_ohlcv, returns, min_train=50)

    assert isinstance(result, ClassifyOutput)
    assert isinstance(result.regimes, np.ndarray)
    assert len(result.regimes) == len(returns)
    assert result.last_regime in {0, 1, 2}
    assert result.n_states == 3


# ---- T8: Registry integration -------------------------------------------


REGISTRY_CONFIGS = [
    ("threshold", ThresholdConfig, False),
    ("hmm", HMMGenericConfig, True),
    ("messina", HMMMMessinaConfig, True),
    ("robust_hmm", RobustHMMConfig, True),
    ("fshmm", FSHMMConfig, True),
]


@pytest.mark.parametrize(
    "name,config_cls,expected_hmm",
    REGISTRY_CONFIGS,
    ids=[t[0] for t in REGISTRY_CONFIGS],
)
def test_resolve_engine_produces_correct_type(name, config_cls, expected_hmm):
    """resolve_engine with each config produces the right type."""
    engine = resolve_engine(config_cls())
    assert isinstance(engine, HMMEngineBase) == expected_hmm


def test_all_engines_in_registry():
    """All five engines are present in ENGINE_REGISTRY."""
    assert set(ENGINE_REGISTRY.keys()) == {
        "threshold", "hmm", "messina", "robust_hmm", "fshmm"
    }
