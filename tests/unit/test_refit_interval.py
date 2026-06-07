"""Regression tests for Issue D: configurable per-engine refit interval.

Ensures each engine has the correct default_refit_every value and that
custom values are respected.
"""

from __future__ import annotations

import pytest

from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine
from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine


ENGINE_CLASSES_WITH_DEFAULTS = [
    (HMMGenericEngine, 50),
    (HMMMMessinaEngine, 50),
    (RobustHMMEngine, 100),
    (FSHMMEngine, 100),
]


@pytest.mark.parametrize("engine_class,expected", ENGINE_CLASSES_WITH_DEFAULTS)
def test_default_refit_every_per_engine(engine_class, expected):
    """Each engine subclass has the correct default refit interval."""
    engine = engine_class()
    assert engine.default_refit_every == expected, (
        f"{engine_class.__name__}: expected {expected}, "
        f"got {engine.default_refit_every}"
    )


def test_custom_refit_every_on_generic():
    """Custom default_refit_every propagates through constructor."""
    engine = HMMGenericEngine(default_refit_every=200)
    assert engine.default_refit_every == 200


def test_custom_refit_every_on_fshmm():
    """FSHMM custom default_refit_every works alongside other params."""
    engine = FSHMMEngine(
        default_refit_every=200,
        saliency_threshold=0.3,
        max_iter=10,
    )
    assert engine.default_refit_every == 200
    assert engine.saliency_threshold == 0.3
    assert engine.max_iter == 10


def test_custom_refit_every_on_robust():
    """RobustHMM custom default_refit_every works alongside robust_method."""
    engine = RobustHMMEngine(
        default_refit_every=300,
        robust_method="mcd",
    )
    assert engine.default_refit_every == 300
    assert engine.robust_method == "mcd"


def test_base_class_default():
    """HMMEngineBase itself has default_refit_every=50."""
    from hmm_futures_analysis.regime.engines._hmm_engine import HMMEngineBase

    engine = HMMEngineBase()
    assert engine.default_refit_every == 50


def test_config_dataclasses_have_default_refit_every():
    """All HMM config dataclasses expose default_refit_every."""
    from hmm_futures_analysis.regime.engine_configs import (
        FSHMMConfig,
        HMMGenericConfig,
        HMMMMessinaConfig,
        RobustHMMConfig,
    )

    assert HMMGenericConfig().default_refit_every == 50
    assert HMMMMessinaConfig().default_refit_every == 50
    assert RobustHMMConfig().default_refit_every == 100
    assert FSHMMConfig().default_refit_every == 100


def test_default_refit_every_survives_resolve_engine():
    """resolve_engine should pass default_refit_every through to the engine."""
    from hmm_futures_analysis.regime.engine_configs import HMMGenericConfig
    from hmm_futures_analysis.regime.engine_protocol import resolve_engine

    config = HMMGenericConfig(default_refit_every=123)
    engine = resolve_engine(config)
    assert engine.default_refit_every == 123


def test_default_refit_every_survives_resolve_engine_default():
    """resolve_engine should pass through default_refit_every when using config default."""
    from hmm_futures_analysis.regime.engine_configs import RobustHMMConfig
    from hmm_futures_analysis.regime.engine_protocol import resolve_engine

    config = RobustHMMConfig()  # default 100
    engine = resolve_engine(config)
    assert engine.default_refit_every == 100
