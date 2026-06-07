"""Regression tests for Issue A: engine deduplication.

Verifies that HMMGenericEngine, HMMMMessinaEngine, and RobustHMMEngine
share the base class classify() method, while FSHMMEngine retains its own.
"""

from __future__ import annotations

import pytest

from hmm_futures_analysis.regime.engines._hmm_engine import HMMEngineBase
from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine
from hmm_futures_analysis.regime.engines.hmm_generic import HMMGenericEngine
from hmm_futures_analysis.regime.engines.hmm_messina import HMMMMessinaEngine
from hmm_futures_analysis.regime.engines.robust_hmm import RobustHMMEngine


# ---- classify() deduplication ----


def test_hmm_generic_does_not_define_classify():
    """HMMGenericEngine inherits classify() from base, does not define its own."""
    assert "classify" not in HMMGenericEngine.__dict__, (
        "HMMGenericEngine should not define its own classify()"
    )


def test_hmm_messina_does_not_define_classify():
    """HMMMMessinaEngine inherits classify() from base, does not define its own."""
    assert "classify" not in HMMMMessinaEngine.__dict__, (
        "HMMMMessinaEngine should not define its own classify()"
    )


def test_robust_hmm_does_not_define_classify():
    """RobustHMMEngine inherits classify() from base, does not define its own."""
    assert "classify" not in RobustHMMEngine.__dict__, (
        "RobustHMMEngine should not define its own classify()"
    )


def test_fshmm_defines_own_classify():
    """FSHMMEngine should still override classify() with its own custom EM."""
    assert "classify" in FSHMMEngine.__dict__, (
        "FSHMMEngine should define its own classify()"
    )


# ---- _fit_on_slice() hook ----


def test_base_has_fit_on_slice():
    """HMMEngineBase defines _fit_on_slice as a concrete method."""
    assert hasattr(HMMEngineBase, "_fit_on_slice")
    assert callable(HMMEngineBase._fit_on_slice)


def test_robust_hmm_overrides_fit_on_slice():
    """RobustHMMEngine overrides _fit_on_slice for robust correction."""
    assert "_fit_on_slice" in RobustHMMEngine.__dict__, (
        "RobustHMMEngine should override _fit_on_slice"
    )


def test_generic_engine_uses_base_fit_on_slice():
    """HMMGenericEngine does NOT override _fit_on_slice — uses base."""
    assert "_fit_on_slice" not in HMMGenericEngine.__dict__, (
        "HMMGenericEngine should not override _fit_on_slice"
    )


def test_messina_engine_uses_base_fit_on_slice():
    """HMMMMessinaEngine does NOT override _fit_on_slice — uses base."""
    assert "_fit_on_slice" not in HMMMMessinaEngine.__dict__, (
        "HMMMMessinaEngine should not override _fit_on_slice"
    )


# ---- class identity ----


def test_use_messina_flags():
    """Only messina engine has use_messina = True."""
    assert HMMGenericEngine.use_messina is False
    assert HMMMMessinaEngine.use_messina is True
    assert RobustHMMEngine.use_messina is False
    assert FSHMMEngine.use_messina is False


# ---- classify works end-to-end ----


@pytest.mark.parametrize(
    "engine_class",
    [HMMGenericEngine, HMMMMessinaEngine, RobustHMMEngine],
)
def test_classify_returns_valid_result(engine_class):
    """Each deduplicated engine's classify() produces a valid ClassifyResult."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    n = 200
    data = pd.DataFrame(
        {
            "log_ret": rng.randn(n) * 0.01,
            "volatility": np.abs(rng.randn(n)) * 0.02,
            "momentum": rng.randn(n) * 0.05,
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    engine = engine_class(n_states=3, pca_variance=None)
    result = engine.classify(data)

    assert result.regime in {0, 1, 2}
    assert result.means is not None
    assert result.posteriors is not None
    assert result.posteriors.shape == (3,)
    assert abs(result.posteriors.sum() - 1.0) < 1e-6
