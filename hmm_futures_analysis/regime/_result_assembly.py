"""Result assembly for the regime detection pipeline.

Extracts result construction from the pipeline god-function into a
dedicated helper.  All inputs arrive as keyword-only arguments so the
call site in ``pipeline.run()`` is self-documenting.
"""

from __future__ import annotations

import math
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from .markov_chain import forecast_n_steps

if TYPE_CHECKING:
    from .pipeline import MarkovStats

# ── Constants ───────────────────────────────────────────────────────

_STATE_NAMES = ("bear", "sideways", "bull")
_FRAMEWORK_VERSION = "hmm_test v0.2.0"
_DISCLAIMER = (
    "Regime detection is probabilistic. Past transitions do not guarantee "
    "future regimes. Not financial advice."
)


# ── Result type ─────────────────────────────────────────────────────


class PipelineResult(NamedTuple):
    """Immutable result from :func:`pipeline.run`.

    Call ``._asdict()`` to get a JSON-compatible dict for serialization.
    """

    source: str
    engine: str
    dates: dict[str, str]
    current_regime: dict[str, str | int]
    signal: float
    next_state_probabilities: dict[str, float]
    transition_matrix: list
    stationary_distribution: dict[str, float]
    persistence_diagonal: dict[str, float]
    regime_counts: dict[str, int]
    walk_forward: dict
    forecast: dict
    engine_info: dict
    framework: str
    disclaimer: str
    verdict: dict
    duration_forecast: dict | None = None
    regime_transitions: list | None = None
    timing: dict | None = None


# ── Helpers ─────────────────────────────────────────────────────────


def _probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    return {
        "bear": float(probs[0]),
        "sideways": float(probs[1]),
        "bull": float(probs[2]),
    }


def _nan_to_none(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _compute_dynamic_threshold(
    duration_forecast: dict | None,
    base_threshold: float = 0.1,
) -> float:
    """Compute a regime-aging-adjusted threshold for Sideways verdict."""
    if duration_forecast is None:
        return base_threshold

    days_in = duration_forecast.get("days_in_regime")
    scale = duration_forecast.get("weibull_scale")
    shape = duration_forecast.get("weibull_shape")

    if days_in is None or scale is None or shape is None or shape <= 0:
        return base_threshold

    from scipy.special import gamma as _gamma  # type: ignore[unused-ignore]

    _days = float(days_in)
    _scale = float(scale)
    _shape = float(shape)
    expected_total = _scale * _gamma(1.0 + 1.0 / _shape)
    if expected_total <= 0:
        return base_threshold

    aging_ratio = _days / expected_total

    if aging_ratio <= 1.0:
        return base_threshold

    threshold_mult = max(0.3, 2.0 - aging_ratio)
    return base_threshold * threshold_mult


def _compute_verdict(
    current_regime: int,
    signal: float,
    forecast_1: dict[str, float],
    sideways_threshold: float = 0.1,
) -> dict[str, object]:
    """Synthesize regime + 1-step forecast into a single actionable verdict."""
    if current_regime == 2:  # Bull
        if forecast_1["bull"] > forecast_1.get("bear", 0):
            verdict = "bullish"
        else:
            verdict = "transition_bear"
    elif current_regime == 0:  # Bear
        if forecast_1["bear"] > forecast_1.get("bull", 0):
            verdict = "bearish"
        else:
            verdict = "transition_bull"
    else:  # Sideways
        if abs(signal) < sideways_threshold:
            verdict = "neutral"
        elif signal > 0:
            verdict = "transition_bull"
        else:
            verdict = "transition_bear"

    return {
        "verdict": verdict,
        "confidence": round(float(abs(signal)), 4),
    }


# ── Main assembly function ──────────────────────────────────────────


def _assemble_result(
    *,
    source: str,
    engine_name: str,
    markov: MarkovStats,
    walk_forward: dict,
    engine_info: dict,
    timing: dict | None,
    duration_forecast_result: dict | None = None,
    regime_transitions: list | None = None,
) -> PipelineResult:
    """Assemble a PipelineResult from its constituent parts.

    All arguments are keyword-only so the call site in ``pipeline.run()``
    is self-documenting.
    """
    # Forecast probabilities (1-step only, ADR-0025).  With current_probs as
    # the one-hot e_current (issue #112) this is T[current_regime, :] — the
    # same distribution as next_state_probabilities (SKILL.md JSON contract:
    # both fields carry the current-regime row of the transition matrix).
    next_probs = forecast_n_steps(markov.transmat, markov.current_probs, 1)
    forecast_1 = _probs_to_dict(next_probs)

    # Verdict
    sideways_threshold = _compute_dynamic_threshold(duration_forecast_result)
    verdict_out = _compute_verdict(
        markov.current_regime,
        markov.signal,
        forecast_1,
        sideways_threshold=sideways_threshold,
    )

    return PipelineResult(
        source=source,
        engine=engine_name,
        dates={
            "start": markov.dates["start"],
            "end": markov.dates["end"],
        },
        current_regime={
            "name": _STATE_NAMES[markov.current_regime],
            "index": markov.current_regime,
        },
        signal=markov.signal,
        next_state_probabilities=_probs_to_dict(next_probs),
        transition_matrix=markov.transmat.tolist(),
        stationary_distribution=_probs_to_dict(markov.stationary),
        persistence_diagonal=markov.persistence,
        regime_counts=markov.regime_counts,
        walk_forward=walk_forward,
        forecast={
            "1_step": forecast_1,
        },
        engine_info=engine_info,
        framework=_FRAMEWORK_VERSION,
        disclaimer=_DISCLAIMER,
        verdict=verdict_out,
        duration_forecast=duration_forecast_result,
        regime_transitions=regime_transitions,
        timing=timing,
    )
