"""Regime duration forecasting via survival analysis (issue #28).

Post-processing layer that fits Weibull distributions to historical regime
spell lengths and produces duration forecasts (expected remaining days,
hazard rate, survival quantiles).  Engine-agnostic — works with any regime
sequence from any engine.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import weibull_min

_STATE_NAMES = ("bear", "sideways", "bull")
_MIN_SPELLS = 3  # minimum completed spells per regime to fit


class _Spell(NamedTuple):
    regime: int
    duration: int
    censored: bool


def _extract_spells(regimes: np.ndarray) -> list[_Spell]:
    """Walk the regime sequence and return contiguous spells.

    The final spell (current regime) is marked as right-censored because
    it hasn't ended yet.
    """
    if len(regimes) == 0:
        return []

    spells: list[_Spell] = []
    start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[start]:
            spells.append(_Spell(int(regimes[start]), i - start, False))
            start = i
    # Final spell — right-censored
    spells.append(_Spell(int(regimes[start]), len(regimes) - start, True))
    return spells


def _fit_weibull(durations: np.ndarray) -> tuple[float, float]:
    """Fit a Weibull distribution to completed spell durations.

    Returns (shape, scale) using maximum-likelihood via scipy.optimize.
    """
    # Use method-of-moments as initial guess for numerical stability
    mean_d = np.mean(durations)
    std_d = np.std(durations, ddof=1)
    if std_d <= 0 or mean_d <= 0:
        # Degenerate — return shape=1 (exponential), scale=mean
        return 1.0, float(mean_d)

    c_v = std_d / mean_d
    # Approximate shape from coefficient of variation
    shape0 = max(0.5, 1.0 / c_v)
    scale0 = mean_d

    def neg_ll(params: np.ndarray) -> float:
        c, lam = params
        if c <= 0 or lam <= 0:
            return 1e30
        ll = np.sum(weibull_min.logpdf(durations, c, scale=lam))
        return -ll

    result = minimize(neg_ll, x0=[shape0, scale0], method="Nelder-Mead")
    shape = float(result.x[0])
    scale = float(result.x[1])
    return shape, scale


def _conditional_expected_remaining(
    shape: float, scale: float, t: float
) -> float:
    """E[T − t | T > t] for a Weibull(shape, scale).

    Uses numerical integration of the survival function:
        E[T-t | T>t] = ∫_t^∞ S(u) du / S(t)
    where S is the Weibull survival function.
    """
    sf_t = weibull_min.sf(t, shape, scale=scale)
    if sf_t <= 0:
        return 0.0

    integral, _ = quad(lambda u: weibull_min.sf(u, shape, scale=scale), t, np.inf)
    return float(integral / sf_t)


def _hazard_rate(shape: float, scale: float, t: float) -> float:
    """Instantaneous hazard rate h(t) = f(t)/S(t) for Weibull(shape, scale)."""
    sf = weibull_min.sf(t, shape, scale=scale)
    if sf <= 0:
        return 0.0
    return float(weibull_min.pdf(t, shape, scale=scale) / sf)


def _median_survival(shape: float, scale: float) -> float:
    """Median of Weibull distribution: scale * (ln 2)^(1/shape)."""
    return float(scale * (np.log(2) ** (1.0 / shape)))


def forecast_duration(
    regimes: np.ndarray,
    state_names: tuple[str, ...] = _STATE_NAMES,
    model: str = "weibull",
) -> dict | None:
    """Compute duration forecast for the current regime.

    Parameters
    ----------
    regimes : np.ndarray[int]
        Regime sequence (values 0, 1, 2) from any engine.
    state_names : tuple[str, ...]
        Labels for regime indices.
    model : str
        Survival model. Only "weibull" is supported; "cox" raises ImportError.

    Returns
    -------
    dict with keys: current_regime, days_in_regime, expected_remaining_days,
    hazard_rate, survival_50pct, weibull_shape, weibull_scale.
    Returns None if the regime sequence is empty.
    """
    if model == "cox":
        raise ImportError(
            "Cox PH model requires statsmodels. "
            "Install with: pip install statsmodels. "
            "Currently only --duration-model weibull is supported."
        )

    if len(regimes) == 0:
        return None

    spells = _extract_spells(regimes)
    if not spells:
        return None

    # Identify the current (censored) spell
    current_spell = spells[-1]
    current_regime_idx = current_spell.regime
    days_in_regime = current_spell.duration

    # Collect completed spells for the current regime
    completed = np.array(
        [s.duration for s in spells if s.regime == current_regime_idx and not s.censored],
        dtype=float,
    )

    if len(completed) < _MIN_SPELLS:
        # Not enough history to fit — return partial result with null fields
        return {
            "current_regime": state_names[current_regime_idx],
            "days_in_regime": days_in_regime,
            "expected_remaining_days": None,
            "hazard_rate": None,
            "survival_50pct": None,
            "weibull_shape": None,
            "weibull_scale": None,
        }

    shape, scale = _fit_weibull(completed)
    expected_remaining = _conditional_expected_remaining(shape, scale, float(days_in_regime))
    h_rate = _hazard_rate(shape, scale, float(days_in_regime))
    median = _median_survival(shape, scale)

    return {
        "current_regime": state_names[current_regime_idx],
        "days_in_regime": days_in_regime,
        "expected_remaining_days": round(expected_remaining, 2),
        "hazard_rate": round(h_rate, 4),
        "survival_50pct": round(median, 2),
        "weibull_shape": round(shape, 4),
        "weibull_scale": round(scale, 2),
    }
