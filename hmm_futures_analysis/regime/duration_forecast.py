"""Regime duration forecasting via survival analysis (issues #28, #29).

Post-processing layer that fits Weibull distributions to historical regime
spell lengths and produces duration forecasts (expected remaining days,
hazard rate, survival quantiles).  Supports Weibull (default) and Cox PH
(via lifelines) models.  Engine-agnostic — works with any regime sequence
from any engine.
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


def _conditional_expected_remaining(shape: float, scale: float, t: float) -> float:
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


def _build_spell_covariates(
    spells: list[_Spell], prices: "pd.Series"
) -> "pd.DataFrame":
    """Build per-spell covariate DataFrame from spells and price series.

    Columns: duration, event, regime_idx, realized_vol, spell_return.
    """
    import pandas as pd

    # Reconstruct start indices from spells by walking the regime array
    start = 0
    rows = []
    for spell in spells:
        end = start + spell.duration
        price_window = prices.iloc[start:end]

        if len(price_window) < 2:
            realized_vol = 0.0
            spell_return = 0.0
        else:
            log_returns = np.log(price_window / price_window.shift(1)).dropna()
            realized_vol = (
                float(log_returns.std(ddof=0)) if len(log_returns) > 0 else 0.0
            )
            if np.isnan(realized_vol):
                realized_vol = 0.0
            spell_return = float(np.log(price_window.iloc[-1] / price_window.iloc[0]))

        rows.append(
            {
                "duration": spell.duration,
                "event": not spell.censored,
                "regime_idx": spell.regime,
                "realized_vol": realized_vol,
                "spell_return": spell_return,
            }
        )
        start = end

    return pd.DataFrame(rows)


def _fit_coxph(
    regimes: np.ndarray,
    spells: list[_Spell],
    prices: "pd.Series",
    current_regime_idx: int,
    days_in_regime: int,
) -> dict | None:
    """Fit Cox PH model and return covariate-adjusted predictions.

    Lazy-imports lifelines. Returns None if fit fails or insufficient data.
    """
    try:
        from lifelines import CoxPHFitter
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Cox PH model requires the 'lifelines' package. "
            "Install with: pip install 'hmm-futures-analysis[survival]'"
        ) from exc

    import pandas as pd

    covariates = _build_spell_covariates(spells, prices)

    # Filter to current regime's completed spells
    regime_covs = covariates[
        (covariates["regime_idx"] == current_regime_idx) & covariates["event"]
    ].copy()

    if len(regime_covs) < _MIN_SPELLS:
        return None

    # Drop regime_idx — not a covariate
    fit_df = regime_covs[["duration", "realized_vol", "spell_return"]].copy()
    fit_df.columns = ["T", "realized_vol", "spell_return"]

    try:
        cph = CoxPHFitter()
        cph.fit(fit_df, duration_col="T")
    except Exception:
        return None

    # Current spell covariates
    current_covs = _build_spell_covariates(spells[-1:], prices[-days_in_regime:])  # noqa: E501
    if len(current_covs) == 0:
        return None

    current_row = pd.DataFrame(
        {
            "realized_vol": [float(current_covs["realized_vol"].iloc[0])],
            "spell_return": [float(current_covs["spell_return"].iloc[0])],
        }
    )

    # Conditional survival prediction
    # NOTE: lifelines' conditional_after does NOT normalize to S(t|T>t)=1,
    # so we manually compute: S(u|T>t) = S(u)/S(t) for u >= t.
    try:
        t_cond = float(days_in_regime)
        sf_raw = cph.predict_survival_function(current_row)
        sf = sf_raw.loc[sf_raw.index >= t_cond]
        if len(sf) == 0:
            expected_remaining = 0.0
        else:
            s_at_t = float(sf.iloc[0])
            if s_at_t <= 0:
                expected_remaining = 0.0
            else:
                sf_normalized = sf / s_at_t
                t_vals = sf_normalized.index.values.astype(float)
                s_vals = sf_normalized.iloc[:, 0].values.astype(float)
                expected_remaining = float(np.trapezoid(s_vals, t_vals))

        # NOTE: realized_vol coefficient can be large in magnitude on short
        # historical windows (e.g. -65 on 89 spells). This is a known limitation
        # of Cox PH with few observations — the model may overfit. Consider
        # increasing MIN_SPELLS or adding L2 regularization for small samples.

        baseline_hazard_at_t = None
        bh = cph.baseline_hazard_
        matching = bh.loc[bh.index >= days_in_regime]
        if len(matching) > 0:
            baseline_hazard_at_t = float(matching.iloc[0]["baseline hazard"])
    except Exception:
        return None

    coefficients = {k: float(v) for k, v in cph.params_.items()}
    concordance = float(cph.concordance_index_)

    return {
        "cox_coefficients": coefficients,
        "concordance_index": round(concordance, 4),
        "baseline_hazard_at_t": round(baseline_hazard_at_t, 4)
        if baseline_hazard_at_t is not None
        else None,
        "cox_expected_remaining_days": round(expected_remaining, 2),
    }


def forecast_duration(
    regimes: np.ndarray,
    state_names: tuple[str, ...] = _STATE_NAMES,
    model: str = "weibull",
    prices: "pd.Series | None" = None,
) -> dict | None:
    """Compute duration forecast for the current regime.

    Parameters
    ----------
    regimes : np.ndarray[int]
        Regime sequence (values 0, 1, 2) from any engine.
    state_names : tuple[str, ...]
        Labels for regime indices.
    model : str
        Survival model: "weibull" (default) or "cox".
    prices : pd.Series | None
        Price series required for model="cox" (covariate computation).
        Ignored for model="weibull".

    Returns
    -------
    dict with keys: current_regime, days_in_regime, expected_remaining_days,
    hazard_rate, survival_50pct, weibull_shape, weibull_scale.
    When model="cox", additional cox_* keys are present.
    Returns None if the regime sequence is empty.
    """
    if model == "cox" and prices is None:
        raise ValueError(
            "Cox PH model requires price data for covariate computation. "
            "Pass a prices Series."
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
        [
            s.duration
            for s in spells
            if s.regime == current_regime_idx and not s.censored
        ],
        dtype=float,
    )

    if len(completed) < _MIN_SPELLS:
        # Not enough history to fit — return partial result with null fields
        result = {
            "current_regime": state_names[current_regime_idx],
            "days_in_regime": days_in_regime,
            "expected_remaining_days": None,
            "hazard_rate": None,
            "survival_50pct": None,
            "weibull_shape": None,
            "weibull_scale": None,
        }
        if model == "cox":
            result["cox_coefficients"] = None
            result["concordance_index"] = None
            result["baseline_hazard_at_t"] = None
            result["cox_expected_remaining_days"] = None
        return result

    shape, scale = _fit_weibull(completed)
    expected_remaining = _conditional_expected_remaining(
        shape, scale, float(days_in_regime)
    )
    h_rate = _hazard_rate(shape, scale, float(days_in_regime))
    median = _median_survival(shape, scale)

    result = {
        "current_regime": state_names[current_regime_idx],
        "days_in_regime": days_in_regime,
        "expected_remaining_days": round(expected_remaining, 2),
        "hazard_rate": round(h_rate, 4),
        "survival_50pct": round(median, 2),
        "weibull_shape": round(shape, 4),
        "weibull_scale": round(scale, 2),
    }

    # --- Cox PH extension (issue #29) ---
    if model == "cox":
        cox_result = _fit_coxph(
            regimes, spells, prices, current_regime_idx, days_in_regime
        )
        if cox_result is not None:
            result.update(cox_result)
        else:
            result["cox_coefficients"] = None
            result["concordance_index"] = None
            result["baseline_hazard_at_t"] = None
            result["cox_expected_remaining_days"] = None

    return result
