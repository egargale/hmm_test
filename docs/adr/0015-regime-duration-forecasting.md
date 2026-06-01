# Regime duration forecasting via survival analysis (Weibull + Cox PH)

The pipeline outputs the current regime label and transition matrix, but a
trader needs to know "how much longer will this bull market last?" Duration
forecasting answers that with expected remaining days and hazard rates,
using survival analysis models fit to historical regime spell lengths.

## Considered options

### A) No forecast (status quo)

Rely on the transition matrix alone. The trader infers duration from the
one-step transition probabilities — e.g. if P(Bull → Bull) = 0.95, the
expected remaining time is 1/(1-0.95) = 20 days under a memoryless
assumption.

**Rejected because**: the transition matrix captures one-step probability,
not the conditional distribution of sojourn times. The memoryless
assumption (geometric distribution) is implicit and rarely tested — actual
regime durations may be Weibull-shaped (increasing or decreasing hazard),
making the geometric a poor fit. The trader has no way to compute
"P(duration > 30 | already 10 days in)" without a survival model.

### B) Simple empirical distribution

Collect the histogram of all completed spell lengths for each regime.
Report quantiles and mean directly from data — no parametric model.

**Rejected because**: (1) the empirical distribution cannot extrapolate
beyond the longest observed spell; (2) E[T−t | T > t] requires observations
of spells that lasted at least `t` days, which shrinks to zero as `t` grows;
(3) no closed-form hazard rate — the hazard is a noisy histogram ratio.

### C) Weibull survival analysis (chosen — default)

Fit a Weibull distribution to completed spell lengths via MLE (scipy),
then compute closed-form conditional expected remaining days, hazard rate,
and median survival.

**Chosen because**: (1) zero new hard dependencies — scipy is already a
project dependency; (2) Weibull is flexible enough to model increasing
(shape > 1), decreasing (shape < 1), or constant (shape ≈ 1, exponential)
hazard — the three canonical regime-aging behaviours; (3) the conditional
expectation E[T−t | T > t] has a tractable numerical integral via
`scipy.integrate.quad`; (4) shape and scale are interpretable — traders can
see "this bull market has shape 2.3, meaning its risk of ending increases
with age"; (5) method-of-moments initialisation makes MLE reliable on as
few as 3–5 spells.

### D) Cox Proportional Hazards model (chosen — opt-in extension)

Fit a Cox PH model with per-spell covariates (realised volatility, spell
return) to adjust the baseline hazard for market conditions.

**Chosen because**: (1) covariate adjustment is valuable — a bull market
with high realised vol may have a shorter expected remaining life than one
with low vol, even with the same baseline hazard; (2) it is an *additive*
extension to the Weibull model — both models are fit, and Cox adds
`cox_coefficients`, `concordance_index`, and `cox_expected_remaining_days`
to the output dict; (3) lazy-imports `lifelines` so the project has zero
runtime dependency — users opt in with `pip install
'hmm-futures-analysis[survival]'`.

**Tradeoffs**: (1) Cox PH requires price data for covariate computation —
passing `prices=None` with `--duration-model cox` raises a clear ValueError;
(2) on short histories (< 10 completed spells), Cox coefficients can be
unstable — the code documents known cases of realised_vol coefficients as
large as −65 on 89 spells; (3) `lifelines` is a non-trivial dependency
(patched scikit-learn-style API, 500+ KB installed).

### E) Bayesian structural time series

Model the entire price series with a Bayesian state-space model that
learns regime durations as latent variables (e.g. Markov-switching with
duration-dependent transition probabilities, or a changepoint model with
recurrent Chinese restaurant process prior).

**Rejected because**: (1) heavy compute — MCMC or variational inference
on every walk-forward slice would add minutes per run; (2) the project
already has four regime-detection engines — a full Bayesian model would
be a fifth engine, not a post-processing layer; (3) no clear accuracy
improvement over Weibull for the marginal task of "how many more days in
this bull market?" — the Weibull fit to spell lengths already captures
the empirical duration distribution.

## Decision details

### Weibull (default) — no new dependencies

```python
# Core fitting — scipy only
shape, scale = _fit_weibull(completed_durations)
expected_remaining = _conditional_expected_remaining(shape, scale, days_in)
hazard = _hazard_rate(shape, scale, days_in)
median = _median_survival(shape, scale)
```

The function `forecast_duration()` in
`hmm_futures_analysis/regime/duration_forecast.py` is the single public
interface. It is engine-agnostic — it operates on a `np.ndarray` of regime
labels and returns a flat dict. No engine imports it; the pipeline calls it
as an optional post-processing step.

### Cox PH (opt-in) — lazy lifelines import

```python
if model == "cox":
    cox_result = _fit_coxph(regimes, spells, prices, current_regime_idx,
                            days_in_regime)
    if cox_result is not None:
        result.update(cox_result)
```

Cox is always additive to Weibull — even when `--duration-model cox` is
active, the Weibull parameters are computed first and Cox fields are merged
into the output. If the Cox fit fails (insufficient spells, lifelines
missing, numerical failure), the Cox fields are set to `None` and the
Weibull forecast remains available.

### CLI flags (both opt-in, backward-compatible)

| Flag | Default | Choices | Purpose |
|------|---------|---------|---------|
| `--duration-forecast` | `False` | flag | Enable duration forecasting (opt-in) |
| `--duration-model` | `weibull` | `weibull`, `cox` | Survival model variant |

### Engine-agnostic protocol

The duration forecast runs on the *output* of any engine — threshold,
messina, hmm, or fshmm. It reads `classify_out.regimes` (a 1-D array of
int labels) and optionally `prices` (a pd.Series for Cox covariates).
It does not know or care which engine produced the regimes. This preserves
[[ADR-0003]] (engine self-containment) and [[ADR-0001]] (independent
engines).

## Consequences

### 3-spell minimum (`_MIN_SPELLS = 3`)

At least 3 completed spells of the current regime type are required to fit
the Weibull distribution. With fewer than 3, the forecast returns `None`
for all numerical fields and the caller sees `expected_remaining_days: null`.
This guards against degenerate fits on very short histories. The threshold
was chosen empirically: 2 points produce a perfect fit (2 parameters) with
no degrees of freedom for error estimation; 3 points provide a minimum
diagnostic check.

### Weibull shape interpretation

- **shape > 1**: increasing hazard (regime becomes more likely to end as
  it ages — "this bull market is getting old")
- **shape ≈ 1**: exponential (memoryless — duration is independent of age)
- **shape < 1**: decreasing hazard (regime stabilises over time — "this
  bear market is settling in")

These interpretations are documented in the function docstring and surfaced
in the JSON output for downstream consumption.

### Dynamic threshold coupling

The duration forecast is consumed by `_compute_dynamic_threshold()` in
`pipeline.py` to adjust the Sideways verdict threshold. When the current
regime has outlasted its Weibull-expected total duration by 1.7×, the
threshold shrinks linearly to 0.3× of its base value. This makes the
verdict logic progressively more sensitive to a regime change signal as
the regime ages — a practical application of the survival analysis output.

This coupling has one downside: if `forecast_duration()` returns `None`
(e.g. insufficient spells), the dynamic threshold degrades gracefully to
`base_threshold`, so verdict generation is never blocked.

### Cox covariate risk on small samples

The code includes a documented warning that Cox PH coefficients can be
large in magnitude on small historical windows (e.g. realised_vol
coefficient of −65 on 89 spells). This is a known small-sample behaviour
of Cox PH — the model overfits when the number of events is small relative
to the number of covariates. Future work could add L2 regularisation
(ridge Cox) or increase `_MIN_SPELLS` when covariates are active.

### Install-time dependency split

- **`pip install hmm-futures-analysis`**: Weibull only (scipy, zero new deps)
- **`pip install 'hmm-futures-analysis[survival]'**: Weibull + Cox (adds
  `lifelines>=0.27.0`)

The `pyproject.toml` declares `survival = ['lifelines>=0.27.0']` as an
optional dependency.

### Output contract

When `--duration-forecast` is active, the JSON output gains a
`duration_forecast` block:

```json
{
  "duration_forecast": {
    "current_regime": "bull",
    "days_in_regime": 15,
    "expected_remaining_days": 22.34,
    "hazard_rate": 0.0183,
    "survival_50pct": 35.00,
    "weibull_shape": 2.15,
    "weibull_scale": 42.00,
    "cox_coefficients": {
      "realized_vol": -0.87,
      "spell_return": 0.32
    },
    "concordance_index": 0.72,
    "baseline_hazard_at_t": 0.0054,
    "cox_expected_remaining_days": 18.50
  }
}
```

This block is documented in the README output contract and verified by CI
tests (`test_docs_currency.py`).

### Cross-references

- [[ADR-0007]] — Hysteresis and dwell-time filters in walk-forward layer.
  Both features are post-processing steps that modify the pipeline output
  without changing engine internals.
- [[ADR-0003]] — Engine self-containment contract. The duration forecast
  operates on engine *output* (regime labels), not engine internals.
- [[ADR-0001]] — Three independent engines. Duration forecast is
  engine-agnostic and does not create cross-engine coupling.
