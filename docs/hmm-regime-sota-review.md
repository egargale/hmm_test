# HMM Regime Detection: How It Works & SOTA Soundness Assessment

## 1. How Regime Detection Works (End-to-End Pipeline)

The system implements **5 engines** sharing a common pipeline architecture:

```
Raw OHLCV → Feature Engineering → HMM Fit (EM) → State → Regime Mapping
                                                          ↓
                                              Walk-Forward Classify
                                                          ↓
                                          Whipsaw Filters (dwell + hysteresis)
                                                          ↓
                                    Markov Stats + Duration Forecast + Verdict
```

### Step-by-step

**① Feature Engineering** (`feature_engineering.py`, `messina_features.py`)
- **Generic**: ~50 SMA-based features (returns, momentum, volatility ratios)
- **Messina**: 19 Wilder's-smoothed features (ADX, RSI, ATR-based)
- Z-score normalization: `(X - mean) / (std + 1e-8)` per feature

**② HMM Fitting** (`_hmm_engine.py:45-93`)
- Uses `hmmlearn.GaussianHMM` with **diagonal covariance** (`covariance_type="diag"`)
- EM: `n_iter=30`, `tol=1e-4`, initialized with `random_state=42`
- Optional PCA whitening (ADR-0005) when `pca_variance` is set
- **Auto state count**: BIC over k∈[2, min(max_states, n//10)] with 3 restarts per k (`_hmm_pipeline.py:23-97`)

**③ Regime Label Mapping** (`_hmm_engine.py:217-273`)
- Sorts HMM states by `means_[:, 0]` (first feature dimension = return signal)
- Lowest mean → Bear (0), middle → Sideways (1), highest → Bull (2)
- For n_states > 3: collapses to 3 buckets via `min(2, i * 3 // n)`
- Cross-cycle state remapping via greedy nearest-neighbor matching on means

**④ Walk-Forward Classification** (`_hmm_pipeline.py:180-243`)
- Expanding window from `min_train=252` bars
- Adaptive refit cadence: `refit_every = min(20, max(1, (n - min_train) // 100))`
- At refit bars: full `classify()` call with `prev_means` for label consistency
- Between refits: carry forward last result

**⑤ Whipsaw Filtering** (`walk_forward.py:202-237`)
- **Dwell-time**: requires `dwell_bars` consecutive bars with same new regime
- **Hysteresis**: requires posterior margin `P(new) - P(current) > delta`
- AND logic: both must agree to switch

**⑥ Duration Forecasting** (`duration_forecast.py`)
- Weibull survival analysis on regime spell lengths (MLE fit)
- Conditional expected remaining days: `E[T-t | T>t]` via numerical integration
- Optional Cox PH with realized-vol and spell-return covariates
- **Dynamic threshold**: shrinks sideways verdict when regime outlasts Weibull expected duration

---

## 2. Soundness Assessment Against State-of-the-Art

### ✅ What the project does well (SOTA-aligned)

| Aspect | Implementation | SOTA Status |
|--------|---------------|-------------|
| **Gaussian HMM foundation** | `hmmlearn.GaussianHMM`, diagonal covariance | ✅ Standard baseline for financial regime detection (Hamilton 1989, 1990) |
| **Walk-forward evaluation** | Expanding window, no lookahead, lagged positions | ✅ Correct — prevents the #1 sin in regime detection (lookahead bias) |
| **BIC model selection** | Multi-restart BIC over candidate state counts | ✅ Standard practice (McLachlan & Peel 2000). The `n // 10` cap for effective_max is a sensible data-adaptive guard |
| **Label permutation problem** | State remapping via nearest-neighbor matching on means across refit cycles | ✅ This is a known hard problem; greedy NN on means is a pragmatic and widely-used solution |
| **Whipsaw/dwell filters** | AND logic with configurable dwell + hysteresis | ✅ Industry-standard approach to reduce regime-switching noise |
| **Feature Saliency HMM** | Adams et al. 2016 with per-feature ρ_k, Bernoulli latent saliency, MAP update | ✅ Correctly implements the paper's EM formulation |
| **Survival analysis** | Weibull MLE with conditional remaining life, right-censoring of current spell | ✅ Standard parametric survival (Kalbfleisch & Prentice 2002). Right-censoring is correctly handled |
| **Degenerate fit detection** | State collapse (<5%), low-data warning, over-robustness check | ✅ Warn-and-proceed is the right philosophy for production systems |
| **Robust corrections** | Huber IRLS and MinCovDet post-hoc on emission parameters | ✅ Sound — Huber at k=1.345 achieves 95% Gaussian efficiency (Huber 1964). MCD is well-established (Rousseeuw 1984) |

### ⚠️ Areas with soundness concerns

**1. Single fixed random seed (`random_state=42`) — HIGH concern**

```python
# _hmm_engine.py:87
model = hmm.GaussianHMM(..., random_state=42, ...)
```

SOTA practice uses **multiple random restarts** (10-50) and selects the best by log-likelihood. The BIC path does 3 restarts, but the main fitting path uses exactly one seed. EM for HMM is **guaranteed to find only local optima** — a single seed risks poor convergence, especially with:
- High-dimensional feature spaces (~50 features)
- Financial returns (heavy tails, low signal-to-noise)
- Overlapping state distributions

**Impact**: The model may routinely converge to suboptimal solutions. This is the single most impactful concern.

**2. Regime mapping on first feature only — MEDIUM concern**

```python
# _hmm_engine.py:244
state_means = means[:, 0]  # sort by column 0 only
```

States are mapped to bear/sideways/bull based solely on the **first feature dimension's** mean. With ~50 features, column 0 may not be the most informative or may have different meaning across refits (especially with PCA). SOTA would use:
- A **projection onto returns** (ensuring the ordering dimension is financially meaningful)
- Or the **stationary distribution** to identify dominant vs. rare states
- Or a **supervised mapping** using known regime labels as anchor points

**3. No explicit non-stationarity handling — MEDIUM concern**

The expanding window assumes the data-generating process is approximately stationary. Financial time series exhibit:
- **Structural breaks** (regime changes in volatility, not just returns)
- **Concept drift** (the relationship between features and regimes evolves)

SOTA approaches include:
- **Rolling windows** (fixed-width, e.g., 504 bars) instead of expanding
- **Change-point detection** before HMM fitting
- **Online HMM** variants with forgetting factors
- **Time-varying transition matrices** (e.g., Diebold et al. 1994 Markov-switching with time-varying probabilities)

The expanding window means early observations always influence the current fit, which can be stale.

**4. Diagonal covariance assumption — LOW-MEDIUM concern**

```python
covariance_type="diag"
```

This assumes features are **conditionally independent** given the regime. In practice:
- SMA features at different horizons are highly correlated
- Momentum and volatility features co-move

Full covariance (`"full"`) or tied covariance (`"tied"`) would be more flexible, though at the cost of more parameters. The PCA whitening option partially mitigates this by decorrelating features first, but it's optional and off by default.

**5. FSHMM EM convergence guarantees — LOW concern**

The FSHMM implementation has a subtle issue in the M-step:

```python
# fshmm.py:244-248
signal_exp = np.exp(log_signal_upd[1:])  # can overflow/underflow
xi_all = np.einsum("ti,tj,ij->tij", gamma[:-1], signal_exp, A)
```

The transition M-step exponentiates log-likelihoods directly rather than working in log-space throughout. For long sequences or extreme log-likelihoods, this can cause numerical issues. The `1e-300` clamps help but don't fully resolve this. A log-space formulation for ξ (the transition posteriors) would be more robust.

Additionally, the FSHMM rho MAP update formula:

```python
# fshmm.py:234-237
discriminant = np.maximum(T_hat**2 - 4 * k_param * u_sum, 0)
rho_new = (T_hat - np.sqrt(discriminant)) / (2 * k_param)
```

This uses `k_param = 1.0` (Beta(1,1) prior = uniform). The Adams et al. paper uses a Beta(α, β) prior. With k_param=1.0, the MAP reduces to a specific case. This is fine but means the prior provides **no regularization** — rho can go to 0 or 1 freely. A stronger prior (e.g., Beta(2,2)) would provide shrinkage and prevent degenerate saliency weights.

**6. Walk-forward refit cadence — LOW concern**

```python
refit_every = max(1, (n - min_train) // 100)
refit_every = min(refit_every, 20)
```

The cadence is **time-based**, not event-based. SOTA approaches refit when:
- Posterior entropy exceeds a threshold (the model is uncertain)
- The KL divergence between current and last-fit parameters exceeds a threshold
- A change-point test triggers

This is pragmatic but misses opportunities for adaptive refitting.

### ⚠️ Architectural / methodological notes (not bugs, but debatable choices)

**7. The "5% state collapse" threshold is arbitrary**

```python
# pipeline.py:244
min_fraction = 0.05
```

This is a heuristic. There's no theoretical basis for 5%. For some assets, a genuine bear regime may occupy <5% of bars. The warning is correct in flagging this, but the user needs to know it's a rule-of-thumb.

**8. No ensembling across engines**

SOTA financial regime detection increasingly uses **ensemble methods** — combining multiple HMM fits (different seeds, different feature sets) via:
- Majority voting on regime labels
- Averaging posteriors
- Bayesian Model Averaging (BMA)

The current architecture (5 independent engines, no cross-engine aggregation) is clean but leaves performance on the table.

**9. No explicit out-of-sample validation protocol**

The walk-forward backtest provides one form of OOS evaluation, but there's no:
- **Purged cross-validation** (de Prado 2018) to prevent leakage from overlapping regime labels
- **Combinatorial purged cross-validation** for robust performance estimation
- **Bootstrap confidence intervals** on Sharpe and other metrics

The single walk-forward path gives one realization, which may be noisy.

---

## 3. Summary Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Mathematical correctness** | ✅ Sound | EM, forward-backward, BIC, Weibull survival are all correctly implemented |
| **FSHMM (Adams 2016)** | ✅ Faithful | Accurate reproduction of the paper's EM formulation with saliency variables |
| **Robustness layer** | ✅ Sound | Huber IRLS and MCD are textbook-correct post-hoc corrections |
| **Production safety** | ✅ Good | Degenerate fit detection, warn-and-proceed, no hard crashes |
| **Numerical stability** | ⚠️ Adequate | log-space where it matters, but FSHMM transition M-step could be more robust |
| **vs. SOTA (single-fit)** | ⚠️ Gaps | **Single random seed is the biggest concern** — SOTA uses 10-50 restarts minimum |
| **vs. SOTA (methodology)** | ⚠️ Gaps | No non-stationarity handling, no online HMM, no time-varying transitions |
| **vs. SOTA (evaluation)** | ⚠️ Gaps | No purged CV, no bootstrap CI, no ensemble methods |

**Bottom line**: The math is **correct** — the EM algorithm, forward-backward, BIC selection, Weibull survival, and FSHMM saliency EM are all faithful to their source literature. The main soundness gaps are **methodological** rather than implementational: the single-seed fitting, expanding-window stationarity assumption, and first-feature-only regime mapping are the three areas where the implementation departs most from current SOTA practice. The single-seed issue is the highest-impact item — adding multi-restart fitting would likely improve regime quality more than any other change.

---

## 4. Review Session Outcome (2026-06-03)

A design-grilling session was conducted on 2026-06-03 to walk through each
concern and decide on follow-up actions.

### Resolved

| # | Concern | Action | Detail |
|---|---------|--------|--------|
| 2 | Regime mapping on first feature only | **DONE** | Documented in both
``_classify_hmm_slice`` and ``_remap_to_prev_states`` docstrings.
Column 0 is ``log_ret`` (no PCA) or PC1 (with PCA) -- both monotonic
with market direction. |
| 5 | FSHMM EM numerical stability | **SKIP** | Verified numerically:
``np.exp()`` stays within float64 range for all realistic feature counts
(D ≤ 50) given the 1e-8 variance floor. Concern is theoretical only. |
| 7 | 5% state collapse threshold | **DONT_TOUCH** | No evidence of false
triggers. Leave as-is. |

### Deferred

| # | Concern | Trigger / Shelf |
|---|---------|-----------------|
| 1 | Single random seed (random_state=42) | Fix when a backtest shows
suboptimal convergence from single-seed EM. BIC path already uses
multi-restart with diverse seeds. |
| 3 | Non-stationarity (expanding window) | Fix when degradation appears.
Known fix: cap at 756 bars (~3 years). |
| 6 | Walk-forward refit cadence (time-based) | Fix when posterior entropy
spikes reveal missed regime transitions. |
| 8 | No cross-engine ensembling | Tracked in technology scan (Issue #25). |
| 9 | No purged CV / bootstrap CIs | Evaluated and deprioritized in
technology scan. |

### Pending

| # | Concern | Status |
|---|---------|--------|
| 4 | Diagonal covariance / PCA default | PCA being tested for on-by-default
with threshold determined by test results. |

### Stale corrections (review written against older code)

The SOTA review was written before the robust-correction refactor and
several fixes landed. The following concerns are already addressed:

- **Huber IRLS iteration count**: Review claims "4 iterations fixed".
  Current code uses ``max_iter=10`` with ``tol=1e-6`` convergence check
  and early termination. Fixed in ``_hmm_engine.py:_huber_correction()``.
- **MCD minimum-points guard**: Review flagged wrong threshold.
  Current code uses ``max(n_components + 1, n_features + 1, 5)`` --
  the suggested fix. Fixed in ``_hmm_engine.py:_mcd_correction()``.
- **FSHMM plateau early-exit**: Already has 3-iteration flat log-likelihood
  detection. Present in ``fshmm.py`` since implementation.

---

## References

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Hamilton, J.D. (1990). "Analysis of Time Series Subject to Changes in Regime." *Journal of Econometrics*, 45(1-2), 39-70.
- Adams, R.P., Wallach, H.M., & Ghahramani, Z. (2016). "Learning the Structure of Deep Sparse Graphical Models." *AISTATS*.
- McLachlan, G.J. & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Huber, P.J. (1964). "Robust Estimation of a Location Parameter." *Annals of Mathematical Statistics*, 35(1), 73-101.
- Rousseeuw, P.J. (1984). "Least Median of Squares Regression." *JASA*, 79(388), 871-880.
- Kalbfleisch, J.D. & Prentice, R.L. (2002). *The Statistical Analysis of Failure Time Data*. Wiley.
- Diebold, F.X., Lee, J.H., & Weinbach, G.C. (1994). "Regime Switching with Time-Varying Transition Probabilities." In *Nonstationary Time Series Analysis and Cointegration*, Oxford University Press.
- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.
