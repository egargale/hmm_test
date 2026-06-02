# Robust HMM engine — outlier-resistant emission estimation

The standard GaussianHMM underlying the `hmm` engine fits emission parameters
via maximum-likelihood EM, which is vulnerable to outlier bars — a single
flash-crash bar, earnings outlier, or data glitch can distort the per-state
covariance estimate, causing the model to misclassify the current regime or
degenerate into a single-state fit. This problem is acute on instruments with
fat-tailed return distributions: crypto pairs (BTC, ETH), small-cap equities,
and event-driven stocks around earnings.

We added a fourth HMM engine, `robust_hmm`, that applies outlier-resistant
correction to the emission parameters after the standard EM fit converges.
The engine offers two methods: **Huber IRLS** (iteratively reweighted least
squares on means and variances) and **MCD** (Minimum Covariance Determinant
replacement of the emission covariance matrix).

## Considered options

### A) No robust correction — rely on standard GaussianHMM alone

Use the existing `hmm` engine as-is for all instruments. Outlier bars are
treated as legitimate data points — the EM algorithm assigns them to the
state whose emission distribution best accommodates the extreme value.

**Rejected because**: (1) a single outlier bar can inflate a state's variance
to the point where the state becomes a catch-all "extreme events" bucket,
losing discriminative power for that state's actual regime; (2) on short
histories (< 500 bars), even one outlier can shift the transition matrix
enough to flip the dominant state — the model becomes non-reproducible across
refits with different outlier placement; (3) the problem is well-documented
in financial HMM literature (Mateos et al. 2020, Rydén et al. 1998) with no
reason to expect our implementation to be exempt.

### B) Pre-filter outliers from input data before fitting

Apply a pre-processing step that detects and removes or clips outlier bars
before feature engineering or HMM fitting. Candidates included: z-score
thresholding, MAD-based outlier detection, and Tukey fences on raw OHLCV.

**Rejected because**: (1) "outlier" in raw price space is poorly defined —
a 3-sigma move in a calm period is normal in a volatile period; static
thresholds introduce regime-dependent bias; (2) pre-filtering introduces
lookahead bias when implemented naively — removing a bar because it "looks
like an outlier" uses the full-sample distribution, which leaks future
information; (3) raw-price outliers are not necessarily feature outliers —
a flash-crash bar may produce normal ATR-based features if volatility is
already elevated; (4) options A and C are mutually exclusive with pre-filtering
in the pipeline — if both a threshold engine and robust HMM are used on the
same data, the engineer must decide whether pre-filtering runs once for all
engines or per-engine, adding configuration surface with no clear benefit.

### C) Post-hoc robust correction of emission parameters — chosen

After standard EM converges, replace the emission means and covariances with
estimates that are resistant to outliers. Two sub-options were evaluated:

#### C1) Huber IRLS correction

Apply iteratively reweighted least squares to each state's emission parameters
independently. The Mahalanobis distance of each observation from the state
mean determines an observation weight via the Huber loss function (ψ with
tuning constant `k = 1.345`, giving ~95% asymptotic efficiency under Gaussian
noise). Observations beyond `k` receive reduced weight proportional to
`k / distance`, shrinking their influence on the re-estimated mean and
variance.

**Implemented in `_hmm_shared._huber_correction()`** — a concise 35-line loop
that fits into the existing `_hmm_shared.py` module.

#### C2) Minimum Covariance Determinant (MCD) correction

For each state, select observations with posterior probability > 0.3, then
fit a MinCovDet estimator (Rousseeuw & Van Driessen, 1999) on that subset.
The MCD finds the subset of size `h ≈ n_samples / 2` whose covariance has
the smallest determinant — the subset most likely to be free of outliers —
and uses its location and covariance as the robust estimates.

**Implemented in `_mcd_correction()`** — wraps scikit-learn's `MinCovDet`.
An upper bound of 200 points is sampled when the state membership exceeds
this threshold, both to control compute cost and because MinCovDet's
breakdown point degrades with excessive sample size.

**Chosen because**: (1) the correction is opt-in via `--robust-method` — the
existing `hmm` engine is unchanged; (2) both methods operate post-EM on the
converged model, so the standard EM warm-start, BIC state selection, and
walk-forward loop are unaffected; (3) two methods allow users to trade off
speed (Huber) vs. aggressiveness (MCD) per instrument; (4) the API surface
is minimal — one extra CLI flag, one config field; (5) the correction is in
`_hmm_shared.py`, shared by the `robust_hmm` engine but not forced on the
other HMM engines.

A default `k = 1.345` for Huber gives the standard ~95% asymptotic efficiency
under Gaussian noise — meaning on clean data the correction is nearly a no-op
(the IRLS converges in 1–2 iterations with weights near 1.0).

### D) Apply robust correction to all HMM engines automatically

Rather than a separate engine, modify `_fit_hmm_on_slice()` in `_hmm_shared.py`
to always apply Huber or MCD after fitting, gated by a new `robust_method`
parameter on every HMM engine config.

**Rejected because**: (1) the standard `hmm` and `messina` engines are
deliberately simple reference implementations — coupling them to robust
correction would make it impossible to compare "pure" HMM vs. robust HMM
on the same features; (2) the correction would become mandatory rather than
opt-in, increasing cognitive load for users who do not need it; (3) the
`_fit_hmm_on_slice` function's contract — "return a fitted GaussianHMM" —
would be violated if it returned a model whose emission parameters do not
match the EM solution; (4) separate engines are the pattern established by
ADR-0001 and ADR-0009 — each engine is a self-contained unit with its own
config and CLI flags.

## Decision details

### Architecture

The `robust_hmm` engine is a thin wrapper around the shared fitting pipeline:

```
                    ┌─────────────────────┐
                    │  engineer_features   │
                    │  (use_messina=False) │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  robust_fit_gaussian │
                    │  _hmm                │
                    │                     │
                    │  1. _fit_hmm_on_    │
                    │     slice()         │
                    │     (standard EM)   │
                    │  2. Huber/MCD       │
                    │     correction      │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  _classify_hmm_     │
                    │  slice()            │
                    │  (state→regime map) │
                    └─────────────────────┘
```

The function `robust_fit_gaussian_hmm()` returns the same shape as
`_fit_hmm_on_slice()` — `(model, center, scale, pca_n, pca_transform)` —
making it a drop-in replacement in the walk-forward loop. The corrected model
still satisfies the GaussianHMM interface (`.means_`, `.covars_`,
`.predict_proba()`) required by `_classify_hmm_slice()` and the hysteresis
filter (ADR-0007).

### File-level changes (original commit `ad866d1`)

- **`engines/robust_hmm.py`** (new, 282 lines) — `RobustHMMEngine` class
  implementing `RegimeEngineProtocol` with `precompute()`, `classify()`, and
  `enrich_info()`. Uses `engineer_features(data, use_messina=False)` from
  `_hmm_shared.py` — same ~50 generic features as the `hmm` engine.
- **`_hmm_shared.py`** (modified) — added `robust_fit_gaussian_hmm()`,
  `_huber_correction()`, and `_mcd_correction()`. The corrections operate
  directly on the fitted model's `.means_` and `.covars_` attributes,
  mutating them in-place.
- **`engine_protocol.py`** (modified) — added `RobustHMMConfig` dataclass
  and registered `"robust_hmm"` in the `ENGINE_REGISTRY` and `HMM_ENGINES`
  set.

### Config and CLI

```python
@dataclass
class RobustHMMConfig:
    name: str = "robust_hmm"
    features: str = "generic"
    n_states: int | str = 3
    pca_variance: float | None = None
    robust_method: str = "huber"          # "huber" | "mcd"
```

- CLI flag: `--robust-method {huber,mcd}` (default `huber`)
- `--n-states auto` (BIC-based, ADR-0006) is supported — `select_n_states()`
  uses the same `_fit_hmm_on_slice()` call; the robust correction is applied
  after the optimal `k` is selected.
- `--pca-variance` is supported (ADR-0005) — PCA whitening runs inside
  `_fit_hmm_on_slice()` before the robust correction sees the transformed
  features.
- The engine is registered in `HMM_ENGINES` so the walk-forward loop provides
  posteriors, hysteresis filtering (ADR-0007), and state matching.

### Test coverage

- `tests/test_robust_hmm.py` (new, 191 lines) — tracer bullet tests:
  - Given synthetic data with injected outliers, robust correction produces
    different means/covars than standard HMM
  - Huber correction reduces the influence of outlier bars on the state mean
  - MCD correction is more aggressive than Huber in reducing outlier influence
  - `robust_hmm` with `robust_method="huber"` and `robust_method="mcd"` both
    produce valid `ClassifyResult` with `engine="robust_hmm"`
  - Pipeline integration: `pipeline.run()` with `RobustHMMConfig` produces
    valid walk-forward output
  - CLI integration: `--engine robust_hmm` is a recognised choice;
    `--robust-method` is accepted
  - Error case: insufficient data raises `ValueError`

### Relationship to the `hmm` engine

The `robust_hmm` engine is structurally identical to `hmm` except for the
robust correction step:

| Aspect | `hmm` | `robust_hmm` |
|--------|-------|-------------|
| Feature set | Generic (~50) | Generic (~50) |
| Feature function | `engineer_features(use_messina=False)` | Same |
| HMM fitting | `_fit_hmm_on_slice()` | `robust_fit_gaussian_hmm()` |
| Post-processing | `_classify_hmm_slice()` | Same |
| State matching | `_match_states()` / `_remap_to_prev_states()` | Same |
| Hysteresis support | Yes (ADR-0007) | Same |
| BIC support | Yes (ADR-0006) | Same |
| PCA support | Yes (ADR-0005) | Same |
| Walk-forward loop | Full posteriors | Same |

The only difference is that after EM converges, the emission parameters are
re-estimated with an outlier-resistant estimator. Everything else — feature
engineering, state-to-regime mapping, walk-forward label consistency, config
schema — is shared code.

This also means the `robust_hmm` engine inherits the same overfitting risk
on short histories as the `hmm` engine (documented in ADR-0013). The robust
correction mitigates *outlier* distortion but does not address the
feature-count-to-observation ratio problem.

## Consequences

- **~1.2–1.5× compute overhead** for the Huber method vs. standard `hmm` on
  typical datasets (~2000 bars × 50 features). The IRLS loop runs up to 10
  iterations per state, each computing Mahalanobis distances and weight
  updates — negligible on per-slice time, but additive over ~100 walk-forward
  refit points. The MCD method is heavier (MinCovDet is O(n · p²) and runs
  on the selected state subset), approximately 2–3× the `hmm` baseline.

- **Over-robustness on clean data.** The Huber correction with `k = 1.345`
  asymptotically discards ~5% of observations as outliers under a Gaussian
  null — meaning on genuinely well-behaved data with no outliers, it will
  still slightly down-weight a small fraction of valid observations. This is
  a theoretical concern; in practice the weight of inlier observations is
  near 1.0 and the effect on the re-estimated mean is negligible. Users who
  want the unmodified EM solution should use the `hmm` engine.

- **MCD requires sufficient in-state observations.** MinCovDet requires at
  least `n_features + 1` observations per state. When a state's posterior
  membership is small, the `_mcd_correction` skips that state entirely
  (leaving the standard EM estimate in place). This means the MCD method
  may be a partial no-op on short histories or when one regime is rare,
  silently falling back to uncorrected estimates for the sparse states.

- **CLI surface.** The `--robust-method` flag adds one more CLI option to
  an already-large argument surface. Users must remember that this flag is
  only meaningful with `--engine robust_hmm` — it is silently ignored by
  other engines (the `_hmm_shared` correction functions are not called).

- **Added alongside wasserstein in the same commit.** The `robust_hmm` and
  `wasserstein` engines were implemented in one change (commit `ad866d1`),
  sharing a common extension of `_hmm_shared.py`. They are functionally
  independent — the shared module addition was a refactoring of common HMM
  utilities (`_fit_hmm_on_slice`, `_match_states`, `select_n_states`) that
  both new engines depend on.

- **Not yet tested on crypto data in production.** The Huber and MCD methods
  were validated on synthetic outlier-injected data and a limited set of
  equity futures (ES, NQ). The MCD method in particular may benefit from
  instrument-specific tuning of the posterior threshold (currently hard-coded
  at 0.3 in `_mcd_correction`) and the max-point cap (200). These parameters
  are candidates for config-driven customisation in a future iteration.

- **State matching is unaffected.** The robust correction mutates the emission
  means after EM, but the means are still sorted by mean return and matched
  to previous walk-forward states via `_match_states()` from `_hmm_shared.py`.
  The correction changes the mean values but preserves their ordering — the
  state-to-regime mapping remains consistent across refits.

- **Added to `HMM_ENGINES` set.** The walk-forward loop treats `robust_hmm`
  as an HMM engine: it receives posteriors, supports hysteresis filtering
  (ADR-0007), and uses the shared `_classify_hmm_slice` post-processing
  pipeline for state-to-regime mapping and label consistency.

- **Config data class.** `RobustHMMConfig` in `engine_protocol.py` follows
  the pattern established in ADR-0011, with `name`, `features`, `n_states`,
  `pca_variance`, and `robust_method` fields. `resolve_engine()` constructs
  the engine by stripping metadata fields and passing the rest to
  `RobustHMMEngine()`.

- **BIC state count compatibility.** The `select_n_states()` function (ADR-0006)
  uses `_fit_hmm_on_slice()` internally, which does **not** apply robust
  correction. The optimal state count is selected on standard EM fits; the
  robust correction is applied when `RobustHMMEngine.classify()` calls
  `robust_fit_gaussian_hmm()`. This means the BIC-selected `k` reflects
  standard HMM log-likelihood, not the robust-corrected model's — a subtle
  mismatch that is acceptable because the BIC is used only to pick `k`, not
  to evaluate the final model's fit quality.
