# BIC-based HMM state count selection

HMM engines require `n_states` to be specified upfront. Hardcoding to 3
assumes the market always has exactly three regimes, which may not hold
across all instruments or time periods. The dead `gaussian_hmm.py` had
`evaluate_model_quality()` that computed BIC/AIC — ADR-001 noted it as
"lost, extract as standalone utility."

## Considered options

### A) Fixed `n_states` (status quo)

Keep `--n-states 3` as the only option. Users who want a different count
must manually specify it.

**Rejected because**: three regimes is an assumption, not a discovery.
Different markets (commodities vs. equities) and timeframes (intraday vs.
weekly) may exhibit 2–6 meaningful regimes. Forcing three states can
over-split a two-regime market or under-split a four-regime one.

### B) Auto-selection via BIC in `_hmm_shared.py` (chosen)

Add `select_n_states()` as a shared utility in `_hmm_shared.py`. It fits
GaussianHMM for each candidate `k` in `[2, max_states]` with multiple
random restarts, computes BIC for each, and returns the `k` with the lowest
BIC. Triggered by `--n-states auto` on the CLI.

**Chosen because**: (1) BIC penalizes model complexity, naturally guarding
against overfitting on short data windows; (2) `select_n_states()` calls
`_fit_hmm_on_slice()` internally, so it automatically gets PCA-consistent
feature counts when PCA is active (ADR-0005); (3) shared utility in
`_hmm_shared.py` respects ADR-0003 self-containment — it's a utility, not
cross-engine coupling; (4) backward-compatible — default remains `--n-states 3`.

## Consequences

- **New function**: `select_n_states(features, max_states, random_state,
  n_restarts, pca_variance)` in `_hmm_shared.py`. Returns `int`.

- **Guard against short data**: `effective_max = min(max_states, max(2,
  n // 10))` prevents fitting 6-state HMMs on 30-bar windows.

- **BIC formula**: `BIC = -2 * log_likelihood + n_params * log(n)` where
  `n_params = k * d_eff + k * d_eff + k * (k - 1)` (means + diag
  covariances + transition probabilities). `d_eff` is the effective
  dimensionality after PCA whitening (if active).

- **CLI change**: `--n-states` now accepts `'auto'` or an integer `>= 2`.
  Default remains `3`. Parsed via `_parse_n_states()` in `cli.py`.

- **Pipeline resolution**: `pipeline.run()` resolves `'auto'` to a concrete
  `int` by calling `select_n_states()` on the precomputed features before
  engine construction. Both HMM engines benefit.

- **Threshold engine unaffected**: `n_states` is ignored by the threshold
  engine. `'auto'` is silently accepted but unused.

- **Multiple restarts**: Default 3 restarts per candidate `k` with different
  seeds to reduce sensitivity to EM local optima.
