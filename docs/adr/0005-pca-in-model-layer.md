# PCA placement in model layer

The generic HMM engine feeds ~50 features to GaussianHMM with only z-score
normalization. With `diag` covariance, this means ~50 independent variance
parameters per state — overparameterized for typical datasets (252–2000 bars).
PCA whitening can reduce dimensionality, but the question is where in the
pipeline it should live.

## Considered options

### A) Feature layer — inside `engineer_features()` or `precompute()`

PCA would run once on the full dataset in `precompute()`, producing a
dimensionality-reduced feature set that the engine then fits on normally.

**Rejected because**: `precompute()` sees the entire dataset. Running PCA
there introduces lookahead bias — the principal components would be computed
from future data not yet available at each expanding-window slice. This
violates the project's hard constraint on bias-free walk-forward analysis
(see CONTEXT.md: "all indicators are backward-looking, no lookahead bias").

The alternative — running PCA per-slice inside `classify()` — would duplicate
the normalization + reduction pipeline that `_fit_hmm_on_slice()` already
owns, and would require restructuring the precompute/classify split.

### B) Model layer — inside `_fit_hmm_on_slice()` (chosen)

PCA runs **after** z-score normalization, **before** `model.fit()`, within
the per-slice fitting function. Naturally respects the expanding-window
cadence — no lookahead bias possible because `_fit_hmm_on_slice()` is called
per-slice with only data up to that point.

**Chosen because**: (1) the function already owns z-score normalization, so
PCA is the natural next step in the same pipeline: raw features → z-score →
PCA → fit; (2) no lookahead bias — PCA is fit on the same slice the model
sees; (3) `select_n_states()` calls `_fit_hmm_on_slice()` internally, so BIC
comparison automatically gets PCA-consistent feature counts with no extra
wiring; (4) `_match_states()` is space-agnostic — nearest-neighbor matching
works in any consistent-dimensional space.

## Consequences

- **New parameter on `_fit_hmm_on_slice()`**: `pca_variance: float | None`.
  When set (e.g. 0.95), PCA whitening is applied after z-score normalization.
  When `None` (default), no PCA — existing behavior preserved.

- **Sticky component count**: The first refit determines the PCA component
  count from the variance threshold. Subsequent refits use that fixed count
  so that `prev_means` and `new_means` have consistent dimensionality for
  `_match_states()`. The count is stored as engine instance state
  (`_pca_n_components`).

- **Return signature change**: `_fit_hmm_on_slice()` returns a 4-tuple
  `(model, center, scale, pca_n_components_used)` instead of a 3-tuple. All
  internal callers (`select_n_states`, both engine `classify()` methods,
  `hmm_adapter`) need updating. This is a private function — no public API
  change.

- **Both HMM engines opt in independently**: `pca_variance` is available on
  both `HMMGenericEngine` and `HMMMMessinaEngine`, defaulting to `None` on
  both. Per ADR-0003, each engine is self-contained. The Messina engine (19
  features) doesn't need PCA, but the param is available for experimentation.

- **No CLI flag for now**: `pca_variance` is constructor-only. It threads
  through `pipeline.run()` and `walk_forward_backtest()` as a keyword
  argument. CLI exposure is deferred to a follow-up issue if needed.

- **Dual engine instances preserved**: The pipeline and walk-forward backtest
  construct independent engine instances. Both independently determine their
  PCA component count on first refit. Since both start from the same
  `min_train` slice, they converge on the same count. No coupling introduced.
