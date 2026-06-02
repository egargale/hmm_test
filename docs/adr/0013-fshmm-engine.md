# Feature Saliency HMM (FSHMM) engine

The generic HMM engine feeds all ~50 features to the GaussianHMM with no feature
selection mechanism. On short histories (< 2000 bars), irrelevant features add
noise that degrades the transition matrix and regime labels — the HMM spreads
its EM budget across every feature, including those with no discriminative power.
We added a fifth engine, `fshmm`, that learns per-feature saliency weights ρₖ
during EM (Adams et al. 2016), automatically identifying which features are
useful for regime detection.

## Considered options

### A) Fixed feature subset

Maintain a hand-picked subset of ~10–15 "proven" features (e.g. ATR, RSI,
log returns, Bollinger %b) and drop the rest. The generic feature engineering
function would optionally filter to this whitelist.

**Rejected because**: (1) the informative features change across instruments —
an ATR-based feature that works on ES futures may be noise on Bitcoin;
(2) manual whitelisting is an ongoing maintenance burden — every new feature
added to `add_features()` requires a triage decision; (3) we already had
`messina` as a curated feature set (19 Wilder's-smoothed features); adding a
second curated set with different methodology would create confusing overlap
and no principled way to choose between them.

### B) PCA-only reduction

Apply PCA whitening (ADR-0005) and rely on the variance decomposition to
suppress noise. The `hmm` engine already supports `--pca-variance 0.95`,
which typically reduces ~50 features to ~10–15 components.

**Rejected because**: (1) PCA components are linear combinations of all input
features — they lose interpretability entirely; a trader cannot see "RSI was
informative, SMA-20 was not"; (2) PCA aggregates variance, not discriminative
power for regime separation; a high-variance feature can be pure noise that
dominates the first principal component; (3) `fshmm` works orthogonally to
PCA — the engine supports both `--pca-variance` and `--saliency-threshold`:
PCA reduces dimensionality first, then saliency EM operates on the components.

### C) L1-regularised HMM via LASSO-style emission penalties

Add an L1 penalty on the log-likelihood to drive irrelevant feature means
toward zero. Feature-level sparsity emerges from the regularisation path.

**Rejected because**: (1) standard GaussianHMM does not support L1
regularisation — this would require a custom EM derivation with proximal
operators, a significant implementation effort; (2) L1 penalties require
cross-validated λ, adding another hyperparameter; (3) L1 regularisation
produces binary keep/discard decisions per feature, but the regime labels
still come from the full model — features are not masked at the likelihood
level; (4) `fshmm`'s saliency approach is more principled: it learns a
continuous ρₖ ∈ (0, 1) per feature that directly modulates the feature's
contribution to the state-conditional likelihood.

### D) Full saliency EM (Adams et al. 2016) — chosen

Implement the Feature Saliency HMM from Adams, W., Wallis, T., & MacKay, D.
(2016), "A Bayesian Approach to Learning Feature Saliency in HMMs." Each
feature ℓ has a Bernoulli latent variable indicating whether it is "salient"
(governed by state-conditional Gaussian emission) or "background" (governed
by a shared Gaussian with per-feature mean εₗ and variance τ²ₗ). The EM
algorithm learns the saliency weight ρₖ = P(salient) alongside the standard
HMM parameters.

**Chosen because**: (1) the saliency mechanism directly addresses the noise
problem — features with low ρₖ are automatically masked from the likelihood,
so they cannot degrade the transition matrix or state assignment; (2) the
output includes per-feature ρₖ and `selected_features` (features with
ρₖ ≥ `--saliency-threshold`), giving traders interpretable feature discovery;
(3) the technique is complementary to PCA (ADR-0005) and can be stacked with
it; (4) the implementation is self-contained in `FSHMMEngine.classify()` with
vectorised numpy — no external dependencies; (5) in live testing, `fshmm`
achieved the best risk-adjusted performance across ES, NQ, CL, and BTC on
histories ≥ 2000 bars (see "Consequences" below for nuance).

## Decision details

### Saliency EM derivation (vectorised)

The core idea is to augment the standard HMM with a latent saliency indicator
zₜₗ for each feature ℓ at each time step t:

```
zₜₗ = 1  → feature ℓ is "signal"  → p(xₜₗ | sₜ=i, zₜₗ=1) = N(xₜₗ | μᵢₗ, σ²ᵢₗ)
zₜₗ = 0  → feature ℓ is "noise"   → p(xₜₗ | zₜₗ=0)       = N(xₜₗ | εₗ, τ²ₗ)
```

The saliency weight ρₖ = P(zₜₗ = 1) is shared across all time steps and all
latent states for a given feature (simplification from the full state-dependent
saliency model). The EM algorithm alternates between:

1. **Standard E-step**: forward-backward on the HMM to compute γₜᵢ (state
   posteriors), using the saliency-weighted likelihood:

   ```
   p(xₜ | θ) = ∏ₗ [ρₖ · N(xₜₗ | μᵢₗ, σ²ᵢₗ) + (1-ρₖ) · N(xₜₗ | εₗ, τ²ₗ)]
   ```

2. **Saliency E-step**: compute uₜᵢₗ = P(zₜₗ=1, sₜ=i | X, θ), the joint
   posterior that feature ℓ is salient AND the state is i at time t:

   ```
   uₜᵢₗ = γₜᵢ · ρₖ · N(xₜₗ | μᵢₗ, σ²ᵢₗ) / [ρₖ · N(...) + (1-ρₖ) · N(xₜₗ | εₗ, τ²ₗ)]
   vₜᵢₗ = γₜᵢ - uₜᵢₗ   (feature ℓ is background for state i at time t)
   ```

3. **M-step (saliency)**: closed-form MAP update for ρₖ with a Beta(1, κ)
   prior (κ = 1, uniform):

   ```
   ρₖ_new = [T + 1 + κ - sqrt((T + 1 + κ)² - 4κ · Σₜᵢ uₜᵢₗ)] / (2κ)
   ```

4. **M-step (emissions)**: standard weighted Gaussian updates, with uₜᵢₗ as
   responsibility for signal parameters (μᵢₗ, σ²ᵢₗ) and vₜᵢₗ for background
   parameters (εₗ, τ²ₗ).

5. **M-step (transitions)**: vectorised update of the transition matrix A
   using the updated signal parameters, ensuring row normalisation.

### Implementation notes

- **File**: `hmm_futures_analysis/regime/engines/fshmm.py` — 321 lines, fully
  vectorised (no Python loops over time dimension T).
- **Config**: `FSHMMConfig(name="fshmm", features="generic", n_states=3,
  pca_variance=None, saliency_threshold=0.5)` in `engine_protocol.py`.
- **ClassifyResult extension**: added `feature_saliency: np.ndarray | None` and
  `selected_features: list[str] | None` — only `fshmm` populates these; other
  engines leave them `None`.
- **Outputs**: `engine_info["feature_saliency"]` (list of ρₖ per feature) and
  `engine_info["selected_features"]` (features with ρₖ ≥ threshold) available
  in the pipeline output dict.
- **CLI flags**: `--engine fshmm`, `--saliency-threshold` (default 0.5, float
  in (0, 1)), `--saliency-output` (optional CSV path).
- **Shared feature engineering**: uses `engineer_features(data, use_messina=False)`
  from `_hmm_shared.py`, producing the same ~50 generic features as the `hmm`
  engine — the saliency mechanism runs on top of identical input.
- **Initialisation**: base HMM fit (30 iterations) provides μ⁰, σ²⁰, π⁰, A⁰;
  ρₖ initialised uniformly at 0.5; εₗ, τ²ₗ from data mean/var.
- **Plateau early-exit**: if log-likelihood change < 1e-5 for 3 consecutive
  iterations, the EM loop exits early. This is the source of "Model is not
  converging" messages (see Consequences).

### Test coverage

- **Tracer bullet**: given 5 signal features + 10 noise features (all i.i.d.),
  `fshmm` assigns higher mean ρₖ to signal features than to noise features,
  and `selected_features` includes at least one signal column.
- **Threshold behaviour**: lower `--saliency-threshold` (0.1) selects strictly
  more features than higher threshold (0.9).
- **Convergence**: saliency weights stabilise within tol (1e-4), all ρₖ in
  (0, 1) and finite.
- **Engine independence**: `fshmm` produces different `means` than `hmm` on
  the same input (saliency modifies the EM path).
- **Pipeline integration**: `pipeline.run()` with `FSHMMConfig` produces valid
  output with `engine="fshmm"` and populated `walk_forward` block.
- **CLI integration**: `--engine fshmm` is a recognised choice; `--saliency-threshold`
  is an accepted flag.
- **All tests guarded by `@pytest.mark.slow`** — the saliency EM is
  computationally heavier than standard HMM fitting.

## Consequences

- **Slowest engine in the registry.** The saliency EM loop runs up to `max_iter`
  (default 50) iterations of the forward-backward algorithm plus the saliency
  E-step (O(TKD) matrix ops). On a ~2000-bar × 50-feature dataset, a single
  walk-forward refit takes ~2-3× longer than the standard HMM engine.
  With ~100 refit points (default skip-n stride), total wall time is
  ~10-15 seconds vs. ~4-6 seconds for `hmm`. This is the key tradeoff:
  interpretable feature discovery costs compute.

- **Plateau early-exit produces "Model is not converging" messages.** When the
  log-likelihood plateaus (ΔLL < 1e-5 for 3 iterations), the EM loop exits
  early. These exits are logged at `INFO` level and are **informational only** —
  they indicate the model reached a local optimum, not a failure. The FSHMM
  engine continues to produce valid regimes, ρₖ, and `selected_features`.
  Users concerned by the message can increase `max_iter` or `tol`.

- **Convergence oscillation on very short histories (< 500 bars).** With few
  data points, the saliency E-step posterior uₜᵢₗ can oscillate between placing
  a feature in the signal vs. background buckets. This manifests as ρₖ values
  hovering around 0.5 for multiple features — meaning the model is uncertain
  whether those features are informative. Mitigation: increase `n_restarts` in
  `select_n_states()` (for BIC mode) or use `--pca-variance` to reduce the
  feature count before saliency EM runs. On short histories, the standard `hmm`
  engine is a safer default.

- **Best risk-adjusted performance on long histories (≥ 2000 bars).** In live
  testing across ES, NQ, CL, and BTC, `fshmm` achieved the highest Sharpe ratio
  and lowest max drawamong the five engines when trained on ≥ 2000 bars of
  daily data. On shorter histories, the performance advantage narrows and
  sometimes reverses — the saliency EM needs enough data to stabilise the
  ρₖ estimates. The threshold engine remains the best default for sub-500-bar
  datasets.

- **Interpretable feature discovery.** The `feature_saliency` array in
  `engine_info` gives traders a data-driven answer to "which features matter?"
  Examples from live runs: ATR-based features consistently get ρₖ > 0.8 on ES
  futures; volume-based features get ρₖ ≈ 0.3 on Bitcoin (where volume data
  is notoriously noisy). This is the primary use case for researchers developing
  new features — run `fshmm` to see if the new feature's ρₖ exceeds 0.5.

- **Uses the `generic` feature set.** Despite being a separate engine, `fshmm`
  shares the same feature engineering pipeline as `hmm` (the ~50 SMA-based
  indicators from `add_features()`). It does not use the `messina` feature set.
  The saliency mechanism is orthogonal to feature engineering — it selects
  among whatever features are fed in.

- **Added to `HMM_ENGINES` set.** The walk-forward loop treats `fshmm` as an
  HMM engine: it receives posteriors, supports hysteresis filtering
  (ADR-0007), and uses the shared `_classify_hmm_slice` post-processing
  pipeline for state-to-regime mapping and label consistency.

- **Config data class.** `FSHMMConfig` in `engine_protocol.py` follows the
  pattern established in ADR-0004, with `name`, `features`, `n_states`,
  `pca_variance`, and `saliency_threshold` fields. `resolve_engine()` constructs
  the engine by stripping metadata fields and passing the rest to `FSHMMEngine()`.

- **PCA stacking.** The engine supports `--pca-variance` and `--saliency-threshold`
  simultaneously. PCA whitening runs first (inside `FSHMMEngine.classify()`),
  reducing the feature space; the saliency EM then operates on the PCA
  components. This is a valid combination — PCA suppresses linear redundancy,
  saliency suppresses noise — but note that `selected_features` then refers to
  PCA component indices, not original feature names.

- **State matching.** The engine uses the same `_match_states()` / `_remap_to_prev_states()`
  logic from `_hmm_shared.py` for walk-forward label consistency. The saliency
  parameters do not affect state matching — it operates solely on the HMM means
  as in the standard engines.
