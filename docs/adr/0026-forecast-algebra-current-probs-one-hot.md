# ADR-0026: Forecast algebra — `current_probs` is the one-hot `e_current`

**Status**: Accepted
**Date**: 2026-11-24
**Issue**: #112 (Slice 1) / parent #111

## Context

`_build_markov_stats()` in `regime/pipeline.py` populated
`MarkovStats.current_probs` with `transmat[last_regime]` — a **row of the
transition matrix `T`** — and `forecast_n_steps()` computes
`current_probs @ T^n`. The "1-step" forecast therefore returned
`T[i, :] @ T = T²[i, :]`: a two-step distribution masquerading as a
one-step forecast, while `next_state_probabilities` (also sourced from the
row) returned `T[i, :]` — so the two JSON fields that the SKILL.md output
contract says must be equal silently disagreed (up to Δ=0.18 on the
threshold panel, see issue #112). On the HMM engines the defect was masked:
their learned matrices are so close to identity (persistence ≈ 1) that
`T² ≈ T`.

## Decision

`MarkovStats.current_probs` holds the **one-hot unit vector of the current
regime**, `e_current = np.eye(transmat.shape[0])[last_regime]`:

1. `forecast.1_step = forecast_n_steps(T, e_current, 1) = e_current @ T =
   T[current_regime, :]` — equal to `next_state_probabilities`, which the
   assembly layer now sources from the same computed vector.
2. `signal` is computed from the transition row
   `T[current_regime, :]` (`P(bull) − P(bear)`), independent of
   `current_probs`. Signal is unchanged — it was already the transition
   row's `[2]-[0]` difference when `current_probs` was row-valued. The
   verdict rule (`_compute_verdict`: bull − bear of the 1-step forecast,
   dynamic neutral band) is also unchanged, but its input is now the true
   one-step distribution: before this fix it consumed the `T²[i, :]` row,
   so verdict values can change and the verdict is now genuinely 1-step
   (issue #112 lists `verdict`/`duration_forecast` among the fields
   expected to regenerate differently).
3. The matrix is always 3×3 regime-space: HMM engines aggregate
   `model.transmat_` into a fixed 3×3 whatever the fitted state count
   ([[ADR-0025]]), and the threshold path builds it with the default
   `n_states=3`. `np.eye` is sized from `transmat.shape[0]` instead of
   hardcoded, but `compute_persistence_diagonal()`, `compute_signal()` and
   `_probs_to_dict()` keep their explicit bear/sideways/bull indexing — even
   degenerate auto-recovery ([[ADR-0018]]) maps 2 latent states onto the
   Bear(0)/Bull(2) buckets of the same 3×3 matrix, so no 2×2 ever reaches
   this code.
4. The n-step algebra generalizes: forecast at horizon n is
   `e_current @ Tⁿ` (row-convention; for n=1 exactly `T[current_regime, :]`),
   as pinned by `test_forecast_horizons_match_matrix_powers`.

## Consequences

- `forecast.1_step` values change in JSON output (the fix); the threshold
  panel shows changes up to Δ=0.183 (SPY). `next_state_probabilities`,
  `transition_matrix`, `signal`, `stationary_distribution`,
  `persistence_diagonal`, `regime_counts` and `walk_forward` are unchanged.
  `verdict` and `duration_forecast` values that consume the forecast follow
  the fix — the verdict is now genuinely 1-step (issue #112 acceptance
  criteria expect these fields to regenerate differently; the PR notes the
  defect "can change an action").
- Field names, dict keys, and the rest of `MarkovStats`' public surface are
  untouched; `_result_assembly` and `pipeline._build_markov_stats` remain
  the only touchpoints.
- Tests in `tests/unit/test_build_markov_stats.py` that codified the old
  row-valued field were renamed to assert the one-hot invariant;
  regression coverage lives in `tests/unit/test_forecast_algebra.py` and
  `tests/unit/test_result_assembly.py::TestForecastAlgebraIssue112`.

## Related

- [[ADR-0011]] — engine dispatch consolidation (the seam where
  `current_probs` and `signal` are produced).
- [[ADR-0018]] — degenerate-fit detection and recovery; its 2-state mapping
  still emits the 3×3 regime-space matrix, so nothing here is shape-generic.
- [[ADR-0025]] — HMM internal transmat for forecasts; 1-step forecast
  surface (defines the JSON `forecast` block this invariant governs).
