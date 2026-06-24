# ADR-0025: Use HMM internal transition matrix for forecasts; drop multi-step forecasts

**Status**: Accepted
**Date**: 2026-06-24

## Context

The regime detection pipeline has five engines that produce regime labels.
Historically, ALL engines fed their classified regime sequence into
`build_transition_matrix()` — an empirical count-based matrix — and
forecasts were computed by raising that matrix to powers (1, 5, 20 steps).
This meant:

1. The HMM's learned internal transition matrix (`model.transmat_`) was
   computed during EM fitting but **discarded** — the EM algorithm expends
   compute learning transition dynamics from the full feature space, but
   the pipeline ignored those learned probabilities in favor of counting
   regime-label transitions.

2. The empirical count matrix only captures 1D regime-sequence dynamics.
   The HMM's internal transmat captures transition probabilities informed
   by the full multi-dimensional feature space — a richer signal.

3. Multi-step forecasts (5-step, 20-step) compound error. A transition
   matrix raised to the 20th power converges to the stationary distribution
   regardless of quality, making 20-step forecasts indistinguishable
   across engines.

## Considered options

### A) Keep empirical count matrix (status quo)

**Rejected because**: (1) the HMM EM algorithm already computes a learned
transition matrix as part of model fitting — discarding it wastes compute
and information; (2) the empirical count matrix is equally available from
the output regime sequence for post-hoc analysis; (3) for a forecasting
use case, the generative model's own transition probabilities are the
theoretically correct source.

### B) Use HMM internal transmat, keep multi-step forecasts

**Rejected because**: (1) multi-step forecasts (5, 20) converge rapidly to
the stationary distribution — a 20-step forecast from any reasonable 3×3
stochastic matrix is within 0.03 of the stationary distribution, regardless
of engine quality; (2) maintaining multi-step forecasts creates a misleading
impression of forecasting horizon precision; (3) the `--forecast-horizon`
flag was never exposed to users and remained an internal implementation
detail.

### C) Use HMM internal transmat, 1-step forecast only — chosen

For HMM-family engines (hmm, messina, robust_hmm, fshmm), the transition
matrix in pipeline output is now the HMM's internal `model.transmat_`
remapped from HMM latent state space to the 3-regime space (Bear=0,
Sideways=1, Bull=2). The threshold engine (which has no HMM) continues
to use the empirical count matrix.

Forecasts are reduced to 1-step only. The `forecast` output dict now
contains a single key: `{"1_step": {"bear": ..., "sideways": ..., "bull": ...}}`.

## Decision details

### Transmat aggregation

The HMM's internal transmat is K×K (K = number of latent states, typically
3, but 2 in degenerate recovery or N for BIC-selected N > 3). It is
aggregated to a 3×3 regime-space matrix by summing:

```
regime_transmat[r_from, r_to] += model.transmat_[i, j]
  where label_map[i] = r_from, label_map[j] = r_to
```

Rows are normalized to sum to 1.0. For regimes with no mapped HMM states
(e.g. Sideways in a 2-state model), the row is filled uniformly [1/3, 1/3, 1/3].

### Data flow

```
HMMEngine.classify()
  → _classify_hmm_slice(model, ...)
    → aggregate model.transmat_ → regime_transmat (3×3)
    → ClassifyResult(transmat=regime_transmat)

_hmm_classify_pipeline()
  → walk-forward loop captures last refit's transmat
  → ClassifyOutput(transmat=last_transmat)

pipeline.run()
  → classify_out.transmat passed to _build_markov_stats()
  → if not None: use HMM transmat
  → if None (threshold): fall back to build_transition_matrix(regimes)

_result_assembly._assemble_result()
  → forecast_n_steps(markov.transmat, current_probs, 1)
  → _compute_verdict(forecast_1)  # uses 1-step, not 20-step
```

### Backward compatibility

- `ClassifyResult` gains `transmat: np.ndarray | None = None` — default None
  preserves backward compatibility with existing code that constructs
  ClassifyResult manually (e.g. threshold engine, tests).
- `ClassifyOutput` gains `transmat: np.ndarray | None = None` — default None,
  same reasoning.
- The `forecast` dict changes from `{1_step, 5_step, 20_step}` to `{1_step}`.
  Downstream consumers that access `forecast["5_step"]` or `forecast["20_step"]`
  will get a `KeyError` — an intentional break to signal the reduced surface.
- Test fixtures, mock data, and contract tests updated to reflect 1-step only.

## Consequences

- **HMM engines now surface their learned transition dynamics.** The forecast
  for HMM engines is no longer degenerate even when the engine produces few
  regime changes (1-2 trades). The HMM's internal transmat captures transition
  probabilities that the empirical count matrix cannot — even a single-regime
  walk-forward run has a meaningful learned transmat from the feature space.

- **Threshold engine unchanged.** It has no HMM, no internal transmat.
  Continues using empirical count matrix from `classify_regimes()`.

- **Verdict now uses 1-step forecast.** The `_compute_verdict` parameter
  renamed from `forecast_20` to `forecast_1`. Verdict logic unchanged —
  it always compared bull vs bear probabilities, just now at 1-step horizon
  instead of 20-step.

- **Presenter updated.** Terminal output shows "FORECAST (1-step)" instead
  of "FORECAST (n-step)" with sub-keys. One flat block per regime.

- **Future portability improved.** A forecast project can now extract the
  HMM transmat directly without needing to re-classify the regime sequence
  or rebuild an empirical matrix. The engine's `ClassifyResult.transmat`
  is a self-contained 3×3 forecast-ready matrix.

- **Code touched:** `engine_protocol.py`, `_hmm_engine.py`, `_hmm_pipeline.py`,
  `pipeline.py`, `_result_assembly.py`, `presenter.py`, `CONTEXT.md`,
  `SKILL.md`, `README.md`. Tests: `test_regime_contract.py`,
  `test_regime_pipeline.py`, `test_verdict.py`, `test_cli_cache_flags.py`.
