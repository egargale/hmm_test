# Push per-engine execution model into `run_classify()` on the engine protocol

**Status**: Accepted
**Date**: 2026-06-02
**Priority**: Strong
**Supersedes**: ADR-0012 (partial — `_classify_threshold_pipeline` deleted, `_hmm_classify_pipeline` becomes engine-internal)

## Context

After ADR-0009 deepened the engine seam with `RegimeEngine` (`precompute` + `classify`), and ADR-0011 consolidated dispatch into config dataclasses, `pipeline.run()` still owns the engine execution model through a `config.is_hmm` flag:

1. **Classify phase** branches on `is_hmm` to route between `_hmm_classify_pipeline` (walk-forward refit loop for 4 HMM-family engines) and `_classify_threshold_pipeline` (one-shot vectorized call for threshold). The threshold wrapper is a 5-line pass-through that exists only for symmetry.

2. **Walk-forward backtest** branches on `is_hmm` to decide whether to pass pre-computed regimes/posteriors (HMM: replay arrays) or not (threshold: re-classifies per bar). This makes threshold O(n²) — `classify_regimes` is called per bar, each call computing a rolling sum over the entire slice, despite the pipeline already having the full regimes array from the classify phase.

3. **`_walk_forward_classify` has 3 modes** (regimes replay / precomputed refit / per-bar classify). Mode 3 (per-bar classify) is only used by the threshold walk-forward path.

The pipeline knows an engine taxonomy that should live behind the engine seam. Adding a sixth engine type still requires touching the pipeline's classify-phase branching, even though ADR-0011 eliminated engine-name branches everywhere else.

## Considered options

### A) Keep the branch, fix the O(n²) separately

Pass `classify_out.regimes` to `walk_forward_backtest` for all engines (unconditional). This fixes the performance bug without touching the classify-phase branch.

**Rejected because**: it leaves `is_hmm` in the classify phase. The pass-through `_classify_threshold_pipeline` stays alive. Two `is_hmm` branches become one. The execution model still leaks across the seam.

### B) Push `run_classify()` into the engine protocol (chosen)

Add `run_classify(self, prices, ohlcv, returns, min_train, **kwargs) -> ClassifyOutput` to `RegimeEngine`. Each engine owns its execution model:

- **Threshold**: calls `classify_regimes` once, wraps in `ClassifyOutput`. (The current `_classify_threshold_pipeline` body moves into `ThresholdEngine.run_classify`.)
- **HMM-family** (hmm, messina, robust_hmm, fshmm): one-liner delegating to `_hmm_classify_pipeline(self, ...)`.

Pipeline calls `eng.run_classify(...)` unconditionally. The `is_hmm` flag disappears from pipeline. Regimes are always passed to walk-forward, fixing O(n²).

**Chosen because**: (1) completes ADR-0003 — each engine now owns its *execution shape*, not just its output block; (2) deletes the pass-through wrapper; (3) fixes the O(n²) threshold bug as a natural consequence; (4) the `**kwargs` pass-through keeps profiling hooks as pipeline-internal plumbing without widening the engine interface.

### C) Merge with engine collapse (ADR candidate 1 — single parameterised `HMMEngine`)

Collapse the 4 HMM engines into one class with constructor-injected `FitStrategy`. `run_classify` lives on the single HMMEngine.

**Rejected because**: it's a bigger refactor that touches every engine file. Option B achieves the same seam at lower risk. Engine collapse can follow as a separate pass without re-litigating this decision.

## Consequences

### Protocol change

`RegimeEngine` gains one method:

```python
def run_classify(self, prices: pd.Series, ohlcv: pd.DataFrame | None,
                 returns: pd.Series, min_train: int,
                 **kwargs) -> ClassifyOutput: ...
```

`precompute()` and `classify()` remain on the protocol — they are called internally by `run_classify` and are used by existing unit tests. Removal is deferred.

### `_classify_threshold_pipeline` deleted

The 5-line wrapper in `pipeline.py` is deleted. Its body moves into `ThresholdEngine.run_classify()`.

### `is_hmm` removed from pipeline

The classify-phase `if getattr(config, "is_hmm", False)` branch is replaced by `classify_out = eng.run_classify(...)`. The `is_hmm` property remains on config dataclasses (used by `HMM_ENGINES` set for downstream queries) but pipeline no longer reads it for dispatch.

### Threshold O(n²) → O(n)

After the change, pipeline always has `classify_out.regimes` and always passes them to `walk_forward_backtest`. The walk-forward replay path (mode 1) is used for all engines. Threshold no longer re-classifies per bar.

### `walk_forward_backtest` signature changes

```python
# Before
def walk_forward_backtest(prices, *, engine, min_train=252, dwell_bars=0,
                          hysteresis_delta=0.0, regimes=None, posteriors=None) -> dict:

# After
def walk_forward_backtest(prices, *, regimes, min_train=252, dwell_bars=0,
                          hysteresis_delta=0.0, posteriors=None) -> dict:
```

`engine` param removed (unused after regimes are always pre-computed). `regimes` goes from optional to required. The function now has one responsibility: apply dwell/hysteresis filters and compute position PnL from pre-computed regime labels.

### `_walk_forward_classify` mode 3 deleted

Per-bar `engine.classify` without precomputed features (mode 3) has no callers after this change. The mode is removed, leaving modes 1 (regimes replay) and 2 (precomputed adaptive refit).

### Profiling hooks

`_phases` dict and `_classify_times` list are passed via `**kwargs` through `run_classify` to `_hmm_classify_pipeline`. Threshold engines receive and ignore them. The profiling hooks remain pipeline-internal plumbing — they are not part of the engine interface.

### BIC resolution stays inside the HMM classify path

`_hmm_classify_pipeline` resolves `n_states="auto"` via BIC and mutates `engine.n_states = resolved`. This is unchanged — the mutation stays inside the engine's `run_classify` call path. Pipeline reads the resolved value from `ClassifyOutput.n_states`.

### Test impact

| Tests | Change |
|-------|--------|
| `test_classify_pipeline.py` | Call `engine.run_classify()` instead of `_classify_threshold_pipeline` / `_hmm_classify_pipeline` |
| `test_walk_forward_classify.py` | Mode 3 test deleted; modes 1 & 2 unchanged |
| `test_walk_forward_engine_param.py` | Updated for new `walk_forward_backtest` signature (no `engine`, required `regimes`) |
| `test_regime_engine.py` (walk-forward tests) | Pass `regimes=` to `walk_forward_backtest` instead of `engine=` |
| `test_filters.py` | Same — pass `regimes=` + `posteriors=` |
| `test_pca_whitening.py` | Same — pass `regimes=` |
| `test_regime_pipeline.py` | Unchanged — calls `pipeline.run()` end-to-end |
| `test_regime_engine.py` (is_hmm tests) | Unchanged — tests config properties, not pipeline dispatch |
| `test_profile_pipeline.py` | Unchanged — calls `pipeline.run()` end-to-end |
