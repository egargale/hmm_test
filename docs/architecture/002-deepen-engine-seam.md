# ADR-002: Deepen the Engine Seam

**Status**: Proposed
**Date**: 2026-05-26
**Priority**: Strong

## Context

The package supports three engines (threshold, messina, hmm) but dispatches them via string-matching `if/elif` in `walk_forward.py` (lines 136–148). The HMM engines share a monolithic `hmm_state_from_slice()` function in `hmm_adapter.py` with 5 boolean parameters controlling its behaviour across two modes (precompute vs per-bar fit) and two HMM variants (generic vs messina).

This is a **shallow seam**: the interface exposes every internal distinction, and adding a 4th engine means editing 3 files (`walk_forward.py`, `hmm_adapter.py`, `pipeline.py`).

## Current Structure

```
walk_forward_backtest(engine="threshold")
  ├── if engine == "threshold"  → _walk_forward_threshold()
  │                                 └── markov_chain.classify_regimes()
  ├── elif engine in _HMM_ENGINES → _walk_forward_hmm(use_messina=bool)
  │                                 └── hmm_state_from_slice(
  │                                       use_messina, return_features,
  │                                       prev_means, precomputed)
  └── performance_metrics (2 fns)
```

### Problems

1. **String dispatch**: `engine` is a string validated in multiple places (`_VALID_ENGINES` in both `pipeline.py` and `walk_forward.py`). No compile-time or import-time guarantee of consistency.

2. **God function**: `hmm_state_from_slice()` has 5 boolean-ish parameters. It handles precomputation, per-bar fitting, and state matching — three separate concerns behind one interface.

3. **HMM engine coupling**: `_walk_forward_hmm()` takes a `use_messina` boolean to pick between messina and generic features. This means the HMM adapter knows about both feature sets internally.

4. **No testability at the seam**: you can't mock "an engine" to test walk-forward independently. You'd have to mock specific functions inside the if/elif branches.

## Proposed Design

Define a `RegimeEngine` protocol (structural typing, no inheritance required):

```python
class RegimeEngine(Protocol):
    def precompute(self, data: pd.DataFrame) -> EngineFeatures | None: ...
    def classify(self, data: pd.DataFrame, up_to: int, state: EngineState | None) -> tuple[int, EngineState]: ...
```

Each engine becomes its own module:

- `regime/engines/threshold.py` — wraps `markov_chain.classify_regimes()`
- `regime/engines/hmm_generic.py` — HMM with generic features (~44)
- `regime/engines/hmm_messina.py` — HMM with Messina features (18)

`walk_forward_backtest` takes `engine: RegimeEngine` instead of `engine: str` + kwargs. Engine instantiation happens once in `cli.py` or `pipeline.py` from the string arg.

```python
ENGINE_REGISTRY: dict[str, type[RegimeEngine]] = {
    "threshold": ThresholdEngine,
    "hmm": HmmGenericEngine,
    "messina": HmmMessinaEngine,
}
```

## Consequences

**Positive:**
- Adding a new engine = write one module, register in one dict
- Each engine's walk-forward, feature, and HMM logic lives in one place
- Testable in isolation: mock the engine protocol to test walk-forward; test each engine independently
- Two adapters = real seam (currently only hypothetical)

**Negative:**
- More files (3 engine modules vs current 2 files)
- Protocol definition adds a small abstraction layer
- `hmm_adapter.py` would be split across engine modules

## Related

- [[ADR-001]] Excise dead weight — do this first; dead `hmm_models/` and `model_training/` confuse the dependency graph
- [[ADR-004]] CLI extraction — engine instantiation could move to the CLI layer
