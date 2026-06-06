# 0022: Break the _hmm_pipeline ↔ walk_forward reverse dependency

## Status

Accepted

## Context

`walk_forward.py` imported `_walk_forward_classify` from `_hmm_pipeline.py` solely to use **mode 1**: replaying pre-computed regime labels without calling any engine. This mode is a trivial 3-line loop:

```python
for t in range(min_train, n):
    yield t, ClassifyResult(regime=int(regimes[t]))
```

This created a reverse dependency: `walk_forward.py` (a regime-agnostic backtest module) imported from `_hmm_pipeline.py` (an HMM-specific module). The git forensics coupling score between `pipeline.py` ↔ `walk_forward.py` was **1.00** (14 co-changes, never changed independently). Every change to `_walk_forward_classify` required a corresponding change in `walk_forward.py`.

The walk-forward backtest has no business importing HMM internals. Mode 1 doesn't use any HMM imports — it just iterates `range(min_train, n)` and yields pre-computed regime labels.

## Decision

Extract the regime replay generator into `walk_forward.py` as `_replay_regimes(regimes, min_train)`. This is a local function that yields `(t, regime)` tuples — no engine imports, no HMM imports, no ClassifyResult wrapper.

`_walk_forward_positions()` now calls `_replay_regimes()` instead of importing `_walk_forward_classify(mode=1)` from `_hmm_pipeline.py`.

Mode 1 is retained in `_walk_forward_classify` with a deprecation note for backward compatibility with existing tests.

## Consequences

- **`walk_forward.py` no longer imports from `_hmm_pipeline.py`**. The dependency graph is now strictly downward: `pipeline.py → walk_forward.py → nothing except backtesting/`.
- **The 1.00 coupling score should decrease** — future changes to the HMM walk-forward classify loop don't require changes to `walk_forward.py`.
- **Tests unchanged** — `_walk_forward_classify` mode-1 tests still pass (the function retains mode-1 for backward compat).
- **~6 lines removed from walk_forward.py** (the import + call) and ~10 lines added (the `_replay_regimes` generator).
- **New `_replay_regimes` generator** is independently testable with just a numpy array — no HMM imports needed.
