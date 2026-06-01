# ADR-0009: Deepen Engine Seam with RegimeEngine Protocol

**Status**: Accepted
**Date**: 2026-05-26
**Priority**: Medium
**Depends on**: ADR-001 (excised dead modules)
**Implemented**: 2026-05-26 (commit `0b13329` — RegimeEngine protocol, ENGINE_REGISTRY, 268 new tests)

## Context

The walk-forward backtest in `walk_forward.py` dispatches to three engines
(`threshold`, `hmm`, `messina`) via a string `if/elif` chain (lines 136–148).
`_VALID_ENGINES` is duplicated in `pipeline.py`. Adding a new engine requires
touching multiple files and hard-coding new branches.

The engines share no common interface — each is a loose collection of module-level
functions called by name. This makes it hard to test engines in isolation, mock
them in walk-forward tests, or add engines without modifying the dispatcher.

## Decision

Introduce a `RegimeEngine` protocol and an `ENGINE_REGISTRY` dict. Each engine
becomes a class satisfying the protocol. The walk-forward loop and pipeline
resolve engines through the registry, eliminating `if/elif` dispatch and the
duplicated `_VALID_ENGINES` set.

### Protocol definition

```python
from typing import Protocol
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ClassifyResult:
    regime: int                        # 0=bear, 1=sideways, 2=bull
    means: np.ndarray | None = None    # for state matching (HMM engines)

class RegimeEngine(Protocol):
    def precompute(self, data: pd.DataFrame) -> pd.DataFrame | None: ...
    def classify(self, data: pd.DataFrame, prev_means: np.ndarray | None = None) -> ClassifyResult: ...
```

### Design decisions (review gaps resolved)

1. **Skip-N refit ownership** — `walk_forward_backtest` owns the refit cadence.
   `classify()` classifies a single slice. The walk-forward loop decides *when*
   to call it and holds the last result between refits.

2. **Engine constructor params** — Each engine takes its tuning params at
   construction:
   - `ThresholdEngine(window=20, threshold=0.05)`
   - `HMMGenericEngine(n_states=3)`
   - `HMMMMessinaEngine(n_states=3)`

3. **Shared HMM utilities** — `_fit_hmm_on_slice()`, `_match_states()`,
   `_engineer_features()` move to `regime/engines/_hmm_shared.py`. Both HMM
   engine classes import from there.

4. **Input validation boundary** — Each engine validates its own inputs in
   `precompute()` and `classify()`. Top-level checks (data length, engine
   string) stay in `walk_forward_backtest`.

5. **`classify()` return type** — Returns **regime index** (0/1/2). The
   `_STATE_MAP = {0: -1, 1: 0, 2: 1}` mapping to trading positions stays in
   the walk-forward code. Regime index is the domain concept; position is a
   trading concept.

6. **`precompute()` returning `None`** — Threshold engine returns `None`.
   The walk-forward loop checks: if precomputed data is `None`, pass raw
   returns to `classify()`; otherwise pass precomputed features.

### Engine registry

```python
ENGINE_REGISTRY: dict[str, type[RegimeEngine]] = {
    "threshold": ThresholdEngine,
    "hmm": HMMGenericEngine,
    "messina": HMMMMessinaEngine,
}
```

### File layout

```
regime/
├── engine_protocol.py          # RegimeEngine, ClassifyResult, ENGINE_REGISTRY
├── engines/
│   ├── __init__.py
│   ├── threshold.py            # ThresholdEngine
│   ├── hmm_generic.py          # HMMGenericEngine
│   ├── hmm_messina.py          # HMMMMessinaEngine
│   └── _hmm_shared.py          # shared HMM utilities
├── markov_chain.py             # unchanged
├── pipeline.py                 # use registry
├── walk_forward.py             # dispatch via protocol
└── hmm_adapter.py              # deprecated (logic moves to engines/)
```

### Backward compatibility

`walk_forward_backtest(engine="threshold")` continues to work — the string is
resolved via `ENGINE_REGISTRY` internally. The public API of
`walk_forward_backtest` and `pipeline.run` does not change.

## Consequences

**Positive:**
- Adding a new engine requires one class + one registry entry — no dispatcher changes
- Engines are testable in isolation without the full walk-forward machinery
- Walk-forward tests can inject mock engines
- `_VALID_ENGINES` duplication eliminated (single source of truth in registry)
- `hmm_adapter.py` multi-concern function `hmm_state_from_slice()` decomposes into
  focused `precompute()` / `classify()` methods

**Negative:**
- One more layer of indirection (registry lookup vs direct dispatch)
- `hmm_adapter.py` becomes deprecated — callers importing from it need updating
- HMM engines carry state (`prev_means`) across calls, making them slightly
  less pure than threshold — but this is inherent to the state-matching algorithm

## Related

- [[ADR-0008]] Excise dead weight modules (prerequisite — already merged)
