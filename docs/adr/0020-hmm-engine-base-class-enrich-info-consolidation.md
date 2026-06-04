# HMM engine base class + enrich_info consolidation

**Status**: Accepted
**Date**: 2026-06-04
**Priority**: Medium
**Depends on**: [[ADR-0003]], [[ADR-0009]], [[ADR-0017]]
**Implemented**: 2026-06-02 (issues #78, #79)

## Context

After ADR-0009 introduced the `RegimeEngine` protocol and ADR-0017 pushed
`run_classify()` into the protocol, four HMM-family engines (`hmm`,
`messina`, `robust_hmm`, `fshmm`) shared approximately 90% boilerplate in
their `__init__`, `precompute`, and `run_classify` implementations. Each
engine duplicated the same parameter storage (`n_states`, `pca_variance`),
the same feature-engineering dispatch (generic vs. Messina), and the same
delegation to `_hmm_classify_pipeline`.

Separately, engines that produced metadata beyond the standard
`ClassifyOutput` fields did so via an `enrich_info()` method discovered at
call sites by `hasattr` duck typing. The pipeline consumed it like:

```python
if hasattr(engine, "enrich_info"):
    info.update(engine.enrich_info())
```

This had two problems: (1) a new engine that forgot to implement
`enrich_info` would silently produce incomplete metadata, and (2) the
pattern would multiply as more engines added diagnostic fields.

Two changes were implemented in #78 and #79:

1. **`HMMEngineBase` abstract base class** (#78) — extracts the shared
   boilerplate into a single class that all four HMM engines inherit.

2. **`enrich_info` → `ClassifyOutput.engine_info`** (#79) — moves engine
   metadata from a duck-typed method into a first-class field on the
   classify output.

## Decision

### 1. `HMMEngineBase` as an implementation detail, not a protocol member

Introduce `HMMEngineBase` in `_hmm_engine.py` as an ABC providing default
`__init__`, `precompute`, `_build_engine_info`, and `run_classify`.
Concrete HMM engines inherit it and override `classify()` (and optionally
the other methods) to inject differentiated logic.

The base class is **not** part of the `RegimeEngine` protocol
([[ADR-0009]]). The protocol remains the public contract; `HMMEngineBase`
is a private implementation detail shared by the four HMM engines.
Non-HMM engines (threshold) continue to implement the protocol directly.

### 2. Fold `enrich_info` into `ClassifyOutput.engine_info`

Replace the `hasattr`-duck-typed `enrich_info()` method with a typed field
on `ClassifyOutput`:

```python
@dataclass
class ClassifyOutput:
    regimes: np.ndarray
    posteriors: np.ndarray | None = None
    last_regime: int = 1
    warmup_bars: int | None = None
    n_states: int = 3
    engine_info: dict | None = None   # ← new
```

HMM engines populate it via `_build_engine_info()` called inside
`run_classify()`. The pipeline reads `classify_out.engine_info`
unconditionally — `None` means no engine metadata, which is the correct
default for threshold.

### 3. `enrich_info` is NOT added to the `RegimeEngine` protocol

The threshold engine has no enrichment to add. Adding an optional method
to the protocol would reintroduce the same `hasattr` / `Optional` pattern
at a different layer. A `ClassifyOutput` field is cleaner because:

- The pipeline already consumes `ClassifyOutput` — no new type needed.
- `None` is the natural sentinel for "no metadata" rather than a missing
  method.
- The field is type-checked; forgetting to populate it produces a clear
  `None` rather than silently missing data.

## Considered alternatives

### A) Protocol mixin instead of ABC

Define a `HMMSupport` mixin protocol with shared method signatures, letting
each HMM engine satisfy both `RegimeEngine` and `HMMSupport`.

**Rejected because**: a mixin protocol doesn't reduce boilerplate — each
engine still needs to implement the mixin's methods. An ABC with default
implementations is the right tool for "shared default behaviour with
selective overrides."

### B) Keep `enrich_info` as a protocol method (Optional)

Add `def enrich_info(self) -> dict | None: ...` to `RegimeEngine`, with a
default return of `None`.

**Rejected because**: this re-introduces the `hasattr` / `Optional`
problem at the protocol level. Every caller would still check `is not None`
or use `getattr`. Putting the field on `ClassifyOutput` means the pipeline
already has the value — no extra dispatch needed.

### C) Single `HMMEngine` class with injected `FitStrategy`

Collapse the four HMM engines into one class parameterised by a fit
strategy object.

**Rejected because**: too large a refactor for the immediate need. The
engines have diverged enough (robust post-hoc correction, saliency
weights, Messina vs. generic features) that a single class would become a
configuration-driven dispatch itself. This can be revisited if a clear
pattern emerges.

## Consequences

### Adding a 6th HMM engine

A future engine (e.g., Student-t emissions from #42) now only needs to:

1. Inherit `HMMEngineBase`
2. Override `classify()`
3. Optionally override `_build_engine_info()` for extra metadata

No pipeline changes required. No `enrich_info` method to remember. The
engine is wired in by adding one entry to `ENGINE_REGISTRY`.

### Adding a non-HMM engine

Continue implementing the `RegimeEngine` protocol directly. Set
`engine_info = None` (or omit it) on the returned `ClassifyOutput`. The
pipeline handles `None` gracefully.

### Pipeline simplification

The `_build_engine_info` helper in `pipeline.py` reads
`classify_out.engine_info` unconditionally via `getattr` with a `None`
default — no engine-type branching:

```python
engine_meta = getattr(classify_out, "engine_info", None)
if engine_meta:
    info.update(engine_meta)
```

### Threshold engine unchanged

Threshold does not inherit `HMMEngineBase` and does not populate
`engine_info`. Its `ClassifyOutput` has `engine_info=None`. This is
correct — threshold produces no HMM-specific metadata.

## Related

- [[ADR-0003]] Engine self-containment contract — each engine owns its
  outputs; `engine_info` extends this ownership to metadata
- [[ADR-0009]] Deepen engine seam with `RegimeEngine` protocol — the
  protocol that `HMMEngineBase`-derived classes satisfy
- [[ADR-0017]] Push `run_classify()` into engine protocol — the method
  that `HMMEngineBase.run_classify` provides as a shared default
