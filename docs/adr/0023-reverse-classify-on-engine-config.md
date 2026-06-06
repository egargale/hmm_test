# 0023: Absorb `reverse_classify` into engine config

## Status

Accepted

## Context

The `reverse_classify` boolean was threaded through 4 call layers:

```
cli.py → pipeline.run(reverse_classify=) → eng.run_classify(reverse=)
  → _hmm_classify_pipeline(reverse=) → _walk_forward_classify(reverse=)
```

Only the innermost function (`_walk_forward_classify`) actually used it. The threshold engine ignored it entirely — a no-op parameter on every call. A side-channel field `ClassifyOutput.reverse_classify` carried the same fact back up for the lookahead warning.

The deletion test: the iteration direction logic is genuinely complex and belongs in `_walk_forward_classify`. But the interface was nearly as complex as the implementation — 4 boolean parameters + 1 side-channel field for a single concept.

Key observation: `reverse_classify` only affects HMM engines. The threshold engine does one-shot vectorized classification — no walk-forward loop, no iteration direction. This makes `reverse_classify` an HMM-specific property, not a generic pipeline parameter.

## Decision

Move `reverse_classify` from the pipeline parameter chain into the HMM engine config dataclasses. It now travels the same path as other engine-specific parameters (`pca_variance`, `robust_method`, `saliency_threshold`):

```
cli.py → engine_config_builder sets config.reverse_classify
       → pipeline.run(engine_config=) passes config through
       → resolve_engine(config) → engine.reverse_classify
       → _hmm_classify_pipeline reads getattr(engine, 'reverse_classify', False)
```

Changes:
- `HMMGenericConfig`, `HMMMMessinaConfig`, `RobustHMMConfig`, `FSHMMConfig` gain `reverse_classify: bool = False`
- `ThresholdConfig` does NOT gain it (threshold doesn't do walk-forward)
- `HMMEngineBase.__init__` accepts and stores `reverse_classify`
- `_hmm_classify_pipeline` reads it from the engine object instead of a parameter
- `pipeline.run()` reads it from `engine_config` for the lookahead warning
- `cli.py` no longer passes `reverse_classify` as a separate kwarg

## Consequences

- **4-layer boolean threading eliminated**: `pipeline.run()`, `eng.run_classify()`, `_hmm_classify_pipeline()` no longer accept `reverse`/`reverse_classify`
- **`_walk_forward_classify(reverse=)` remains** — internal to `_hmm_pipeline.py`, called by `_hmm_classify_pipeline`
- **ThresholdConfig is clean** — no unused `reverse_classify` field
- **`ClassifyOutput.reverse_classify` field remains** — still set by `_hmm_classify_pipeline`, still used by `pipeline.run()` for the lookahead warning
- **`engine_config_builder.py` gains `reverse_classify` wiring** for HMM configs
- **Tests updated**: `test_classify_pipeline.py` sets `reverse_classify` on config/engine instead of passing it to `run()` or `_hmm_classify_pipeline()`
