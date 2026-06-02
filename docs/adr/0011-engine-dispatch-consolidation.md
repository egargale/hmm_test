# Engine dispatch consolidation

Engine-specific parameters (`robust_method`, `saliency_threshold`, `window`, `threshold`, `pca_variance`, `n_states`) leaked into generic plumbing — `pipeline.run()` had 18 kwargs and `walk_forward_backtest()` had 16, with `if engine == "robust_hmm"` branches scattered across three files. Adding a sixth engine meant editing five locations. We introduced per-engine config dataclasses to encapsulate engine-specific parameters behind a single seam, shrinking both signatures and eliminating all engine-name branches from the pipeline and walk-forward modules.

## Considered options

### A) Pass-through `**engine_kwargs` dict

Pipeline and walk-forward accept `**engine_kwargs` and forward them blindly to engine constructors. Minimal plumbing, but no type safety, no IDE autocomplete, and the `pipeline.run()` docstring becomes meaningless.

**Rejected because**: the whole point of the consolidation is to make engine dispatch *legible*. An untyped catch-all dict is the opposite of that.

### B) `from_namespace()` factory on each config

Each config dataclass carries a `@classmethod from_namespace(cls, args: argparse.Namespace)` that knows how to extract its own CLI args.

**Rejected because**: it couples configs to `argparse`. Callers that aren't the CLI (tests, notebooks, future API) don't have a `Namespace`. Pure dataclasses constructed with `SomeConfig(**params)` work for everyone.

### C) Per-engine config dataclasses with CLI-side construction (chosen)

Five flat dataclasses — one per engine — carrying all constructor parameters plus a `name` and `features` field. The CLI maps `args.engine` to the right config class and constructs it directly. The registry maps `name → (EngineClass, ConfigClass)`. Pipeline and walk-forward receive a config or a `RegimeEngine` instance, never engine-specific kwargs.

**Chosen because**: (1) each config is the single source of truth for what an engine needs; (2) no inheritance — engines are independent per ADR-0001, and their configs should be too; (3) configs are pure data — no framework dependencies; (4) `pipeline.run()` shrinks from 18 kwargs to 10, `walk_forward_backtest()` from 16 to 6; (5) adding a new engine means writing one config + one engine class + one registry entry, with zero changes to pipeline or walk-forward.

## Consequences

- **`EngineConfig` is an informal protocol, not a base class.** Each config is a standalone `@dataclass`. The registry accepts any dataclass whose fields match the engine constructor. This avoids a shared base that would couple independent engines.

- **`HMM_ENGINES` remains a hardcoded frozenset.** Some pipeline behavior genuinely differs between threshold and HMM engines (BIC selection, posteriors pass-through). The set is two lines in `engine_protocol.py` and is updated alongside the registry when a new HMM engine is added.

- **`n_states='auto'` is resolved before engine construction.** The config may start with `n_states='auto'` (string). Pipeline resolves it to an integer via BIC, then produces a new config with `dataclasses.replace()`. Configs stay immutable.

- **`walk_forward_backtest()` accepts a `RegimeEngine` instance, not a string.** The factory lives in one place. Walk-forward is a pure consumer — given an engine and prices, run the backtest. No dispatch logic.

- **Engine info enrichment is duck-typed.** Engines that need to add metadata to the output JSON implement `enrich_info(info: dict) -> dict`. Pipeline calls it via `hasattr` check. The `RegimeEngine` protocol stays minimal.

- **Each engine validates its own OHLCV requirement.** If an HMM engine receives `None` data, it raises a clear error. Pipeline doesn't need `HMM_ENGINES` for input validation — only for the posteriors pass-through optimization.

- **`_resolve_engine()` and `_VALID_ENGINES` are deleted from `walk_forward.py`.** Engine construction moves to a single factory function in `engine_protocol.py`.

- **`_ENGINE_FEATURES` dict is deleted from `pipeline.py`.** The `features` label lives on each config dataclass.

- **Cross-cutting diagnostics may pass through `run_classify()` via `**kwargs`.** Profiling hooks (`profile`, `_phases`, `_classify_times`) are pipeline-internal diagnostics, not engine-specific configuration. They pass through `run_classify(**kwargs)` without widening the protocol. See ADR-0017.
