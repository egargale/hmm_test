# Pipeline `run()` decomposition

`pipeline.run()` was a 220-line god function containing seven interleaved phases: input validation, n-states resolution, engine-specific regime classification, Markov chain statistics, output dict construction, walk-forward backtest invocation, and profiling. We decomposed it into a ~60-line orchestrator calling four extracted helpers, with two internal dataclasses (`ClassifyOutput`, `MarkovStats`) carrying intermediate state between phases.

## Considered options

### A) Pipeline-of-objects pattern

Each phase becomes a class with `execute()`, composed into a pipeline object. Flexible for branching or conditional phases.

**Rejected because**: the pipeline is strictly linear — validate, classify, compute stats, backtest, assemble. No branching, no conditional skipping, no parallelism. A pipeline-of-objects adds indirection for zero flexibility gain in this domain.

### B) Extract every phase into a helper

Eight phases, eight helpers. Uniform structure: `_validate_prices`, `_resolve_n_states`, `_classify_threshold`, `_classify_hmm`, `_build_markov_stats`, `_build_forecasts`, `_build_engine_info`, `_assemble_result`.

**Rejected because**: three of the eight are trivial — `_classify_threshold` is one line, `_build_forecasts` is three lines, `_resolve_n_states` is ~10 lines tightly interleaved with the BIC/precompute orchestration. Extracting one-liners adds indirection and noise. The principle: extract when the block is long enough to obscure the calling function's flow, or reusable in more than one place. Four of eight met that bar.

### C) Four targeted helpers (chosen)

Extract the four blocks that are genuinely complex or self-contained: `_validate_prices` (15 guard clauses), `_classify_hmm` (50-line walk-forward classify loop), `_build_markov_stats` (~30 lines of pure computation from a regimes array), `_build_engine_info` (~20 lines with duck-typed `enrich_info()` call). Leave the rest inline — threshold classify is one line, forecasts are three lines, BIC resolution is orchestration that belongs in `run()`, and the final result dict is the consumption of all phase outputs.

**Chosen because**: (1) `run()` drops from 220 to ~60 lines — readable top-to-bottom; (2) each helper has a clear contract: pure inputs → typed output; (3) no over-abstraction — the inline blocks are short and read naturally in context.

## Consequences

- **Two internal dataclasses carry intermediate state.** `ClassifyOutput` (regimes, posteriors, warmup bars, engine instance) and `MarkovStats` (transition matrix, stationary distribution, persistence, signal, regime counts, dates, current regime/probs). Both live in `pipeline.py` — they are pipeline-internal bookkeeping, not part of the engine contract. External callers never see them.

- **`_classify_hmm` receives a pre-constructed `RegimeEngine` and precomputed features.** Engine construction, precompute, BIC resolution, and `dataclasses.replace()` for `n_states='auto'` all happen in `run()`. The helper runs the classify loop only. This keeps orchestration in the orchestrator and keeps the helper focused.

- **`_build_engine_info` receives the engine instance directly, not `ClassifyOutput`.** It needs only the config, resolved n_states, and engine instance. Tight contracts — pass only what's needed.

- **`_validate_prices` validates `prices` only.** OHLCV validation is each engine's responsibility (ADR-0004). The pipeline passes `ohlcv` through without checking it.

- **Profiling state is passed as mutable arguments.** `_phases` dict and `_classify_times` list are passed into `_classify_hmm` when `profile=True`. They are not returned in `ClassifyOutput` — profiling is a cross-cutting diagnostic concern, not a domain result.

- **Threshold classification stays inline.** After config consolidation (ADR-0004), the threshold path is one line: `classify_regimes(returns, window=config.window, threshold=config.threshold)`. Wrapping it in a function adds a line of indirection for a line of computation.

- **`run()` remains the single entry point.** The CLI and Python callers call `run()`. The helpers are private (`_`-prefixed). The decomposition is internal — no external API change beyond the signature update from ADR-0004.
