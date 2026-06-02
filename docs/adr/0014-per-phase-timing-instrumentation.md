# Per-phase timing instrumentation

The pipeline was a black box for performance analysis. Users hitting slow runs (especially `fshmm` on long histories) had no way to identify which phase was the bottleneck — feature engineering, HMM fitting, walk-forward classify, Markov stats, or duration forecast. We instrumented the pipeline with per-phase timing using `time.monotonic()` and threaded mutable dict/list containers through the helpers.

## Considered options

### A) No profiling — ship as-is

Rely on `time` invocations at the call site and manual tracing when a run is slow.

**Rejected because**: (1) performance debugging was ad-hoc — every investigation required editing source files to insert `print(time.monotonic())` calls, which then had to be reverted; (2) the regression suite routinely exceeded 120s (issue #32), and we had no systematic way to detect which commit caused a regression; (3) cross-engine comparisons (threshold vs. hmm vs. fshmm) needed identical measurement points to be comparable.

### B) `cProfile` / `profile` module

Run the full pipeline under `cProfile` and analyse the call tree post-hoc to identify hot spots.

**Rejected because**: (1) `cProfile` adds high overhead (~2-5× slowdown), making timing measurements themselves unreliable — the instrumentation becomes the bottleneck; (2) call-granularity profiling produces thousands of frames; identifying the "feature engineering phase" vs. "HMM fitting phase" requires post-hoc manual grouping of call stacks; (3) `cProfile` is platform-specific on some edge cases (e.g., certain Python builds on Windows); (4) the primary audience is the regression test suite (CI) and CLI users — both need summary numbers, not flame graphs.

### C) Manual per-phase timing with `time.monotonic()` — chosen

Insert `t_start = time.monotonic()` at the start of each major phase and `elapsed = time.monotonic() - t_start` at the end, collecting results into a `_phases` dict threaded through helpers. Collect per-call `_classify_times` as a list for distributional stats (min/median/p99).

**Chosen because**: (1) zero external dependencies — `time.monotonic()` is the stdlib; (2) overhead is ~1 μs per reading + a dict insert, imperceptible at pipeline scale (phases take 0.1–10s); (3) cross-platform — works identically on Linux, macOS, Windows; (4) the five phases (precompute, BIC selection, walk-forward classify, walk-forward backtest, total wall) map directly to the code structure a developer reading `pipeline.run()` sees; (5) the result is a JSON-serialisable `timing` dict that fits naturally into the pipeline output.

**Rejected**: per-call timing at granularity finer than the five phases. The `_classify_times` list (one entry per `eng.classify()` call in the walk-forward loop) is a deliberate midpoint — enough distribution detail (min/median/p99) to spot HMM fitting variance without instrumenting individual EM iterations.

### D) Per-call timing with callback or context manager

Wrap every `eng.classify()`, `select_n_states()` iteration, and `_build_markov_stats()` call in a `@timed` decorator or `with Timer()` context manager.

**Rejected because**: (1) five phases is the right granularity for identifying bottlenecks — finer granularity would produce dozens of timing entries that need aggregation to be useful; (2) a Timer context manager adds a third abstraction layer (the mutable-dict pattern is already a design tradeoff — see consequences); (3) the per-call `_classify_times` list already provides distribution-level visibility into the most expensive sub-operation (HMM fitting during walk-forward).

## Decision details

### Implementation

- **File**: `hmm_futures_analysis/regime/pipeline.py` — changes to `run()` and `_classify_hmm()`.
- **File**: `hmm_futures_analysis/regime/engines/_hmm_shared.py` — changes to `select_n_states()`.
- **Commit**: `b266abd`.

### Phases tracked

| Phase key | What it measures | Where |
|---|---|---|
| `precompute` | Feature engineering (calling `engine.precompute()`) | `pipeline.run()` |
| `bic_select_n_states` | BIC loop (fitting HMMs for 2..N states) | `pipeline.run()` |
| `walk_forward_classify` | Full walk-forward classify loop | `_classify_hmm()` — wraps the for-loop |
| `walk_forward_backtest` | Walk-forward backtest pass | `pipeline.run()` |
| `bic_detail` | Per-state-count timing within BIC selection | `select_n_states()` — injected into `_phases` dict |

Plus a `walk_forward_classify_stats` block with min/median/p99/n_calls derived from the `_classify_times` list.

### The mutable-arguments-through-helpers pattern

`_phases: dict[str, float]` and `_classify_times: list[float]` are created in `pipeline.run()`, then passed as keyword arguments through `_classify_hmm()` and `select_n_states()`. The helpers mutate these containers in-place; `pipeline.run()` reads the mutations after the helpers return.

This is a deliberate design choice with the following reasoning:

**Why not a return value?** Profiling is a cross-cutting concern, not a business-logic output. Adding a timing return to `_classify_hmm()` (which already returns `ClassifyOutput`) would either require a tuple return (breaking the natural return type) or a wrapper object — both add ceremony at every call site. The mutable-argument pattern keeps the function signature clean: `profile=True` is the only profiling-related parameter.

**Why not a `ProfilingContext` dataclass?** A single dataclass object wrapping `_phases` and `_classify_times` is equivalent in behaviour but adds an import and a class definition. The plain-dict/plain-list approach is more transparent — any developer can see exactly what's being mutated. Given that the pattern is contained to three files (pipeline.py, _hmm_shared.py, test_profile_pipeline.py), the ceremony of a dedicated type is not justified.

**Trade-offs acknowledged**: (1) the helper functions are no longer pure — they mutate caller-owned state as a side effect; (2) the pattern is unusual enough that a new contributor might look for a return value; (3) if profiling scope grows significantly (e.g., timing every sub-phase of feature engineering), a `ProfilingContext` object should be revisited.

### Profiling is opt-in by default

As of ADR-0014, `profile=True` is the default. The overhead of the instrumentation is negligible: each `time.monotonic()` call is ~1 μs, and the dict/list appends are O(1) amortised. The total instrumentation overhead for a full pipeline run is well under 1 ms — imperceptible at the scale of phases that take 100 ms to 10 seconds.

The `timing` dict in the output adds 3–7 small numeric fields and one small dict (the `bic_detail` sub-dict, ~5 entries). This is negligible compared to the rest of the output (walk-forward tables, regimes array, etc.). Users who need to suppress it can pass `profile=False`.

### Threshold engine

For the threshold engine (no HMM, no BIC selection), only `walk_forward_backtest` and `total_wall_seconds` are meaningful. The `phases` dict will be empty (no HMM phases to measure), and `walk_forward_classify_stats` is absent because there is no `eng.classify()` loop.

## Consequences

- **Zero-dependency profiling.** The entire instrumentation uses `time.monotonic()` from the stdlib. No external profilers, no optional dependencies.
- **Opt-in by default.** Every pipeline run now includes a `timing` key in the output. This is the right default because the overhead is imperceptible and the data is frequently useful. Pass `profile=False` to suppress.
- **Mutable state in helpers.** `_classify_hmm()` and `select_n_states()` mutate dicts/lists owned by the caller. This is an intentional tradeoff (see above) that keeps the cross-cutting concern from polluting return types. If the pattern grows beyond three call sites, a `ProfilingContext` dataclass should be considered.
- **Profiling not in `ClassifyOutput`.** The return type of `_classify_hmm()` remains unchanged — timing is assembled by the caller. This keeps profiling a pipeline-level concern rather than leaking into the classify abstraction.
- **Per-call distribution stats.** The `walk_forward_classify_stats` block (min/median/p99/n_calls) provides distribution-level insight into individual `eng.classify()` calls without instrumenting at finer granularity. This is sufficient to distinguish "one slow fit" from "uniformly slow fits".
- **CI-friendly.** The timing output is JSON-serialisable by construction. Tests can assert phase timing exists and is positive without hardcoding thresholds (see `test_profile_pipeline.py`).
- **CLI-friendly.** When `--json` mode is active (used programmatically), the timing data flows naturally into the JSON output. In terminal mode, timing is not printed (the `_print_terminal()` function selects a subset of output keys).
