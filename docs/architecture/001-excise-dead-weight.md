# ADR-001: Excise Dead Weight Modules

**Status**: Accepted
**Date**: 2026-05-26
**Priority**: Strong
**Revised**: 2026-05-26 (corrected dead-module list and line counts)
**Implemented**: 2026-05-26 (commit `1e2736b`, PR #9 — ~5,650 lines excised, 135/135 tests pass)

## Context

The package was renamed from `scripts/` to `hmm_futures_analysis/` (commit `4d60098`) to support dual distribution as a library and a Claude Code skill. The CLI exposes three independent engines (`threshold`, `messina`, `hmm`) via `--engine`. Analysis shows ~53% of the codebase (~6,060 lines across 12 modules/classes) is unreachable from any engine path.

### Active import graph (all three engines)

```
cli.py
├── data_processing/csv_auto_detect.py
│   ├── csv_parser.py
│   ├── csv_format_detector.py
│   └── data_validation.py
└── regime/pipeline.py
    ├── regime/markov_chain.py
    └── regime/walk_forward.py
        ├── backtesting/performance_metrics.py ← imports PerformanceMetrics from data_types.py
        ├── regime/markov_chain.py
        └── regime/hmm_adapter.py ← imported at module level for ALL engines
            ├── data_processing/feature_engineering.py (add_features only)
            │   └── data_processing/technical_indicators.py (2 symbols: get_default_indicator_config, validate_ohlcv_columns)
            └── data_processing/messina_features.py
```

### Three runtime paths

| Engine | Calls `hmm_adapter`? | Calls `add_features`? | Calls `add_messina_features`? |
|--------|----------------------|-----------------------|-------------------------------|
| `threshold` | No — uses `markov_chain` only | No | No |
| `hmm` | Yes | Yes | Imported but not called |
| `messina` | Yes | No | Yes |

Note: `walk_forward.py` imports `hmm_adapter` at module level, so `feature_engineering`, `technical_indicators`, and `messina_features` are loaded into memory for all engines — even `threshold`. They are not "dead imports" in the Python sense, but only `threshold` avoids calling them.

## Dead modules

The following are never imported or called from any engine path:

| Module | Lines | Why dead |
|--------|-------|----------|
| `hmm_models/` (base, factory, gaussian_hmm, gmm_hmm, \_\_init\_\_) | 2,164 | Own HMM wrapper classes; pipeline uses `hmmlearn` directly via `hmm_adapter.py` |
| `model_training/` (trainer, inference_engine, persistence, \_\_init\_\_) | 1,901 | Training orchestration that wraps `hmm_models/`; never called |
| `backtesting/performance_analyzer.py` | 346 | Wraps `performance_metrics`; pipeline calls metrics directly |
| `backtesting/bias_prevention.py` | 551 | Lookahead detection; never called in walk-forward path |
| `backtesting/utils.py` | 398 | Transaction cost helpers; never called |
| `utils/config.py` | 249 | Pydantic `Config` with `HMMConfig`, `BacktestConfig`; never loaded by any engine |
| `FeatureEngineer` class in `feature_engineering.py` (lines 852–1299) | 448 | Class with selection/quality/summary methods; pipeline calls `add_features()` only |
| **Total** | **6,057** | |

### Alive modules (do NOT delete)

| Module | Lines | Why alive |
|--------|-------|-----------|
| `data_processing/technical_indicators.py` | 146 | Module-level import by `feature_engineering.py` (`get_default_indicator_config`, `validate_ohlcv_columns`). Used by `add_features()` → `hmm` engine. Tested in `tests/test_indicator_config.py`. |
| `utils/data_types.py` | 200 | Contains `PerformanceMetrics` — return type in `performance_metrics.py` (5 references). Other types (`FuturesData`, `HMMState`, `Trade`, etc.) are only re-exported via `utils/__init__.py` but not used by the active pipeline. Conservative approach: keep the whole file. |
| `data_processing/feature_engineering.py` (minus `FeatureEngineer`) | 851 | `add_features()` is the live path for `hmm` engine. `FeatureEngineer` class (lines 852–1299) is dead. |

### Previously claimed dead — actually alive

The original draft of this ADR incorrectly listed these as dead:

- **`data_processing/technical_indicators.py`** — module-level dependency of the live `add_features()` function. Deleting would break `--engine hmm`.
- **`utils/data_types.py` (fully)** — `PerformanceMetrics` is the return type of `calculate_performance()` in `performance_metrics.py`, which is used by all three engines.

### Does not exist

- **`data_processing/feature_selection.py`** — listed in the original draft as ~500 dead lines, but this file was deleted in commit `34ab02b` ("Mega Refactor"). The `FeatureEngineer` class has lazy imports from it (which would fail at runtime), confirming the class is dead.

## Deletion test

For each module: imagine deleting it. Does complexity vanish or migrate?

- **All pass**: zero callers in any engine path would need changes. Complexity vanishes.
- The only risk is external callers using the library directly — but the package was just renamed, so there are no external callers yet.
- `backtesting/__init__.py` and `utils/__init__.py` re-export from dead modules. Both must be updated to remove dead re-exports (otherwise `ImportError` on package import).

## Decision

Delete all dead modules in one commit. Keep:

- `regime/` (all 5 files)
- `data_processing/csv_auto_detect.py`, `csv_parser.py`, `csv_format_detector.py`, `data_validation.py`
- `data_processing/feature_engineering.py` (trimmed: keep `add_features()` + helpers, remove `FeatureEngineer` class at lines 852–1299)
- `data_processing/messina_features.py`
- `data_processing/technical_indicators.py` (alive — used by `hmm` engine)
- `backtesting/performance_metrics.py` (trimmed: keep `calculate_sharpe_ratio`, `calculate_drawdown_metrics`, and all functions that call or return `PerformanceMetrics`)
- `utils/data_types.py` (alive — contains `PerformanceMetrics`)
- `utils/logging_config.py`
- `cli.py`, `__init__.py`

Additionally:
- Delete `hmm_models/` directory entirely
- Delete `model_training/` directory entirely
- Update `backtesting/__init__.py` to remove re-exports of `performance_analyzer`, `bias_prevention`, and `backtesting/utils`
- Update `utils/__init__.py` to remove re-exports of `config` and dead types from `data_types`
- Update `data_processing/__init__.py` to remove re-exports from `technical_indicators` if `technical_indicators` is kept (it should be — it's alive)

## Consequences

**Positive:**
- Package drops from ~11,350 to ~5,300 lines (~53% reduction)
- Every remaining module earns its keep (passes deletion test)
- AI navigation stops bouncing through dead modules
- Install size reduced for `pip install`

**Negative:**
- If someone was using `hmm_models/` or `model_training/` directly, their code breaks
- Some backtesting helpers (transaction costs, bias detection) would need re-adding if future work needs them
- `utils/data_types.py` retains types (`FuturesData`, `HMMState`, etc.) not used by the active pipeline — a conservative trade-off to avoid breaking `PerformanceMetrics`

## Related

- [[ADR-002]] Engine seam deepening — easier after dead modules are gone
- [[ADR-003]] Feature engineering trim — `FeatureEngineer` class removal is part of this ADR
- [[ADR-004]] CLI data loading seam — independent of this ADR

## Note for future planning: Why hmm_models/ was superseded

The dead `hmm_models/` + `model_training/` (~4,060 lines) represent an earlier train-once, evaluate, persist paradigm that was replaced by the walk-forward refit-every-N-bars design in `regime/hmm_adapter.py` (217 lines). Three statistical choices in the live adapter are superior for the actual use case:

### 1. Covariance type: `diag` (live) is correct for high-dimensional features

The dead code defaults to `covariance_type="full"`. With ~44 features for the `hmm` engine, `full` covariance requires 990 parameters per state × 3 states = 2,970 covariance parameters alone. On typical datasets of ~2,500 bars, this is under-determined and would overfit to noise in the covariance structure. The live adapter's `diag` covariance estimates only 44 parameters per state — statistically appropriate for this feature dimensionality.

### 2. Walk-forward label continuity: `_match_states()` is essential

The dead architecture has no mechanism for maintaining state label consistency across consecutive fits. HMM state indices are arbitrary — refitting on bars [0:500] vs [0:600] can swap state 0 and state 2. The live `_match_states()` solves this via nearest-neighbor matching in mean space. Without this, walk-forward backtesting produces incoherent regime labels. The dead architecture was fundamentally incompatible with the walk-forward paradigm.

### 3. Fast refits: pragmatic for ~100-fits-per-run workload

The dead code uses `n_iter=100`, `tol=1e-6` (strict convergence). The live adapter uses `n_iter=30`, `tol=1e-4`. The walk-forward loop refits ~100 times per backtest run. 30 iterations is typically sufficient for EM convergence on financial data, and the aggregate of many fresh refits is more robust than a single precise fit on stale data.

### What was lost

- **BIC/AIC model selection** — useful for choosing `n_states`, could be extracted from `gaussian_hmm.py:evaluate_model_quality()` as a standalone utility
- **Cross-validation with `TimeSeriesSplit`** — never wired into the pipeline but valuable for research
- **Rich state diagnostics** — human-readable descriptions, financial interpretation, transition analysis, persistence statistics
- **Model persistence** — pickle-based save/load with metadata; irrelevant for walk-forward but useful for train-once scenarios
- **GMMHMMModel** — Gaussian Mixture emissions for more complex within-state distributions

If any of these become needed, extract the relevant functions as standalone utilities rather than reviving the class hierarchy.
