> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: AST-backed except where explicitly marked inferred

# System Boundaries & Code Locations

## 1. CLI Entrypoint

- **Code path**: `hmm_futures_analysis/cli.py` (~628 lines)
- **Entry point**: `hmm-regime` console script (defined in pyproject.toml)
- **Purpose**: Parse CLI args (`--csv`, `--ticker`, `--engine`, `--json`, `--cache-dir`, `--refresh`, `--no-cache`, etc.), build engine config dataclasses, dispatch `pipeline.run()`, format output.
- **Hotness**: 🔴 25 changes (high risk)
- **Key exports**: `main()`, `_build_engine_config()`

## 2. Regime Pipeline & Orchestration

- **Code path**: `hmm_futures_analysis/regime/` (excluding `engines/`)
- **Files**: 6 modules (~1200 lines)
- **Hotness**: 🔴 pipeline.py 35 changes (highest), engine_protocol.py 18, walk_forward.py 14

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `pipeline.py` | ~610 | `run()`, `MarkovStats`, `PipelineResult`, verdict logic |
| `engine_protocol.py` | ~100 | `RegimeEngine` Protocol, `ClassifyResult`/`ClassifyOutput`, `ENGINE_REGISTRY` |
| `engine_configs.py` | ~100 | Config dataclasses for 5 engines |
| `walk_forward.py` | ~240 | No-lookahead walk-forward backtest, dwell/hysteresis filters, degenerate recovery |
| `markov_chain.py` | ~150 | `classify_regimes()`, `build_transition_matrix()`, `compute_signal()`, forecasts |
| `regime_transitions.py` | ~50 | `extract_transitions()` for change-point detection |
| `duration_forecast.py` | ~100 | Weibull/Cox survival analysis on regime spell lengths |

## 3. Engine Implementations

- **Code path**: `hmm_futures_analysis/regime/engines/` (7 files, ~1700 lines)
- **Protocol**: All implement `RegimeEngine` with `precompute()` → `classify()` → `run_classify()`
- **Shared base**: `_hmm_engine.py` provides `HMMEngineBase(ABC)` — all 4 HMM engines inherit from it
- **Pipeline helpers**: `_hmm_pipeline.py` provides `_fit_hmm_on_slice()`, `select_n_states()` (BIC), `_remap_states_by_mean()`

| Engine | File | Lines | Feature Set | Differentiator |
|--------|------|-------|-------------|---------------|
| threshold | `threshold.py` | ~35 | Close-only | Vectorized, no HMM, fast |
| hmm_generic | `hmm_generic.py` | ~40 | ~50 SMA-based | Standard HMM with PCA |
| hmm_messina | `hmm_messina.py` | ~35 | 19 Wilder's | HMM with Messina features |
| robust_hmm | `robust_hmm.py` | ~55 | ~50 SMA-based | Huber IRLS or MCD outlier resistance |
| fshmm | `fshmm.py` | ~280 | ~50 SMA-based | Feature Saliency learning during EM |

## 4. Data Processing & Feature Engineering

- **Code path**: `hmm_futures_analysis/data_processing/` (6 files, ~1600 lines)
- **Modules**: `csv_auto_detect.py`, `feature_engineering.py`, `messina_features.py`, `technical_indicators.py`, **`ticker_cache.py`** (NEW)
- **ticker_cache.py**: 96 lines. Provides `get_ticker_data()` on-disk yfinance caching. Supports `--refresh` and `--no-cache`.
- **Fan-in**: `csv_auto_detect` imported by 20 modules (3rd highest)

## 5. Backtesting & Evaluation

- **Code path**: `hmm_futures_analysis/backtesting/` (2 files, 167 lines)
- **Code path**: `hmm_futures_analysis/eval.py` (~190 lines)
- **Note**: `_save_ticker_csv()` removed from eval.py — ticker fetching now delegated to `ticker_cache`

## 6–8. Utilities, Sweep Scripts, Test Suite

| System | Code Path | Lines |
|--------|-----------|-------|
| **Utilities** | `hmm_futures_analysis/utils/` | 3 files, 61 |
| **Sweep Scripts** | `scripts/` + `hmm_sweep.py` | 13 files, ~3500 |
| **Test Suite** | `tests/` | 55+ files, ~11500 |
