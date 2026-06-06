> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: AST-backed except where explicitly marked inferred

# System Boundaries & Code Locations

## 1. CLI Entrypoint

- **Code path**: `hmm_futures_analysis/cli.py` (593 lines)
- **Entry point**: `hmm-regime` console script (defined in pyproject.toml)
- **Purpose**: Parse CLI args (`--csv`, `--ticker`, `--engine`, `--json`, etc.), build engine config dataclasses, dispatch `pipeline.run()`, format `PipelineResult._asdict()` to stdout JSON or human-readable.
- **Hotness**: 🔴 24 changes (high risk)
- **Key exports**: `main()`, `_build_engine_config()`

## 2. Regime Pipeline & Orchestration

- **Code path**: `hmm_futures_analysis/regime/` (excluding `engines/`)
- **Files**: 6 modules (~1200 lines)
- **Purpose**: Central orchestration hub. Consumes engine output (regimes, posteriors), computes Markov chain statistics (transition matrix, stationary, persistence diagonal, signal), runs walk-forward backtest, extracts regime transitions, and assembles the full PipelineResult output block.
- **Hotness**: 🔴 pipeline.py 35 changes (highest), engine_protocol.py 18 changes, walk_forward.py 14 changes

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
- **Shared base**: `_hmm_engine.py` provides `HMMEngineBase(ABC)` with default `run_classify()` — all 4 HMM engines inherit from it.
- **Pipeline helpers**: `_hmm_pipeline.py` provides `_fit_hmm_on_slice()`, `select_n_states()` (BIC), `_remap_states_by_mean()`, `_build_classify_slice()`
- **Hotness**: All medium risk (7-13 changes each)

| Engine | File | Lines | Feature Set | Differentiator |
|--------|------|-------|-------------|---------------|
| threshold | `threshold.py` | ~35 | Close-only | Vectorized, no HMM, fast |
| hmm_generic | `hmm_generic.py` | ~40 | ~50 SMA-based | Standard HMM with PCA |
| hmm_messina | `hmm_messina.py` | ~35 | 19 Wilder's | HMM with Messina features |
| robust_hmm | `robust_hmm.py` | ~55 | ~50 SMA-based | Huber IRLS or MCD outlier resistance |
| fshmm | `fshmm.py` | ~280 | ~50 SMA-based | Feature Saliency learning during EM |

## 4. Data Processing & Feature Engineering

- **Code path**: `hmm_futures_analysis/data_processing/` (5 files, ~1500 lines)
- **Modules**: `csv_auto_detect.py`, `feature_engineering.py` (generic), `messina_features.py` (19 Wilder's), `technical_indicators.py` (low-level TA)
- **Fan-in**: `csv_auto_detect` imported by 20 modules (3rd highest)

## 5. Backtesting & Evaluation

- **Code path**: `hmm_futures_analysis/backtesting/` (2 files, 167 lines)
- **Code path**: `hmm_futures_analysis/eval.py` (~200 lines)
- **Components**: `performance_metrics.py` (Sharpe, drawdown), `eval.py` (multi-ticker harness)

## 6. Utilities

- **Code path**: `hmm_futures_analysis/utils/` (3 files, 61 lines)
- **Files**: `__init__.py`, `data_types.py`, `logging_config.py`

## 7. Hyperparameter Sweep Scripts

- **Code path**: `scripts/` (13 files, ~3500 lines) + `hmm_sweep.py` + `hmm_sweep_missing.py`
- **Not part of the importable package** — standalone CLI scripts.
- **Sweeps**: messina (3 phases), fshmm (4 variants), robust_hmm, hmm

## 8. Test Suite

- **Code path**: `tests/` (52 files, ~11000 lines)
- **38 unit tests** + **14 integration tests**
- **CI**: 2 authors, 127 commits in 90 days

## Non-package directories

| Directory | Purpose |
|-----------|---------|
| `test_data/` | CSV price files + eval results for 18 assets |
| `docs/adr/` | 20 architecture decision records |
| `docs/agents/` | Agent workflow docs (issue tracker, triage, domain) |
| `docs/research/` | Technology landscape scan |
| `references/` | Deep-dive docs on HMM theory, configuration, troubleshooting |
| `.out-of-scope/` | PRDs for rejected/explored alternative approaches |
