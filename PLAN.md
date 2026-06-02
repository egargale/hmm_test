# Implementation Plan: hmm_test → Agent Skill

## Decisions summary

| # | Decision | Resolution |
|---|----------|------------|
| 1 | Skill IS the project root | SKILL.md at project root |
| 2 | Flat scripts/ directory | No; keep subdirectory structure under `scripts/` |
| 3 | Python packaging | Keep `pyproject.toml` + `uv`; rename `src/` → `scripts/` |
| 4 | Skill scope | Layered: SKILL.md covers core workflow; `references/` for deep dives |
| 5 | Description trigger | Narrow regime-detection trigger; SKILL.md routes to references |
| 6 | Output contract | Roan contract base + `hmm_test_extras` block |
| 7 | CLI layer | Removed; one thin `scripts/regime.py` entry point |
| 8 | SKILL.md body | Full workflow reference (~4000 tokens) with strong reference index |
| 9 | File map | Keep backtesting, data_processing, hmm_models, model_training, utils; strip rest |
| 10 | `__init__.py` | Minimal (docstring-only); explicit imports preferred |
| 11 | regime.py approach | hmm_test-native: imports from modules, threshold/HMM dual mode |
| 12 | Dependencies | Core: numpy, pandas, scikit-learn, scipy, hmmlearn. Optional: yfinance, dask, daft. Dev: pytest, ruff |
| 13 | Visualization code | Removed entirely; agent renders locally from JSON output |
| 14 | Processing engines | Removed; streaming logic absorbed into `data_processing/csv_parser.py` |
| 15 | Module map | 5 subpackages + regime.py entry point |
| 16 | Tests | Integration-only (2-3 pipeline tests) for initial ship |
| 17 | Implementation order | 9-step sequence (see below) |

## Target structure

```
hmm_test/
├── AGENTS.md                   # Agent guidance
├── CONTEXT.md                  # Domain language & terminology
├── LICENSE
├── PLAN.md                     # this file
├── README.md
├── SKILL.md                    # Agent-facing skill definition
├── USAGE.md                    # CLI reference and usage guide
├── pyproject.toml              # Package config (hmm-futures-analysis)
├── .python-version
├── run.sh                      # Self-bootstrapping entry point for skill consumers
├── test_data/
│   ├── test_futures.csv
│   ├── BTC.csv
│   ├── SPY.csv
│   └── KO.csv
├── hmm_futures_analysis/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (hmm-regime console script)
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── performance_metrics.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── csv_auto_detect.py
│   │   ├── feature_engineering.py
│   │   ├── messina_features.py
│   │   └── technical_indicators.py
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── engine_protocol.py   # RegimeEngine protocol + ENGINE_REGISTRY
│   │   ├── engine_configs.py     # Per-engine config dataclasses (ADR-0011)
│   │   ├── duration_forecast.py  # Weibull and Cox PH survival analysis
│   │   ├── markov_chain.py
│   │   ├── pipeline.py
│   │   ├── walk_forward.py
│   │   └── engines/
│   │       ├── __init__.py
│   │       ├── threshold.py
│   │       ├── hmm_generic.py
│   │       ├── hmm_messina.py
│   │       ├── robust_hmm.py      # Outlier-resistant estimation
│   │       ├── fshmm.py           # Feature saliency selection
│   │       ├── _hmm_engine.py     # Shared HMM fitting, BIC, state matching
│   │       └── _hmm_pipeline.py   # Shared HMM walk-forward classification
│   └── utils/
│       ├── __init__.py
│       ├── data_types.py
│       └── logging_config.py
├── references/
│   ├── hmm_theory.md
│   ├── feature_engineering.md
│   ├── backtesting_detail.md
│   ├── configuration.md
│   ├── troubleshooting.md
│   └── hmm_silent_failure.md
├── docs/
│   ├── adr/
│   │   ├── README.md
│   │   ├── 0001-three-independent-engines.md
│   │   ├── 0002-same-repo-dual-distribution.md
│   │   ├── 0003-engine-self-containment.md
│   │   ├── 0004-cli-data-loading-seam.md
│   │   ├── 0005-pca-in-model-layer.md
│   │   ├── 0006-bic-state-count-selection.md
│   │   ├── 0007-hysteresis-dwell-time-filters.md
│   │   ├── 0008-excise-dead-weight.md (+ postscript: later follow-ups)
│   │   ├── 0009-deepen-engine-seam.md
│   │   ├── 0010-trim-feature-engineering.md
│   │   ├── 0011-engine-dispatch-consolidation.md
│   │   ├── 0012-pipeline-run-decomposition.md
│   │   ├── 0013-fshmm-engine.md
│   │   ├── 0014-per-phase-timing-instrumentation.md
│   │   ├── 0015-regime-duration-forecasting.md
│   │   └── 0016-robust-hmm-engine.md
│   ├── agents/
│   │   ├── domain.md
│   │   ├── issue-tracker.md
│   │   └── triage-labels.md
│   └── research/
│       └── technology-scan-2026-05.md
│   └── reviews/
│       ├── review-1-correctness.md
│       └── review-2-integration.md
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   └── 16× unit tests (messina, filters, packaging, pipeline types, etc.)
│   └── integration/
│       └── 10× integration tests (engines, PCA, whipsaw, profiling, etc.)
├── .out-of-scope/
│   ├── ensemble-engine.md
│   ├── gh-hmm-engine.md
│   ├── hdp-hmm-engine.md
│   ├── student-t-standalone-engine.md
│   ├── wasserstein-hmm.md
│   └── PRD_HMM_BEST.md         # Archived original product requirements
└── logs/                         # Per-run profiling output
```

## Completed Steps

All 9 implementation steps from the original plan are complete. Key milestones:

| Step | Description | Commit | Status |
|------|-------------|--------|--------|
| 1 | Restructure directories (`src/` → `scripts/` → `hmm_futures_analysis/`) | `4d60098` | ✅ Done |
| 2 | Clean `pyproject.toml` | `4d60098` | ✅ Done |
| 3 | Fix cross-module imports (all relative) | `4d60098` | ✅ Done |
| 4 | Absorb streaming into `csv_parser.py` | `4d60098` | ✅ Done |
| 5 | Write `cli.py` | `4d60098`, `0070e96` | ✅ Done |
| 6 | Write `SKILL.md` | `04978c9` | ✅ Done |
| 7 | Write `references/` | `04978c9` | ✅ Done |
| 8 | Write integration tests | `a5b3099`+ | ✅ Done |
| 9 | Verify | all PRs | ✅ Done |

### Additional completed work (post-plan)

| Change | Commit | PR/Issue |
|--------|--------|----------|
| Excise dead weight modules (~5,650 lines) | `1e2736b` | PR #9, ADR-0008 |
| Implement RegimeEngine protocol | `0b13329` | ADR-0009 |
| Documentation update (CONTEXT, README, SKILL, PLAN) | `07f8043` | — |
| HMM engines drive top-level pipeline stats | `3e4d7da` | Issue #10 |
| Engine self-containment ADR | `a020239` | PR #13, ADR-0003 |
| Integration test: engines produce different regimes | `274e401` | PR #11 |
| Delete unused FeatureEngineer class | `e47ebf5` | PR #14, ADR-0010 |
| Messina feature set refined to 18 indicators (19 cols incl log_ret) | `68863fe` | — |
| CLI data loading seam (`load_prices()`) | `0070e96` | ADR-0004 |
| PCA whitening in model layer | `57d6d19` | PR #18, ADR-0005 |
| BIC-based state count selection (`--n-states auto`) | `1466099` | Issue #17, ADR-0006 |
| Hysteresis/dwell-time whipsaw filters | `a9e8523` | Issue #19, ADR-0007 |
| robust_hmm engine (Huber IRLS + MinCovDet) | `af08b7f` | Issue #24 |
| fshmm engine (feature saliency EM) | `3e939af` | Issue #24 |
| Cox PH duration forecasting | `6683429` | Issue #29 |
| Weibull duration forecasting | `ecefe4e` | Issue #29 |
| Per-phase pipeline timing instrumentation | `b266abd` | Issue #37 |
| Engine config dataclasses + dispatch consolidation | `74ccfab` + `a5d86ee` | ADR-0011, Issue #53 |
| Pipeline `run()` decomposition helpers | `f0c36cf` + `9756704` | ADR-0012 |
| Delete 3 dead CSV modules (csv_parser, csv_format_detector, data_validation) | `e0b772e` | Issue #54 |
| Delete dead functions in performance_metrics.py + technical_indicators.py | `6cc0c79` | Issue #55 |
| Remove dead dataclasses from utils/data_types.py | `bbbb91c` | Issue #56 |
