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
├── SKILL.md                    # Agent-facing skill definition
├── AGENTS.md                   # Agent guidance
├── CONTEXT.md                  # Domain language & terminology
├── pyproject.toml              # Package config (hmm-futures-analysis)
├── .python-version
├── LICENSE
├── PLAN.md                     # this file
├── run.sh                      # Self-bootstrapping entry point for skill consumers
├── test_data/
│   ├── test_futures.csv
│   ├── BTC.csv
│   └── sample_ohlcv.csv
├── hmm_futures_analysis/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (hmm-regime console script)
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── performance_metrics.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── csv_parser.py
│   │   ├── csv_format_detector.py
│   │   ├── csv_auto_detect.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── messina_features.py
│   │   └── technical_indicators.py
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── engine_protocol.py  # RegimeEngine protocol + ENGINE_REGISTRY
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── threshold.py
│   │   │   ├── hmm_generic.py
│   │   │   ├── hmm_messina.py
│   │   │   └── _hmm_shared.py
│   │   ├── hmm_adapter.py      # Legacy HMM adapter (deprecated)
│   │   ├── markov_chain.py
│   │   ├── pipeline.py
│   │   └── walk_forward.py
│   └── utils/
│       ├── __init__.py
│       ├── data_types.py
│       └── logging_config.py
├── references/
│   ├── hmm_theory.md
│   ├── feature_engineering.md
│   ├── backtesting_detail.md
│   ├── configuration.md
│   └── troubleshooting.md
├── docs/
│   ├── adr/
│   │   ├── 0001-three-independent-engines.md
│   │   ├── 0002-same-repo-dual-distribution.md
│   │   ├── 0003-engine-self-containment.md
│   │   └── 0004-cli-data-loading-seam.md
│   ├── architecture/
│   │   ├── 001-excise-dead-weight.md
│   │   ├── 002-deepen-engine-seam.md
│   │   └── 003-trim-feature-engineering.md
│   └── agents/
│       ├── domain.md
│       ├── issue-tracker.md
│       └── triage-labels.md
└── tests/
    ├── conftest.py
    ├── test_engine_independence.py
    ├── test_excise_dead_weight.py
    ├── test_feature_engineering.py
    ├── test_indicator_config.py
    ├── test_load_prices.py
    ├── test_messina_features.py
    ├── test_messina_integration.py
    ├── test_packaging.py
    ├── test_regime_contract.py
    ├── test_regime_engine.py
    └── test_regime_pipeline.py
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
| Excise dead weight modules (~5,650 lines) | `1e2736b` | PR #9, ADR-001 |
| Implement RegimeEngine protocol | `0b13329` | ADR-002 |
| Documentation update (CONTEXT, README, SKILL, PLAN) | `07f8043` | — |
| HMM engines drive top-level pipeline stats | `3e4d7da` | Issue #10 |
| Engine self-containment ADR | `a020239` | PR #13, ADR-0003 |
| Integration test: engines produce different regimes | `274e401` | PR #11 |
| Delete unused FeatureEngineer class | `e47ebf5` | PR #14, ADR-003 |
| Messina feature set refined to 18 indicators (19 cols incl log_ret) | `68863fe` | — |
| CLI data loading seam (`load_prices()`) | `0070e96` | ADR-0004 |
