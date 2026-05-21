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
├── SKILL.md                    # NEW: agent-facing skill definition
├── pyproject.toml              # MODIFIED: stripped dependencies
├── .python-version
├── LICENSE
├── PLAN.md                     # this file
├── test_data/
│   ├── test_futures.csv
│   └── sample_ohlcv.csv
├── scripts/
│   ├── regime.py               # NEW: thin entry point
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── strategy_engine.py
│   │   ├── performance_analyzer.py
│   │   ├── performance_metrics.py
│   │   ├── bias_prevention.py
│   │   └── utils.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── csv_parser.py        # MODIFIED: absorbed streaming logic
│   │   ├── csv_format_detector.py
│   │   ├── csv_auto_detect.py   # yfinance + CSV auto-detect
│   │   ├── data_validation.py
│   │   └── feature_engineering.py
│   ├── hmm_models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gaussian_hmm.py
│   │   ├── gmm_hmm.py
│   │   └── factory.py
│   ├── model_training/
│   │   ├── __init__.py
│   │   ├── hmm_trainer.py
│   │   ├── inference_engine.py
│   │   └── model_persistence.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── data_types.py
│       └── logging_config.py
├── references/
│   ├── hmm_theory.md           # NEW
│   ├── feature_engineering.md  # NEW
│   ├── backtesting_detail.md   # NEW
│   ├── configuration.md        # NEW
│   └── troubleshooting.md      # NEW
├── tests/
│   ├── conftest.py             # MODIFIED: update fixtures, drop dead deps
│   ├── test_regime_pipeline.py # NEW: integration tests
│   └── test_regime_contract.py # NEW: JSON contract validation
```

## Removed

- `src/processing_engines/` — streaming absorbed, Dask/Daft dropped
- `src/visualization/` — agent renders locally
- `src/compatibility/` — legacy shim
- `src/pipelines/` — CLI workflow wrapper
- `cli.py`, `cli_simple.py`, `main.py` — CLI layer
- `SRC_DIRECTORY_STRUCTURE_DESIGN.md` — retroactive design doc
- `.omc/` — tool artifacts
- `docs/` — RST replaced by `references/` + `SKILL.md`
- `examples/` — if present

## Implementation steps

### Step 1: Restructure directories

Move `src/` → `scripts/`. Remove dead packages and root-level CLI files.

```
mkdir -p scripts
mv src/backtesting scripts/
mv src/data_processing scripts/
mv src/hmm_models scripts/
mv src/model_training scripts/
mv src/utils scripts/
rm -rf src/processing_engines src/visualization src/compatibility src/pipelines src/
rm -f cli.py cli_simple.py main.py SRC_DIRECTORY_STRUCTURE_DESIGN.md
rm -rf .omc/ docs/ examples/
```

### Step 2: Clean pyproject.toml

Strip to core + optional dependencies. Remove build-system scripts/config for removed modules. Result:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hmm-futures-analysis"
version = "0.1.0"
description = "Hidden Markov Models for futures market regime detection"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "hmmlearn>=0.2.7",
]

[project.optional-dependencies]
yfinance = ["yfinance>=0.2.0"]
dask = ["dask[complete]>=2021.8.0"]
daft = ["daft>=0.3.0"]
dev = ["pytest>=7.0", "ruff"]

[tool.pytest.ini_options]
pythonpath = ["scripts"]
addopts = ["-ra", "--strict-markers"]

[tool.ruff]
target-version = "py39"
```

### Step 3: Fix all cross-module imports

Every `from utils import ...` → update if needed (should still work since `scripts/` is the pythonpath).
Every `from src.xxx` → remove `src.` prefix.
Remove `tqdm` imports from `backtesting/bias_prevention.py` and `strategy_engine.py`.
Remove `click` references.
Update `tests/conftest.py` imports.

### Step 4: Absorb streaming engine into data_processing

Move chunked CSV reading logic from `processing_engines/streaming_engine.py` into `data_processing/csv_parser.py`. The `process_csv()` function should handle chunked reading natively without a factory pattern.

### Step 5: Write `scripts/regime.py`

Entry point. Accepts `--csv`, `--ticker`, `--json`, `--window`, `--threshold`, `--min-train`, `--hmm`.
Pipeline:
1. Load data (CSV auto-detect or yfinance)
2. Compute returns
3. Classify regimes (threshold or HMM)
4. Build transition matrix
5. Compute stationary distribution, persistence, signal
6. Run walk-forward backtest
7. Output: JSON (with `hmm_test_extras`) or pretty terminal

### Step 6: Write `SKILL.md`

YAML frontmatter with `name: hmm-regime-detection` and targeted description.
Body (~4000 tokens): invocation, JSON contract, composition patterns, gotchas, reference index.

### Step 7: Write `references/`

Five files: hmm_theory, feature_engineering, backtesting_detail, configuration, troubleshooting.
Extract from existing `docs/user_guide/complete_guide.rst` and adapt for agent consumption.

### Step 8: Write integration tests

Two test files:
- `test_regime_pipeline.py` — load CSV, classify regimes, compute signal, verify output fields
- `test_regime_contract.py` — verify JSON schema matches contract

### Step 9: Verify

```bash
uv sync
uv run pytest tests/ -v
uv run python scripts/regime.py --csv test_data/test_futures.csv --json
```
