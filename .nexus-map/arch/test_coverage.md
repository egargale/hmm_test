> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: AST-backed; no tests executed — static coverage map

# Test Coverage (Static)

## Structure

- **55+ test files** (38 unit + 14 integration + 3 new: ticker_cache + CLI flags)
- **~11,500 lines** test code vs. ~10,500 production
- **2 authors**, 129 commits in 90 days

## Coverage by System

| System | Key Test Files | Tests |
|--------|---------------|-------|
| CLI Entrypoint | `test_packaging.py`, `test_reverse_classify_cli.py`, **`test_cli_cache_flags.py`** | ~15 |
| Pipeline | `test_classify_pipeline.py`, `test_pipeline_dataclasses.py`, `test_build_markov_stats.py`, `test_build_engine_info.py`, `test_validate_prices.py`, `test_verdict.py`, `test_duration_forecast.py` | ~35 |
| Walk-forward | `test_walk_forward_classify.py`, `test_walk_forward_degeneration.py`, `test_walk_forward_engine_param.py`, `test_filters.py` | ~15 |
| Engines | 8 unit test files | ~40 |
| Data Processing | `test_messina_features.py`, `test_feature_engineering.py`, `test_indicator_config.py`, `test_load_prices.py` | ~10 |
| **Ticker Cache (NEW)** | **`test_ticker_cache.py`** | **7 classes, 9 tests, 201 lines** |
| Integration | 14 files | 78 classes |
| Packaging | 3 files | ~5 |

## Evidence Gaps

- No direct tests for `technical_indicators.py` in isolation.
- No tests for individual `scripts/` sweep files.
- No benchmark/performance tests.
