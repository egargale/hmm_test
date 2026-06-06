> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: AST-backed; no tests were executed — this is a static coverage map

# Test Coverage (Static)

## Structure

- **52 test files total**: 38 unit + 14 integration
- **~11,000 lines of test code** vs. ~10,000 lines of production code
- **225+ test classes** (unit: 147 classes, integration: 78 classes)
- **2 authors** over 127 commits in 90 days

## Coverage by System

| System | Test Files | Test Classes | Evidence |
|--------|-----------|-------------|----------|
| CLI Entrypoint | `test_packaging.py`, `test_reverse_classify_cli.py`, `test_docs_currency.py` | ~10 | Entry point resolution, reverse-classify flag, docs freshness |
| Pipeline | `test_classify_pipeline.py`, `test_pipeline_dataclasses.py`, `test_build_markov_stats.py`, `test_build_engine_info.py`, `test_validate_prices.py`, `test_verdict.py` | ~35 | Pipeline run, MarkovStats, verdict logic, price validation, engine info assembly |
| Walk-forward | `test_walk_forward_classify.py`, `test_walk_forward_degeneration.py`, `test_walk_forward_engine_param.py`, `test_filters.py` | ~15 | Walk-forward classify, mid-stream degeneration, filter application, param validation |
| Engines | `test_hmm_engine_base.py`, `test_classify_hmm_slice.py`, `test_degenerate_auto_recovery.py`, `test_degenerate_fit.py`, `test_remap_prev_states.py`, `test_engine_filter_defaults.py`, `test_fshmm_delegates_classify.py`, `test_error_paths.py` | ~40 | HMMEngineBase, BIC selection, state remapping, degenerate recovery, auto-recovery |
| Data Processing | `test_messina_features.py`, `test_feature_engineering.py`, `test_indicator_config.py`, `test_load_prices.py` | ~10 | Feature column integrity, TA indicator computation, CSV loading |
| Integration | `test_regime_pipeline.py`, `test_regime_engine.py`, `test_regime_contract.py`, `test_fshmm.py`, `test_robust_hmm.py`, `test_messina_integration.py`, `test_pca_whitening.py`, `test_eval_mode.py`, `test_real_data_suite.py`, `test_profile_pipeline.py`, `test_threshold_hdb_sideways.py`, `test_engine_independence.py`, `test_run_regime_timeout.py` | ~78 | End-to-end pipeline, engine contract compliance, PCA integration, eval mode, real data |
| Packaging | `test_packaging.py`, `test_no_stale_src.py`, `test_tqdm_dependency.py` | ~5 | Import validation, src/ absence, tqdm in deps |
| Sweep | `test_sweep_tqdm.py` | ~5 | tqdm progress bar correctness across 5 sweep phases |

## Evidence Gaps

- **No explicit test for `duration_forecast.py`** unit function `compute_duration_forecast()` directly — only tested indirectly through `test_duration_forecast.py` which tests pipeline integration.
- **No tests for individual scripts/ files** (sweep scripts are manual execution tools).
- **No performance/benchmark tests** beyond the integration test suite.
- **No test for `technical_indicators.py`** in isolation — tested only through feature engineering integration.

## Test Infrastructure

- **Runner**: pytest with `--strict-markers` and `-m "not slow"` default
- **Fixture**: `tests/conftest.py` provides `sample_ohlcv` DataFrame fixture (imported by 9 test modules)
- **Markers**: `slow` for integration tests against real data
- **Config**: `pyproject.toml` `[tool.pytest.ini_options]` with `pythonpath = ["."]`
