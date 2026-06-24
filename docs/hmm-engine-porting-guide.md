# Handoff: HMM Engine Porting to Forecast Project

**Date**: 2026-06-24
**From**: Session evaluating five HMM regime engines for porting
**To**: Fresh agent that will execute the lean port

---

## What was decided

1. **Two engines to port**: `threshold` (close-only, ~1s) and `robust_hmm` (HMM + Huber/MCD, ~80s). See evaluation deck above for full rationale.

2. **ADR-0025 implemented**: HMM engines now use the HMM's internal `model.transmat_` (remapped to 3×3 regime space) instead of empirical count matrix for transition probabilities. Forecasts reduced to 1-step only. See `docs/adr/0025-hmm-internal-transmat-forecast.md`.

3. **Skipped engines**: `hmm` (degeneracy on low-vol equities), `messina` (negative avg Sharpe, no advantage), `fshmm` (100% degenerate on equities). Reference ADR-0013, ADR-0016, ADR-0018 for documented failure modes.

---

## Reference artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Domain glossary | `CONTEXT.md` | Canonical terminology: engine, regime, transition matrix, verdict, etc. |
| ADR-0025 | `docs/adr/0025-hmm-internal-transmat-forecast.md` | HMM internal transmat for forecasts, 1-step only |
| ADR-0016 | `docs/adr/0016-robust-hmm-engine.md` | robust_hmm design: Huber IRLS/MCD, architecture |
| ADR-0018 | `docs/adr/0018-degenerate-fit-detection-recovery.md` | Degeneracy auto-recovery, failure modes, minimum data recs |
| ADR-0005 | `docs/adr/0005-pca-in-model-layer.md` | PCA whitening in HMM fit pipeline |
| ADR-0006 | `docs/adr/0006-bic-state-count-selection.md` | BIC auto n_states selection |
| ADR-0007 | `docs/adr/0007-hysteresis-dwell-time-filters.md` | Walk-forward whipsaw filters |
| ADR-0008 | `docs/adr/0008-excise-dead-weight.md` | What was already removed; context for what's alive |
| ADR-0020 | `docs/adr/0020-hmm-engine-base-class-enrich-info-consolidation.md` | HMMEngineBase structure |
| Engine suitability | `CONTEXT.md` §Engine suitability | Per-engine best-for/weakness table with evidence notes |

---

## Exact files for the lean port

### ALWAYS port (threshold engine — minimal, deterministic, no walk-forward refit)

These give you a working threshold-based regime classifier with transition matrix, 1-step forecast, walk-forward backtest, and optional duration forecast:

```
# Core engine (~40 LOC)
regime/engines/threshold.py
    └── regime/markov_chain.py           # classify_regimes, build_transition_matrix,
                                             compute_stationary_distribution,
                                             compute_persistence_diagonal,
                                             compute_signal, forecast_n_steps
    └── regime/engine_protocol.py        # ClassifyResult, ClassifyOutput dataclasses
                                             (simplify: drop ENGINE_REGISTRY, _LazyRegistry,
                                              resolve_engine, HMM_ENGINES)

# Walk-forward backtest (~200 LOC)
regime/walk_forward.py
    └── backtesting/performance_metrics.py  # calculate_sharpe_ratio, calculate_drawdown_metrics
        └── (drop get_logger dependency — replace with stdlib logging or no-op)

# Duration forecast — engine-agnostic, optional (~250 LOC)
regime/duration_forecast.py
    └── scipy (weibull), lifelines (cox, optional)

# Result assembly (~150 LOC)
regime/_result_assembly.py
    └── regime/markov_chain.py
```

Total for threshold-only port: **~640 LOC** (not counting numpy/pandas/scipy which are external deps).

### ADDITIONALLY port for robust_hmm engine

```
# Engine class (~35 LOC)
regime/engines/robust_hmm.py
    └── regime/engines/_hmm_engine.py       # HMMEngineBase, _classify_hmm_slice,
                                                _fit_hmm_on_slice, _huber_correction,
                                                _mcd_correction, robust_fit_gaussian_hmm
    └── regime/engines/_feature_set.py      # GenericFeatureSet, MessinaFeatureSet
        └── data_processing/feature_engineering.py  # add_features (~850 LOC)
            └── data_processing/technical_indicators.py  # indicator config (~80 LOC)
                └── (drop get_logger → stdlib logging)
            └── ta (pypi: `ta` — SMAIndicator, EMAIndicator, BollingerBands, RSIIndicator)
        └── data_processing/messina_features.py  # NOT needed for robust_hmm (uses generic features)
            → SKIP this file; remove its import from _feature_set.py

# Walk-forward HMM pipeline (~300 LOC)
regime/engines/_hmm_pipeline.py         # _hmm_classify_pipeline, _walk_forward_classify,
                                            select_n_states (BIC), _check_degenerate

# State mapping (~70 LOC)
regime/engines/_state_mapping.py        # build_label_map, _match_states, _remap_to_prev_states
```

Additional for robust_hmm: **~1,370 LOC** (plus external deps: hmmlearn, sklearn, ta).

**Grand total**: ~2,010 LOC for both engines + shared infrastructure.

---

## What to SKIP / simplify for a lean port

### Skip entirely (not needed for either engine)

| File | Why skip |
|------|----------|
| `regime/engines/fshmm.py` | 100% degenerate on equities, complex custom EM |
| `regime/engines/hmm_generic.py` | Degeneracy on low-vol; robust_hmm covers same feature set |
| `regime/engines/hmm_messina.py` | Negative avg Sharpe, no advantage over generic |
| `data_processing/messina_features.py` | Only needed by messina engine |
| `data_processing/ticker_cache.py` | yfinance caching — not needed for CSV-based port |
| `data_processing/csv_auto_detect.py` | CSV loading — port project uses its own data pipeline |
| `regime/pipeline.py` | Full orchestration; inline only the `_build_markov_stats` logic |
| `regime/regime_transitions.py` | Transition events — nice-to-have, not forecast-critical |
| `regime/engine_configs.py` | Config dataclasses; pass params directly |
| `cli.py`, `eval.py`, `presenter.py` | CLI/presentation layer |
| `utils/logging_config.py` | Replace with `logging.getLogger(__name__)` |
| `utils/data_types.py` | Drop; `PerformanceMetrics` type is unused at runtime |

### Simplify in-place

| File | Simplification |
|------|---------------|
| `engine_protocol.py` | Keep only `ClassifyResult`, `ClassifyOutput`. Drop `ENGINE_REGISTRY`, `_LazyRegistry`, `resolve_engine()`, `HMM_ENGINES`, `RegimeEngine` protocol. |
| `_feature_set.py` | Remove `MessinaFeatureSet` and its import of `messina_features`. Keep only `GenericFeatureSet`. |
| `backtesting/performance_metrics.py` | Drop `get_logger` import. Replace `logger.debug(...)` with `logging.getLogger(__name__).debug(...)` or just remove the debug lines. |
| `_hmm_engine.py` | Remove `MessinaFeatureSet` reference (in `HMMEngineBase.featureset` default — already `GenericFeatureSet`). Remove `use_messina` legacy flag. |

### External dependencies

```
numpy, pandas           # both engines (unavoidable)
scipy                   # duration_forecast (weibull fit; optional)
hmmlearn                # robust_hmm (GaussianHMM)
scikit-learn            # robust_hmm (PCA, MinCovDet)
ta                      # robust_hmm feature engineering (SMA, EMA, RSI, Bollinger, etc.)
lifelines               # duration_forecast cox model (optional, heavy)
```

---

## Key architectural notes for the porting agent

1. **Threshold is close-only, robust_hmm requires OHLCV**. The port project must route data accordingly. Threshold classifies on a `pd.Series` of returns; robust_hmm on a `pd.DataFrame` with open/high/low/close/volume columns.

2. **The transition matrix is 3×3 regime-space, not HMM latent-state space.** `_classify_hmm_slice()` aggregates `model.transmat_` from K×K to 3×3. This was implemented in this session (ADR-0025). Do not revert to empirical count matrix for HMM engines.

3. **Walk-forward backtest reuses pre-computed regime labels.** The `walk_forward.py` module takes a `regimes` array (and optional `posteriors` for hysteresis) — it does NOT call any engine. This separation is by design (ADR-0017, ADR-0022).

4. **Degeneracy auto-recovery**: `_hmm_classify_pipeline()` automatically downgrades to `n_states=2` when a 3-state fit collapses (<5% of bars in any state). The 2-state model maps to bear/bull only (no sideways). Per ADR-0018 amendment.

5. **Duration forecast is engine-agnostic.** It takes a `regimes` array — any engine's output works. Weibull is default (scipy only); Cox PH requires lifelines.

6. **PCA is opt-in per engine config.** robust_hmm defaults to `pca_variance=0.90` (evidence from ADR-0019 sweep). threshold has no PCA (single feature).

---

## Suggested skills

| Skill | When to use |
|-------|------------|
| `develop-adr` | If any architectural decisions arise during porting (e.g., changing the forecast horizon, adding a new feature set) |
| `code-simplifier` | After porting, to clean up any import chains or unused code paths that survived the trim |
| `tdd` | If writing tests for the ported engines (red-green-refactor on the ported code) |
| `diagnose` | If the ported robust_hmm produces degenerate fits, crashes, or timing anomalies |
| `ponytail` | For the port itself — the goal is the minimum code that works. Every file that can be skipped should be. |
| `review` | After porting, to verify the ported code matches the original behavior |
| `lsp-navigation` | During porting, to trace import chains and verify no missing dependencies |
