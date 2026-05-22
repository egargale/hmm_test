# Progress

## 2026-05-22: Feature Gap Scout Complete

**Task:** Investigate the feature engineering gap between Messina and Generic HMM modes.

**Status:** ✅ Complete

**Result written to:** `feature_gap_scout.md`

### Key findings:
1. **Generic mode** produces ~44 features across 13 categories (returns, moving averages, volatility, momentum, volume, trend, patterns, enhanced indicators, time features)
2. **Messina mode** produces exactly 12 features (log_ret, SMA200, SMA13, ATR20, ADX14, DI+/-, ADX slope, VSTOP, VSTOP trend, price ratios)
3. **Same-feature-name ≠ same calculation** — ADX/DI in Messina uses Wilder's smoothing (exponential), generic mode uses SMA from `ta` library
4. **HMM model is completely feature-agnostic** — it receives a standardized 2D float64 array; all feature handling is upstream in `hmm_adapter.py`
5. **Feature mode selection** happens in `hmm_adapter.py:run_hmm_regime()` based on `use_messina` bool
6. **Walk-forward backtest does NOT use HMM states** — it uses threshold-based regimes. HMM is a separate analytical block in the JSON output
7. **Fair comparison is feasible** but requires controlling for: (a) differing dimensionality (44 vs 12), (b) Wilder's vs SMA calculation differences, (c) separate pipeline calls per mode
