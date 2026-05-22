# HMM Silent Failure — "Not enough clean rows (0)"

## Symptom
JSON output contains:
```json
"hmm": { "available": false, "reason": "Not enough clean rows (0)" }
```
Threshold results (`current_regime`, `signal`, `walk_forward`, etc.) are returned normally.

## Root Cause
`GaussianHMM.fit()` rejected the entire observation matrix. The preprocessing step that builds the feature matrix (typically: returns + volatility features) produced zero valid rows, OR the fitted model returned zero states. This is NOT a crash — the exception is caught internally and the skill degrades gracefully to threshold-only mode.

**Observed on**: KO, BABA (both returning "Not enough clean rows (0)" despite 2514 bars of data). The issue appears to be in the internal feature engineering for returns-based matrices with this version of hmmlearn.

## What to do
1. **Do not treat as a skill malfunction** — threshold results are valid and complete.
2. If HMM output is required for the pipeline, try a different ticker (some tickers work, others don't with this feature set).
3. Use `--no-hmm` to suppress the failure in automated scans.
4. The skill maintainer (egargale/hmm_test) may need to investigate the preprocessing pipeline for return-only matrices.

## Why threshold still works
Threshold mode uses rolling return sums — no model fitting, no feature engineering that could produce an empty matrix. It is completely independent of the HMM path.