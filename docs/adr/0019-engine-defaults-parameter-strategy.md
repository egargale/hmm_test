# Engine default parameters anchored to sweep evidence

Default parameters for the four HMM engines were originally set conservatively (no PCA, high hysteresis). Parameter sweeps across ~10 years of daily data on 4-5 tickers showed that the original defaults were actively harmful on several engines. We now anchor each engine's defaults to the sweep evidence, and make engine configs the single source of truth for defaults rather than CLI flag defaults.

## Changes

| Engine | Parameter | Old default | New default | Evidence |
|---|---|---|---|---|
| `hmm` | `default_hysteresis_delta` | `0.1` | `0.0` | Hysteresis ≥ 0.05 locks to 1 trade; 0.0 enables active trading |
| `messina` | `default_hysteresis_delta` | `0.1` | `0.0` | Same — 1 trade/ticker with any non-zero hysteresis |
| `messina` | `pca_variance` | `None` | `0.95` | Avg Sharpe -0.31 → +0.32; only config with positive avg Sharpe across tickers |
| `robust_hmm` | `default_hysteresis_delta` | `0.1` | `0.0` | Not varied in sweep; 0.0 consistent with hmm/messina evidence |
| `robust_hmm` | `pca_variance` | `None` | `0.90` | Avg Sharpe -0.185 → +0.050; SPY -0.71 → +0.56; ranked #1 out of 12 configs |
| `fshmm` | *(all)* | *(unchanged)* | — | Sweep confirmed defaults near-optimal |
| `hmm` (generic) | `pca_variance` | `None` | `None` | No universal winner; PCA helps specific tickers but hurts others |

CLI defaults for `--dwell-bars` and `--hysteresis` changed from `0` and `0.0` to `"auto"`, which resolves to each engine's `default_dwell_bars` and `default_hysteresis_delta`. Explicit numeric values still override.

## n_states=2 bug fix

The sweep discovered that BIC selecting 2 states caused a crash: `posteriors_all = np.zeros((n, 3))` in `_hmm_pipeline.py` was hardcoded to 3 columns, but 2-state HMMs produced 2-element posteriors. Fixed by always aggregating posteriors to 3 regime buckets (bear/sideways/bull) regardless of `n_states`. Two states now map to bear (lowest mean) and bull (highest mean) with no sideways — a valid regime decomposition for strongly trending assets.

## Unresolved interaction gaps

Sweeps used surgical 1-at-a-time parameter variation due to computational cost (~25s per HMM walk-forward run). Interaction effects between PCA and non-zero hysteresis were not tested. Specifically: robust_hmm with PCA=0.90 + hyst=0.01/0.02 may further improve results. These are future work.

## Sources

- `test_data/eval-results/messina_sweep/HMMMessina_report.md`
- `test_data/eval-results/hmm_sweep/hmm_report.md`
- `test_data/eval-results/fshmm_sweep/fshmm_report.md`
- `test_data/eval-results/robust_hmm_sweep/HMMRobust_report.md`
