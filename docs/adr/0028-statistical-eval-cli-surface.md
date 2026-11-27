# ADR-0028: Statistical eval CLI surface (`--eval-statistical`) and its report contract

**Status**: Accepted

**Date**: 2026-11-27

**Issue**: #114 (Slice 3 — CLI surface for the [[ADR-0027]] statistical harness)

## Context

Issue #114 wires the Slice 2 statistical harness ([[ADR-0027]]) into the CLI:
a `--eval-statistical TICKERS` flag, a `run_eval_statistical(tickers, engines)`
function in the eval module, and `fmt="statistical_table"` in
`presenter.format_eval`. Three things needed a decision:

1. **Data source.** The issue's report is defined on the committed
   `test_data/<TICKER>.csv` harness fixtures, but the CLI's ticker path loads
   via yfinance ([[ADR-0004]]). On a clean clone offline, "the ticker CLI
   path" cannot reproduce the harness report at all.
2. **Report identity.** The issue asks for a report "identical to the one in
   `runs/statistical_report.txt`" (10 sections incl. per-cell wall times and
   floor verdicts) and simultaneously specifies `fmt="statistical_table"` as
   a *two-grid* summary. The 10-section artifact is owned by the harness test
   module, never by product code ([[ADR-0021]], [[ADR-0027]] decision 3).
3. **ADR numbering.** The issue asks for the forecast-algebra invariant "as
   ADR-0025". That number is taken by [[ADR-0025]] (`0025-hmm-internal-transmat-forecast.md`)
   and the invariant already exists as [[ADR-0026]]
   (`0026-forecast-algebra-current-probs-one-hot.md`, merged with Slice 2).
   The ADR index rules forbid reusing or sharing numbers.

### Forecast-algebra invariant (record location)

The invariant the issue wants recorded lives in **[[ADR-0026]]**. In its
terms:

- **(a)** `MarkovStats.current_probs` is the **one-hot unit vector of the
  current regime**, `e_current` — not a row of the transition matrix. The
  prior code path set `current_probs = T[i, :]` (a row of `T`), so
  `forecast_n_steps()` returned `current_probs @ T = T[i, :] @ T =
  T²[i, :]` — a two-step distribution masquerading as a one-step forecast —
  while `next_state_probabilities` returned `T[i, :]`, so the two JSON
  fields the output contract says must be equal silently disagreed
  ([[ADR-0026]], issue #112). The fix sets `current_probs = e_current`, so
  the 1-step forecast is `e_current @ T = T[current_regime, :]` — exactly
  `T[i, :]`, equal to `next_state_probabilities`.
- **(b)** High persistence **masks** the error: with `T[i,i] ≈ 1` both
  `T²[i,:]` and `T[i,:]` are `≈ e_i`, so the forecast-identity check is the
  only reliable detector — which is why the HMM engines (persistence ≈ 1.0)
  hid the bug and the threshold panel exposed it (Δ up to 0.183 on SPY,
  [[ADR-0026]]).
- **(c)** Cross-refs: [[ADR-0011]] — engine dispatch consolidation (the
  seam where `current_probs` and `signal` are produced), [[ADR-0018]] —
  degenerate-fit detection and recovery; its recovered 2-state fit maps its
  two latent states onto the Bear(0)/Bull(2) buckets of the fixed 3×3
  regime space, so no 2×2 ever reaches the forecast code.

## Decision

1. **`run_eval_statistical(tickers, engines=ALL_ENGINES, ...)`** lives in
   `hmm_futures_analysis/eval.py`. Rows have the same shape as
   `run_eval_tickers` plus two fields: `fleiss_kappa` (single panel-wide
   float, identical on every row) and `bnh_delta` (engine walk-forward
   Sharpe minus the buy-and-hold Sharpe on the engine window).
2. **Data source is fixture-first**: each ticker resolves against committed
   `test_data/<TICKER>.csv` fixtures (`./test_data`, then the repository
   root) and only falls back to the yfinance ticker path when no fixture
   exists. Statistical mode is thereby defined on the harness data and
   reproducible offline from a clean clone.
3. **`fmt="statistical_table"`** renders the two grids the issue specifies:
   the cross-engine agreement summary (one row per ticker; engine vote
   columns; a `majority` column) and the buy-and-hold delta grid (one row
   per engine). It prints to **stderr** like `fmt="table"`; `--json` prints
   the row array (with the two new fields) to stdout.
4. **The `majority` column is report-only** descriptive agreement (modal
   vote; `tie` when the top count is shared). It never feeds position
   selection — ensemble/voting engines stay out of scope
   (`.out-of-scope/ensemble-engine.md`).
5. **One metric implementation.** `fleiss_components` / `fleiss_kappa` /
   `buy_and_hold_stats` live in `hmm_futures_analysis/eval.py`; the Slice 2
   harness imports them, so CLI values cannot drift from the harness report
   ([[ADR-0027]]'s definitional contract stays pinned by the harness
   docstring).
6. **`fleiss_kappa` is `None` (JSON `null`) when undefined** — fewer than
   two engines, or chance agreement exactly 1 — instead of raising; the
   shared helper keeps its raising contract for the harness. The presenter
   renders `n/a`.
7. **`run_eval_statistical` invokes `pipeline_run` with the CLI single-run
   filter defaults** (`dwell_bars="auto"`, `hysteresis_delta="auto"` —
   the ADR-0007/ADR-0019 engine defaults the harness cells run with), so
   its `sharpe`/`regime`/`bnh_delta` values match the Slice 2 report.
   Note: `run_eval_csv`/`run_eval_tickers` omit those kwargs and therefore
   run with the `pipeline.run` defaults (`dwell_bars=0`,
   `hysteresis_delta=0.0` — filters off); their rows have never matched
   the CLI single-run or the harness report. That pre-existing divergence
   is **not** fixed here (out of scope for this slice; it would change
   Slice 1 row values) but is recorded for intake.
8. **ADR numbering**: this decision is **ADR-0028** (next unused number).
   No `0025-forecast-algebra-invariant.md` is created: it would collide with
   [[ADR-0025]]'s number and split [[ADR-0026]]'s content across two files.

## Consequences

- `hmm-regime --eval-statistical SPY,NVDA,BTC,KO,XLE` reproduces, offline
  from fixtures and without running the 25-cell CLI panel, the harness's
  deterministic summary content only: the cross-engine agreement grid
  (votes + `majority`), Fleiss' kappa, and the buy-and-hold **delta** grid.
  It does **not** print the harness's per-ticker B&H Sharpe grid (report
  section A) or the walk-forward Sharpe grid (section B) — the delta grid
  is `engine Sharpe − B&H Sharpe` and the two inputs cannot be recovered
  from it.
- The full 10-section `runs/statistical_report.txt` remains the harness's
  deliverable; the CLI deliberately does not duplicate its writer.
- A single-engine `--eval-statistical` run yields `fleiss_kappa: null`
  rather than an error; consumers must treat the field as nullable.
- `run_eval_statistical` rows carry the harness (CLI single-run) values;
  they are not numerically comparable to `run_eval_tickers` rows until the
  pre-existing filter-default divergence above is resolved.

## Related

- [[ADR-0027]] — statistical harness: metrics, floors, report artifact.
- [[ADR-0026]] — forecast algebra: `current_probs` is one-hot (the
  invariant this issue misattributed to ADR-0025).
- [[ADR-0025]] — HMM internal `transmat_` for 1-step forecasts.
- [[ADR-0021]] — presenter stays pure.
- [[ADR-0018]] — 2-state fits keep the 3×3 regime space.
- [[ADR-0011]] — engine dispatch consolidation (the seam where
  `current_probs` and `signal` are produced).
- [[ADR-0004]] — CSV/ticker data loading seam.
