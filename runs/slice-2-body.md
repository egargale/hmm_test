## Parent

#111

## What to build

A re-runnable statistical evaluation harness that runs the existing CLI on a 5-ticker × 5-engine panel and asserts the algebraic and statistical invariants documented in the parent PRD. The harness is a single pytest module that shells out to the CLI exactly the way a user does interactively (so the test exercises the same code path), parses the JSON, and asserts on the parsed structure. The slice also adds a `fleiss_kappa` helper, a session-scoped buy-and-hold Sharpe fixture, a `statistical` pytest marker, and a markdown report artifact written to `runs/statistical_report.txt`.

End-to-end behaviour after this slice: running `pytest -m statistical` invokes the CLI 25 times, asserts four invariant classes (forecast algebra, cross-engine agreement, persistence, buy-and-hold delta), and writes a committed-once-then-overwritten report file that mirrors the format of the one-off `runs/report.txt`. The default `pytest` invocation (no `-m`) stays under ~30 seconds because the panel is gated by `pytest.mark.slow`.

## Acceptance criteria

- [ ] A new pytest module under the integration-tests directory (named `test_statistical_evaluation.py` or similar) exists, parametrized over the 5-ticker × 5-engine grid. The class is decorated with `@pytest.mark.slow` and `@pytest.mark.statistical`; the `statistical` marker is registered in `pyproject.toml`.
- [ ] A `fleiss_kappa(counts_matrix)` helper is added (in the test module or a shared helper file) with a docstring that states the formula and reproduces the canonical interpretation table (slight / fair / moderate / substantial / almost perfect).
- [ ] A buy-and-hold Sharpe fixture is added (session-scoped, keyed on ticker) that computes the baseline on each ticker's CSV using the same window the engine uses, using `numpy.diff` (no pandas `.iloc` to avoid the linter ambiguity).
- [ ] The harness asserts the forecast-algebra identity (`forecast.k_step == T^k @ e_current` to `abs(Δ) < 1e-9`) for all 5 engines × 5 tickers × 3 horizons = 75 cells.
- [ ] The harness computes Fleiss' κ from the 5-rater × 5-subject current-regime matrix and asserts κ ≥ 0.0 (the "no worse than chance" floor documented in the parent PRD).
- [ ] The harness asserts the median regime spell length per engine exceeds 3 bars (the random-walk baseline). Median-not-min avoids spurious failures from a single degenerate ticker.
- [ ] The harness asserts the median per-engine buy-and-hold Sharpe delta is ≥ −1.0 (catches the worst observed case, `fshmm` at −1.31, without flaking on `robust_hmm` at −0.10).
- [ ] The harness writes a markdown-style report to `runs/statistical_report.txt` after each run, mirroring the layout of `runs/report.txt` (per-ticker buy-and-hold table, per-engine Sharpe grid, per-engine win-rate grid, per-engine max-drawdown grid, per-engine n-trades grid, per-engine profit-factor grid, and the cross-engine agreement summary). The file is created on first run (and committed) and overwritten on subsequent runs.
- [ ] The full panel runs in under 10 minutes on the dev box. The slow and statistical markers both gate it; `pytest -m "not slow"` is the fast path used by pre-commit / default CI.
- [ ] The 25-cell test uses the existing `run_regime` helper from `tests/conftest.py` and reuses the existing `btc_csv` fixture pattern. Add `spy_csv`, `nvda_csv`, `ko_csv`, `xle_csv` session-scoped fixtures of the same shape if they do not already exist.

## Blocked by

- #112
