## Problem Statement

A 5-ticker × 5-engine evaluation (SPY, NVDA, BTC, KO, XLE run through all five engines) surfaced two classes of issue that the current test suite does not cover:

1. **A real, low-visibility algebra defect** in `pipeline._build_markov_stats`. The `forecast.1_step` field is built from `current_probs @ T^1`, but `current_probs` is set to `transmat[last_regime]` (already a row of `T`), so the field is actually `T^2[last_regime, :]` — a 2-step distribution, not a 1-step. It is invisible on HMM engines because their persistence is ≈ 1.0 (so `T[i] @ T ≈ T[i]`), but the threshold engine shows deltas of up to **0.18** between `forecast.1_step` and `next_state_probabilities` on BTC, SPY, NVDA, XLE. The SKILL.md contract states these should be the same. No test in `tests/integration/test_regime_contract.py` or `tests/unit/test_build_markov_stats.py` catches this.

2. **No re-runnable statistical evaluation harness.** The 5×5 analysis that produced the bug discovery is a one-off script in `runs/analyze.py`. It is not part of CI, has no assertions, and won't catch regressions on future engine changes. The findings that *do* belong in CI include:
   - Forecast field algebraic consistency (`forecast.k_step == T^k @ e_i`)
   - Cross-engine agreement metric (Fleiss' κ — current 5×5 = −0.027, no better than chance)
   - Walk-forward skill delta vs buy-and-hold baseline
   - Persistence sanity (avg regime spell > 3 bars)
   - Robustness of the threshold engine to its lower persistence (where the off-by-one becomes visible)

The user (a quantitative engineer running the eval interactively) currently has no way to know whether a future engine change *regressed* the regime-detection quality — only that the wire-format contract still holds.

## Solution

Add a single new pytest module under `tests/integration/` that runs all five engines against the five-ticker `test_data/` panel and asserts both the new algebraic invariants and the cross-engine statistical properties. Fix the forecast off-by-one bug in `_build_markov_stats` so that the new contract assertion passes. Document the change in a new ADR.

The evaluation harness reuses the existing CLI seam — it shells out to `./scripts/run.sh --csv … --json --engine …` exactly the way the user did interactively — and asserts on the parsed JSON. This keeps the harness honest: if the CLI breaks, the harness breaks; it does not duplicate any pipeline logic.

## User Stories

1. As a quantitative engineer evaluating engine changes, I want a re-runnable statistical test that fails the build when a regression drops forecast accuracy below a documented baseline, so that engine quality regressions are caught in CI before merging.

2. As a quantitative engineer, I want the new test to assert `forecast.1_step == T[current_regime_index, :]` for every engine and every ticker in the 5-ticker panel, so that the off-by-one defect in `_build_markov_stats` cannot regress.

3. As a quantitative engineer, I want the new test to assert the same identity for `forecast.5_step` and `forecast.20_step` modulo the high-persistence caveat (a relaxation on those two is acceptable and documented), so that the algebraic contract is fully covered.

4. As a quantitative engineer, I want a Fleiss' κ metric computed across the 5×5 current-regime matrix, with an assertion that κ ≥ 0 (no worse than chance), so that future engine changes cannot silently increase inter-engine divergence.

5. As a quantitative engineer, I want a buy-and-hold Sharpe computed for each ticker on the same window, with the walk-forward Sharpe reported as `engine_sharpe − bnh_sharpe`, so that I can read skill contribution at a glance instead of recomputing it.

6. As a quantitative engineer, I want the new test to assert regime-persistence > 3 bars (random-walk baseline) for every (engine, ticker) pair, so that any future "whipsaw" regression in any engine is caught.

7. As a quantitative engineer, I want the new test to be skippable via a pytest marker (e.g. `pytest -m "not slow"`) and a fast mode that exercises only the threshold engine, so that quick local iteration is not blocked by the 8-minute 5×5 panel.

8. As a quantitative engineer, I want a single CLI flag (`--eval-statistical`) that re-runs the same 5×5 panel and prints a markdown table to stderr matching the format of `runs/report.txt`, so that I can reproduce the report from a clean clone.

9. As a maintainer reading the bug report, I want a regression test that runs the threshold engine against `test_data/SPY.csv` and asserts `forecast.1_step == T[current_regime, :]`, so that the specific defect is pinned to a single reproducer.

10. As a maintainer, I want a new ADR (ADR-0025) that documents the forecast algebra and the chosen `current_probs` invariant, so that future contributors do not reintroduce the off-by-one.

11. As a maintainer, I want the new test module to be parameterized over the existing `test_data/*.csv` ticker fixtures (`SPY.csv`, `NVDA.csv`, `BTC.csv`, `KO.csv`, `XLE.csv`) — not the eval-results/ JSON snapshots, so that the test reflects *current* engine behavior, not historical.

12. As a quantitative engineer, I want the test to also report (not assert) the full per-engine / per-ticker Sharpe grid, win-rate, profit-factor, and max-drawdown as a single pytest-html-table-style artifact in `runs/statistical_report.txt`, so that I can compare against `runs/report.txt` after a code change.

13. As a maintainer, I want the new test to not import private pipeline modules (no internal import of `_build_markov_stats` from the integration test), so that the test exercises the *same* code path as the user — via the CLI.

14. As a quantitative engineer, I want the forecast algebra to be applied to a synthetic regime sequence (e.g. `[0,0,0,1,1,2,2,2,2,0]`) where the answer is known by hand, so that the unit-level invariant (`_build_markov_stats` produces a `current_probs` that is a one-hot) is tested without HMM fitting.

15. As a maintainer, I want the off-by-one fix to use a one-hot construction that does not re-introduce dtype issues for `n_states=auto` (where `transmat` shape can be 2×2 if degenerate recovery triggered), so that the fix is robust to the documented mid-stream degeneration path in ADR-0018.

## Implementation Decisions

- **Forecast algebra fix in `_build_markov_stats`**: Replace `current_probs = transmat[last_regime]` with a one-hot for the current regime. The fix must respect the variable shape of `transmat` (3×3 by default, 2×2 after degenerate recovery per ADR-0018). Use a `np.eye` lookup or a `np.zeros` + index assignment. Update `tests/unit/test_build_markov_stats.py` with a regression assertion.

- **New contract assertion in `tests/integration/test_regime_contract.py`**: Add a new test class `TestForecastAlgebra` that asserts `forecast.1_step == T[current_regime_index, :]` for all five engines on a single ticker (BTC). This is the user-facing surface; the unit test in `_build_markov_stats` covers the internal seam.

- **New integration test module `tests/integration/test_statistical_evaluation.py`**: 5-ticker × 5-engine panel. Mark with `@pytest.mark.slow` (matches the convention in `test_regime_contract.py`). Add a `pytest.mark.statistical` marker and register it in `pyproject.toml`.

- **Cross-engine agreement computation**: Fleiss' κ from a 5-rater × 5-subject matrix. Lift into a tiny helper `fleiss_kappa` in the test module (or a new `tests/_stat_helpers.py` if shared). Document the formula in a docstring with the canonical interpretation table.

- **Buy-and-hold baseline**: Compute on the same window used by the engine. Use `numpy.diff / prev` to avoid any pandas-iloc ambiguity. Cache the result in a `pytest.fixture(scope="session")` keyed on ticker.

- **Fast-mode escape hatch**: A CLI flag `--eval-fast` on `hmm-regime` (or a separate pytest collection filter `pytest -k "statistical and not full"`) that limits the panel to threshold-only. This mirrors the `run_phase2_parallel.sh` style in the repo, where there's a precedent for "fast vs full" sweeps.

- **CLI surface `--eval-statistical`**: Reuses the `eval` plumbing in `hmm_futures_analysis.eval`. New function `run_eval_statistical(tickers, engines)` that returns the same dict shape as `run_eval_tickers` but with `fleiss_kappa` and `bnh_delta` columns added. Reuses the existing `format_eval` presenter with a new `fmt="statistical_table"`.

- **ADR-0025**: A new ADR titled "Forecast algebra: invariant for `current_probs`". References ADR-0018 (degenerate recovery) and ADR-0011 (engine dispatch consolidation) since the fix lives in `_build_markov_stats` which those ADRs are silent on.

- **Artifact persistence**: The 5×5 harness writes `runs/statistical_report.txt` after each run, mirroring the layout of `runs/report.txt`. The fixture does the write; no new logging infrastructure is needed.

- **Synthetic unit test for forecast algebra**: In `tests/unit/test_build_markov_stats.py`, add a `test_current_probs_is_one_hot` that calls `_build_markov_stats` on a known regime sequence and asserts `current_probs` equals `np.eye(n)[last_regime]`. This pins the invariant at the function level without HMM fitting.

- **Persistence assertion (median per engine)**: Add `test_persistence_above_random_per_engine` that loops over the 5-ticker panel and asserts that the *median* regime spell length for each engine exceeds 3 bars. Median (not min) is chosen to avoid spurious failures from a single degenerate ticker.

- **Cross-engine agreement threshold**: κ ≥ 0 (no worse than chance) is the floor. Document the current measured value (−0.027) as the "before" baseline in the ADR; do not assert a stricter threshold in the first cut. Rationale: 5 subjects is too few for a statistically meaningful κ to be much above 0 in expectation, so a stricter threshold would be flaky.

- **Buy-and-hold delta reporting**: `engine_sharpe − bnh_sharpe` reported per (engine, ticker) but only *asserted* on the median-per-engine ≥ −1.0 (i.e. engine is not destroying the buy-and-hold by more than 1σ). The current panel has all five engines in the range [−0.10, −1.31], so −1.0 catches the worst case (fshmm at −1.31) without flaking on the rest. Document the choice in ADR-0025.

## Testing Decisions

- **Test the external behavior, not the implementation.** All new integration tests exercise the CLI as a black box (`run_regime(...)` from `tests/conftest.py`). The only test that imports `_build_markov_stats` is a focused unit test with a synthetic regime sequence. The bug was missed before because nothing exercised the *relationship* between `forecast.1_step` and the transition matrix; the new contract test fixes that.

- **Reuse the existing ticker fixtures.** `tests/conftest.py` already exposes `btc_csv` and `sample_ohlcv`. The new module adds `spy_csv`, `nvda_csv`, `ko_csv`, `xle_csv` fixtures of the same shape (`scope="session"`, skip on missing file).

- **Reuse the existing `run_regime()` helper** from `tests/conftest.py`. Do not spawn a new subprocess pattern.

- **Parametrize over (ticker, engine).** Use `pytest.mark.parametrize` to generate the 25-cell grid. The slow mark on the class keeps the default `pytest` invocation under 30s; `pytest -m statistical` (or `pytest --runslow`) exercises the full panel.

- **Prior art**: `tests/integration/test_regime_contract.py` for the wire-format pattern, `tests/integration/test_eval_mode.py` for the eval-mode pattern, `tests/integration/test_real_data_suite.py` for multi-ticker parameterization, and `tests/unit/test_build_markov_stats.py` for the unit-level seam.

- **What makes a good test**: asserts on the algebraic / statistical invariant documented in the SKILL.md / ADR-0025, not on the precise Sharpe number. The Sharpe grid is *reported* (artifact), not *asserted* (test). A regression that drops fshmm Sharpe from −0.5 to −5.0 would still pass the wire-format test today — the new panel catches it via the median-per-engine buy-and-hold delta assertion.

## Out of Scope

- **Not building a backtester-improvement feature** (e.g. signal-scaled sizing — that's already #85, open, and `ready-for-human`).
- **Not addressing the κ < 0 finding by merging engines.** The cross-engine disagreement is documented as an observation; this PRD only adds the harness to *measure* it. A future PRD could propose an ensemble layer.
- **Not changing the threshold engine's lower persistence.** The off-by-one is a bug; the lower persistence is the design intent. The fix is to compute the forecast correctly, not to change the engine.
- **Not addressing the robust_hmm mid-stream collapse to 2 states.** That is the documented behavior of ADR-0018 / ADR-0096. The new harness will report it as part of the artifact; it is not a regression to fix here.
- **Not addressing the fshmm single-trade statistical-meaningfulness problem.** That requires either a shorter window or longer horizons and is a separate research question.
- **Not re-running on the eval-results/ JSON snapshots.** Those are historical artifacts; the new harness runs against the *current* pipeline output, not the 2026-06-02 snapshot.

## Further Notes

- The 5-ticker panel was chosen interactively to span: low-vol mega-cap (SPY), defensive (KO), cyclical (XLE), high-β single-name (NVDA), and crypto (BTC). The harness makes the panel explicit and reproducible; future contributors can change the panel by editing the test module's `TICKERS` constant, not by re-running an ad-hoc script.
- The off-by-one fix is one line in `_build_markov_stats`, but the test surface is large (5 engines × 5 tickers × 3 forecast horizons = 75 assertions) because the bug is masked by persistence. The new tests are the only way to keep it masked *correctly*.
- The current value of κ = −0.027 is the *floor* to assert on. If a future engine change moves κ above 0.4 (the "moderate" threshold on the Fleiss scale), that's a real signal worth a follow-up PRD; for now we just don't let it get worse.
- This PRD pairs naturally with the existing PRD #99 ("Real-Data Integration Test Suite Across test_data/ Tickers") which is closed — the new test module is the *quantitative* companion to that work.
- The artifact file `runs/statistical_report.txt` is committed once (with the first run) and overwritten on subsequent runs; diffing it against the committed version is a useful pre-merge sanity check.
