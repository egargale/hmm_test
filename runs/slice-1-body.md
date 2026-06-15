## Parent

#111

## What to build

Fix the forecast off-by-one defect in the Markov-stats helper. The function that builds the `MarkovStats` dataclass currently sets `current_probs` to a row of the transition matrix, but downstream forecast code interprets it as the current-regime probability vector, so `forecast.1_step` ends up being a 2-step distribution. The fix replaces that assignment with a one-hot for the current regime, robust to both the default 3-state and the 2-state post-degenerate-recovery shapes. The slice also updates the existing unit test that codified the old (buggy) invariant and adds a new synthetic-sequence test that pins the new invariant at the function level.

End-to-end behaviour after this slice: the `MarkovStats.current_probs` field is a one-hot unit vector for the current regime; `next_state_probabilities` and the JSON `forecast.1_step` field are equal to the row of the transition matrix for the current regime (to floating-point tolerance); `forecast.5_step` is `T^5 @ e_current`; `forecast.20_step` is `T^20 @ e_current`. The wire-format `signal`, `next_state_probabilities`, `verdict`, and `walk_forward` outputs are unchanged.

## Acceptance criteria

- [ ] `MarkovStats.current_probs` is a one-hot unit vector for the current regime (`e_current`), not a row of the transition matrix. The construction handles both 3×3 and 2×2 transmat shapes.
- [ ] For the threshold engine on every test-data ticker, the JSON `forecast.1_step` field equals the `T[current_regime_index, :]` row elementwise to `abs(Δ) < 1e-9`. The HMM engines' `forecast.1_step` already matches by construction; the test must also confirm that (regression guard for the high-persistence case).
- [ ] The existing unit test in `tests/unit/test_build_markov_stats.py` that asserts `current_probs == T[current_regime]` is renamed/updated to assert `current_probs == e_current`. The test name reflects the new invariant.
- [ ] A new synthetic unit test (`test_current_probs_is_one_hot` or similarly named) calls the helper on a known regime sequence such as `[0,0,0,1,1,2,2,2,2,0]` and asserts the one-hot property without HMM fitting.
- [ ] All ten existing tests in `tests/unit/test_build_markov_stats.py` continue to pass; the helper's existing public surface (`MarkovStats` field names, `transmat`, `stationary`, `persistence`, `signal`, `regime_counts`, `dates`) is unchanged.
- [ ] The 25 JSON outputs already committed under `runs/json/` regenerate identically except for the three `forecast.{1,5,20}_step` fields and downstream `verdict`/`duration_forecast` values that consume them; the `current_regime`, `next_state_probabilities`, `transition_matrix`, `stationary_distribution`, `persistence_diagonal`, `regime_counts`, `walk_forward`, and `engine_info` fields are byte-identical before and after the fix. (This is a manual verification step — re-run the harness from Slice 2 and diff the report.)

## Blocked by

None - can start immediately.
