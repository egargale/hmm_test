# Hysteresis and dwell-time whipsaw filters in walk-forward layer

HMM regime classification is noisy — consecutive bars can oscillate between
states, generating excessive trades with low win rates. This is a well-known
HMM weakness in financial time series. Two complementary filters reduce
whipsaw: minimum dwell-time (consecutive bars) and hysteresis (posterior
probability margin).

## Considered options

### A) Filters inside engines

Add dwell/hysteresis logic to each engine's `classify_regimes()` method.

**Rejected because**: violates ADR-0003 engine self-containment. Engines
classify regimes; the walk-forward layer decides whether to act on the
classification. Mixing filtering into engines conflates two responsibilities
and would require changes to all three engines (threshold has no posteriors).

### B) Filters in walk-forward layer (chosen)

Add `_apply_filters()` in `walk_forward.py` that gates position changes
behind both dwell-time and hysteresis checks. Both default to disabled for
backward compatibility. Requires extending `ClassifyResult` with a
`posteriors` field.

**Chosen because**: (1) respects ADR-0003 — engines classify, walk-forward
decides; (2) threshold engine can't provide posteriors — the filter degrades
gracefully (hysteresis is a no-op when `posteriors is None`); (3) no
lookahead — filters use only data available at bar `t`; (4) AND logic
(both filters must agree) is the simplest and most conservative choice.

## Consequences

- **New CLI flags**: `--dwell-bars N` (default 0 = disabled) and
  `--hysteresis D` (default 0.0 = disabled). Thread through `pipeline.run()`
  and `walk_forward_backtest()`.

- **`ClassifyResult` extended**: New `posteriors: np.ndarray | None` field.
  Both HMM engines populate it via `model.predict_proba()`. Threshold engine
  returns `None` (no probabilistic model).

- **`_apply_filters()` logic**: Called at each bar with the candidate new
  regime, current regime, posteriors, and consecutive count. Returns
  `(should_switch, updated_consecutive_count)`. Uses AND logic — both
  dwell-time and hysteresis must agree to switch.

- **Dwell-time filter**: Tracks consecutive bars with the same candidate
  regime. Position only changes after `N` consecutive same-regime
  classifications. Counter resets when the candidate regime changes.

- **Hysteresis filter**: Only switches when `posteriors[new_regime] -
  posteriors[current_regime] > hysteresis_delta`. Requires posterior
  probabilities from `ClassifyResult`. No-op when `posteriors is None`
  (threshold engine).

- **Graceful degradation for threshold engine**: The threshold engine
  returns `posteriors=None`, so hysteresis is automatically a no-op.
  Dwell-time still works (it only needs regime labels, not probabilities).

- **Posterior aggregation**: When `n_states > 3`, HMM engines aggregate
  posteriors by regime bucket (Bear/Sideways/Bull) before returning.
  This keeps the hysteresis comparison meaningful regardless of state count.

- **Both filters in both walk-forward paths**: `_walk_forward_precomputed()`
  and `_walk_forward_raw()` both call `_apply_filters()` at each bar.
