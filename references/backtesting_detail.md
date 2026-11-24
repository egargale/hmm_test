# Walk-Forward Backtesting Detail

## Methodology

The walk-forward backtest implemented in
`regime.walk_forward.walk_forward_backtest()` replays regime labels that the
engine has already produced in a no-lookahead walk-forward classification. It
does **not** re-classify or rebuild a transition matrix per bar inside the
backtest, and it does **not** use future information at any decision point.

The pipeline flow is:

```
engine.run_classify()          # expanding-window walk-forward classification
  -> regimes[t], posteriors[t] # one label/posterior row per bar
  -> _walk_forward_positions() # replay + dwell/hysteresis filters + sizing
  -> positions[t]              # binary int or signal-sized float
  -> pnl[1:] = positions[:-1] * returns[1:]
  -> Sharpe / drawdown / trade stats
```

## Algorithm

For each bar `t` from `min_train` to `len(returns) - 1`:

1. **Replay the pre-computed label** `regimes[t]` from the engine's
   walk-forward classification.
2. **Apply the filters** (`--dwell-bars`, `--hysteresis`) to decide whether
   the held regime may switch. Both filters must agree (AND logic).
3. **Determine the allowed direction** from the filtered regime:
   bear `-1`, sideways `0`, bull `+1`.
4. **Write the position**:
   - **Binary mode (default):** `positions[t] = allowed`.
   - **Signal mode (opt-in, `--signal-sizing`):** project the clipped
     per-bar signal onto the allowed direction (see below).
5. **P&L:** the existing one-bar lag is unchanged:
   `pnl[1:] = positions[:-1] * returns[1:]`. A position formed at bar `t`
   earns the return observed at `t+1`.

Bars before `min_train` remain flat and are excluded from the metrics.

## Position Sizing

### Binary (default)

| Filtered regime | Position | Interpretation |
|-----------------|----------|----------------|
| Bear (0) | -1 | Full short |
| Sideways (1) | 0 | Flat |
| Bull (2) | +1 | Full long |

### Signal-sized (`--signal-sizing`)

For each bar `t`, with `posteriors[t] = [P(bear), P(sideways), P(bull)]`:

```
signal[t]   = clip(P(bull)[t] - P(bear)[t], -1.0, 1.0)
allowed[t]  = {bear: -1, sideways: 0, bull: +1}   # after filters
position[t] = allowed[t] * max(0.0, allowed[t] * signal[t])
```

| Filtered regime | Signal | Position | Interpretation |
|-----------------|-------:|---------:|----------------|
| Bull | +0.9 | +0.9 | Strong bull, partial-to-full long |
| Bull | +0.2 | +0.2 | Weak bull, small long |
| Bear | -0.9 | -0.9 | Strong bear, partial-to-full short |
| Bull | -0.4 | 0.0 | Bull label, bearish signal -> flat |
| Bear | +0.4 | 0.0 | Bear label, bullish signal -> flat |
| Sideways | any | 0.0 | Sideways-held -> always flat |
| Bull | +5.0 | +1.0 | Over-confident synthetic input clipped |
| Bear | -5.0 | -1.0 | Over-confident synthetic input clipped |

The filters still gate **direction**; the signal only scales magnitude within
the allowed direction. A filter-blocked switch keeps the held direction, and
the current bar's signal sets the size within it. A non-finite posterior
becomes flat (`0.0`) and never reaches P&L.

`--signal-floor F` flattens positions with `abs(position) < F` (strict `<`;
a position exactly at `F` is kept). `F` must be in `[0, 1]`; the default
`0.0` disables the floor. `--signal-floor` requires `--signal-sizing`.

Signal sizing needs per-bar posteriors, so it works only with the HMM-family
engines (`hmm`, `messina`, `robust_hmm`, `fshmm`). The `threshold` engine has
`posteriors=None` and raises a clear `ValueError` in signal mode; there is no
silent fallback to binary.

The output always records `walk_forward.sizing_mode`, either `"binary"` or
`"signal"`, including short-data results with null metrics.

## Equity Curve

```
equity[t] = equity[t-1] * (1 + pnl[t])
```

Starting capital is 1.0 (unit-initialised). The first `min_train` bars are set
to NaN and excluded from the equity-series metrics because no trading occurs
during the warm-up period.

## Performance Metrics

### Sharpe Ratio

Calculated via `backtesting.performance_metrics.calculate_sharpe_ratio()`:

```
Sharpe = sqrt(252) * mean(excess_returns) / std(returns)
```

- **Annualisation**: `sqrt(252)` assumes daily data. For intraday data, this
  overstates the Sharpe ratio. The `infer_trading_frequency()` function
  attempts auto-detection but defaults to daily. For non-daily data, interpret
  cautiously.
- **Risk-free rate**: Default 2% annual. Subtracted proportionally from daily
  returns.
- **NaN handling**: If insufficient valid data (<2 bars in the equity curve),
  returns `NaN` -> `null` in JSON output.

### Max Drawdown

```
drawdown[t] = (equity[t] - peak_equity[t]) / peak_equity[t]
max_drawdown = min(drawdown)
```

Peak equity is the running maximum. Drawdown is expressed as a negative
fraction (e.g. `-0.15` = 15% peak-to-trough decline).

### Trade Statistics

A trade is a **flat-to-flat position cycle**, not every position change:

- **Entry:** a position becomes non-zero from zero.
- **Exit:** the position returns to zero.
- A trade still open at the end of the series is counted with the final equity.
- Trade return = `equity[exit] / equity[entry] - 1`.

This means scaling `0.9 -> 0.2 -> 0.9` inside one continuous position is still
one trade. Because fractional positions can stay non-zero for long stretches,
`n_trades`, win rate, and profit factor are **not directly comparable** between
binary and signal-sizing modes, and a sizing run may report very few trades.
The metrics are not redefined for signal mode.

## No-Lookahead Guarantees

| Potential bias | Prevention |
|----------------|------------|
| Using future returns for classification | The engine classifies bar `t` from data up to `t` (`returns[0:t)`); the backtest only replays those labels. |
| Knowing the regime sequence in advance | Labels are produced by the engine's expanding-window walk-forward classify, not a full-sample fit. |
| Filter peeking at the future | `_apply_filters()` sees only the current bar's label/posterior and the held filter state. |
| Signal leaking future information | `posteriors[t]` is the per-bar row produced at bar `t`; signal sizing never reads rows after `t`. |
| Same-bar execution | P&L uses the one-bar lag: the position formed at `t` earns `returns[t+1]`. |

## Transaction Costs

**Not modeled in the walk-forward backtest.** The equity curve assumes
zero-cost execution. For realistic backtesting:

- Use the `backtesting.strategy_engine.backtest_strategy()` function directly,
  which models commission and slippage via `BacktestConfig`.
- Or adjust the Sharpe ratio downward to account for costs (typical: subtract
  0.1-0.3 from Sharpe for retail trading costs).

## Limitations

1. **No transaction costs**: Returns are gross of costs. Real performance will
   be lower, and signal-sized rebalancing may trade more often than the binary
   model in practice.
2. **No position limits**: Fractional positions are assumed always executable.
   Real markets have lot sizes and liquidity constraints.
3. **No market impact**: Trade size does not affect price.
4. **Single-asset**: Cross-asset diversification not modeled.
5. **Daily rebalancing**: Positions are re-evaluated every bar.
6. **Fixed parameters**: Window, threshold, and min_train are constant
   throughout the backtest. Adaptive parameter schemes may improve robustness.
7. **Signal sizing is an exposure model, not a return claim**: The saved
   evaluation data contains no per-bar sizing run, so any performance benefit
   remains a hypothesis to measure.

## Interpreting Results

- **Sharpe > 1.0**: Good risk-adjusted returns.
- **Sharpe 0.5-1.0**: Moderate. May still be useful as a regime filter for
  another strategy.
- **Sharpe < 0.5**: Weak standalone. Use only for regime context, not direct
  trading.
- **Max drawdown > -20%**: Acceptable for most strategies.
- **Max drawdown < -40%**: High risk. Consider position size limits or
  stop-loss overlays.

Remember that walk-forward results are **in-sample for the parameter search
space** (window, threshold). Out-of-sample performance may differ.
Cross-validation by time period is recommended for parameter selection.
