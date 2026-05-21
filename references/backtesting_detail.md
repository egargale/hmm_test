# Walk-Forward Backtesting Detail

## Methodology

The walk-forward backtest implemented in `regime.walk_forward.walk_forward_backtest()` simulates real-time trading without lookahead bias. Crucially, **no future information is used at any decision point**.

## Algorithm

For each bar `t` from `min_train` to `N-1`:

1. **Historical window**: Extract returns `[0:t)` — only data strictly before time `t`.
2. **Regime classification**: Run `classify_regimes()` on the historical returns.
3. **Transition matrix**: Build from the historical regime sequence.
4. **Current regime**: Read the last regime from the historical sequence (`regimes[-1]`).
5. **Next-state probabilities**: Row `current_regime` from the transition matrix.
6. **Signal**: `P(bull) - P(bear)` from next-state probabilities.
7. **Position**: `clip(signal, -1.0, 1.0)` — long, short, or fractional.
8. **P&L at t**: `position × return[t]` — using the actual return observed at time `t`.

## Position Sizing

| Signal range | Position | Interpretation |
|-------------|----------|----------------|
| +0.8 → +1.0 | +1.0 | Full long |
| +0.2 → +0.8 | Fractional | Partial long |
| -0.2 → +0.2 | ~0.0 | Flat |
| -0.8 → -0.2 | Fractional | Partial short |
| -1.0 → -0.8 | -1.0 | Full short |

The signal is continuous, so positions smoothly transition as conviction changes. No discrete entry/exit logic is used — this avoids threshold-hunting artifacts.

## Equity Curve

```
equity[t] = equity[t-1] × (1 + position[t] × return[t])
```

Starting capital is 1.0 (unit-initialised). The resulting equity curve represents cumulative growth.

The first `min_train` bars are excluded from the equity curve (set to NaN) since no trading occurs during the warm-up period.

## Performance Metrics

### Sharpe Ratio

Calculated via `backtesting.performance_metrics.calculate_sharpe_ratio()`:

```
Sharpe = sqrt(252) × mean(excess_returns) / std(returns)
```

- **Annualisation**: `sqrt(252)` assumes daily data. For intraday data, this overstates the Sharpe ratio. The `infer_trading_frequency()` function attempts auto-detection but defaults to daily. For non-daily data, interpret cautiously.
- **Risk-free rate**: Default 2% annual. Subtracted proportionally from daily returns.
- **NaN handling**: If insufficient valid data (<2 bars in equity curve), returns `NaN` → `null` in JSON output.

### Max Drawdown

```
drawdown[t] = (equity[t] - peak_equity[t]) / peak_equity[t]
max_drawdown = min(drawdown)
```

Peak equity is the running maximum. Drawdown is expressed as a negative fraction (e.g. `-0.15` = 15% peak-to-trough decline).

### Number of Trades

Counted as the number of times the position changes:

```
n_trades = sum(diff(position) != 0)
```

This includes both entry and exit transitions. A position change from 0.5 to 0.6 counts as a trade.

## No-Lookahead Guarantees

| Potential bias | Prevention |
|---------------|------------|
| Using future returns for classification | Only `returns[0:t)` used for bar `t` |
| Knowing the regime sequence in advance | Regimes recomputed from scratch at each step |
| Using full-sample transition matrix | Transition matrix rebuilt from historical regimes only |
| Signal leaking future information | Signal computed from current-regime row of historical transition matrix |

## Transaction Costs

**Not modeled in the default walk-forward backtest.** The equity curve assumes zero-cost execution. For realistic backtesting:

- Use the `backtesting.strategy_engine.backtest_strategy()` function directly, which models commission and slippage via `BacktestConfig`.
- Or adjust the Sharpe ratio downward to account for costs (typical: subtract 0.1–0.3 from Sharpe for retail trading costs).

## Limitations

1. **No transaction costs**: Returns are gross of costs. Real performance will be lower.
2. **No position limits**: Fractional positions assumed always executable. Real markets have lot sizes and liquidity constraints.
3. **No market impact**: Trade size does not affect price.
4. **Single-asset**: Cross-asset diversification not modeled.
5. **Daily rebalancing**: Positions re-evaluated every bar. Higher-frequency strategies incur higher costs.
6. **Fixed parameters**: Window, threshold, and min_train are constant throughout the backtest. Adaptive parameter schemes may improve robustness.

## Interpreting Results

- **Sharpe > 1.0**: Good risk-adjusted returns.
- **Sharpe 0.5–1.0**: Moderate. May still be useful as a regime filter for another strategy.
- **Sharpe < 0.5**: Weak standalone. Use only for regime context, not direct trading.
- **Max drawdown > -20%**: Acceptable for most strategies.
- **Max drawdown < -40%**: High risk. Consider position size limits or stop-loss overlays.

Remember that walk-forward results are **in-sample for the parameter search space** (window, threshold). Out-of-sample performance may differ. Cross-validation by time period is recommended for parameter selection.
