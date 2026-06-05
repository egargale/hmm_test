# Adaptive Threshold for the Threshold Engine

The threshold engine will not gain volatility-adaptive thresholds. Its fixed ±5% boundaries are a deliberate design choice for its target envelope (daily std > 2%).

## Why this is out of scope

The root cause analysis on HDB (daily std 1.69%) showed that ±5% = ±0.66σ, causing ~51% of bars to land in Sideways. The proposed fix — `threshold = k × daily_std × √window` — would auto-scale boundaries to realised volatility.

This would change the engine's contract from "simple, deterministic, close-only" to "parameterised by volatility model", adding:

- A volatility estimator (rolling std, EWMA, or similar) with its own tuning surface
- State-dependent behaviour that breaks the engine's reproducibility guarantee
- A new tuning constant `k` with no principled default across asset classes

The threshold engine exists as a **fast baseline** for high-vol assets where fixed thresholds are empirically adequate. For low-vol equities, CONTEXT.md recommends `messina` or `fshmm` — both correctly detect bear on HDB without threshold changes.

Parameter tuning (`--threshold 0.04`, `--window 40`) works as a quick fix for individual tickers but no single (window, threshold) pair generalises across volatility regimes. That is a fundamental limitation of fixed-param engines, not a fixable bug.

## Prior requests

- #93 — "Threshold engine fails to detect sustained bear market on HDB"
