> generated_by: nexus-mapper v2
> verified_at: 2026-06-06
> provenance: Derived from CONTEXT.md and AST analysis

# Core Domain Concepts

## Engine
A self-contained regime classifier implementing the `RegimeEngine` Protocol (`precompute` → `classify` → `run_classify`). Five exist: threshold (close-only, vectorized), hmm (HMM + ~50 generic features), messina (HMM + 19 Wilder's features), robust_hmm (HMM + Huber/MCD outlier resistance), fshmm (HMM + feature saliency learning). Each engine produces `ClassifyOutput` consumed by the pipeline.

## Regime (Market Regime)
The market condition at a point in time: **Bear (0)** = declining, **Sideways (1)** = range-bound, **Bull (2)** = rising. States map to trading positions: 0 → Short (-1), 1 → Flat (0), 2 → Long (+1).

## HMM Latent State
Raw state index (0, 1, ..., n) output by `GaussianHMM.predict()`. These indices are arbitrary and must be mapped to regimes by sorting ascending mean return.

## Walk-forward Backtest
Bias-free backtest where at each bar t, only data up to t-1 is used for classification. Produces discrete trades with Sharpe, max drawdown, win rate, profit factor.

## Signal
`P(next_regime = Bull) - P(next_regime = Bear)` in range [-1, 1].

## Transition Matrix
A 3×3 row-normalized matrix where T[i][j] = probability of i→j transition.

## Degenerate Auto-recovery
When an HMM engine detects a 3-state fit collapsing (one state < 5% of bars), auto-downgrades to n_states=2. Mid-stream degeneration (detected during walk-forward classify) also triggers downgrade.

## PCA Whitening
Optional dimensionality reduction applied after z-score normalization and before `model.fit()`.

## Duration Forecasting
Survival analysis on regime spell lengths using Weibull or Cox models.

## Ticker Disk Cache
On-disk yfinance caching under `~/.cache/hmm-regime/tickers/`. Provides `get_ticker_data()` with refresh/no-cache modes. Sanitizes ticker names for filesystem safety. Recovers from corrupted cache files.
