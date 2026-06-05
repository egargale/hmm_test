# HMM Regime Detection

Detect market regimes (Bull/Bear/Sideways) using threshold-based classification or Hidden Markov Models, with bias-free walk-forward backtesting.

## Language

**Engine**:
A self-contained regime classifier that produces regime labels and posterior probabilities via `run_classify()`. Each engine owns its execution model — one-shot vectorized (threshold) or walk-forward HMM refitting (messina, hmm, robust_hmm, fshmm). The pipeline consumes the engine’s `ClassifyOutput` and assembles the full output block (transition matrix, forecasts, walk-forward backtest) downstream. Five engines exist: `threshold` (fast, close-only), `messina` (HMM with 19 Messina features), `hmm` (HMM with ~50 generic features), `robust_hmm` (HMM with outlier-resistant emissions via Huber IRLS or MinCovDet), and `fshmm` (HMM with per-feature saliency weights learned during EM). The user selects one per invocation.
_Avoid_: model, mode, method, strategy

**Walk-forward backtest**:
A bias-free backtest where at each bar `t`, only data up to `t-1` is used for regime classification. The position at bar `t` is determined by the regime at `t-1`. Produces discrete trade-level results: Sharpe, max drawdown, trade count, win rate, profit factor, total return.
_Avoid_: rolling backtest, historical simulation

**Regime**:
The market condition at a point in time, one of {Bear (0), Sideways (1), Bull (2)}. Bears are declining markets, Bullls are rising markets, Sideways are range-bound. State 0 always maps to Short, 1 to Flat, 2 to Long in the trading model.
_Avoid_: state (ambiguous — could mean HMM latent state or mapped regime)

**HMM latent state**:
The raw state index (0, 1, 2) output by a GaussianHMM `predict()` call. These indices are arbitrary and must be mapped to regimes by sorting ascending mean return: lowest → Bear, middle → Sideways, highest → Bull.
_Avoid_: state, cluster

**Signal**:
`P(next_regime = Bull) - P(next_regime = Bear)`, range [-1, 1]. Positive = bullish, negative = bearish. Derived from the transition matrix row for the current regime.
_Avoid_: score, conviction

**Transition matrix**:
A 3×3 row-normalized matrix where `T[i][j]` = probability of transitioning from regime `i` to regime `j`. Row sums to 1.0. Persistence diagonal entries (i → i) measure regime stickiness.
_Avoid_: confusion matrix, Markov chain

**OHLCV**:
Open, High, Low, Close, Volume — the five price columns required for feature engineering. Messina and HMM engines require OHLCV. The Messina and HMM engines require OHLCV and raise ``ValueError`` when missing.
_Avoid_: price data, market data

**Feature engineering**:
The process of computing technical indicators (log returns, SMAs, ATR, RSI, MACD, Bollinger, VSTOP, etc.) from OHLCV data. Two modes exist: `generic` (~50 indicators, SMA-based) and `messina` (19 indicators, Wilder's smoothing, VSTOP, ADX/DI, interaction terms, level gaps, KDJ oscillator). Features are precomputed once on the full dataset and sliced per bar in the walk-forward loop — all indicators are backward-looking (no lookahead bias).
_Avoid_: indicator calculation, TA computation

**Discrete trade**:
A trade with an entry time, entry price, exit time, exit price, and P&L. Positions are {-1, 0, 1}. A trade is opened when the regime changes and closed when it changes again. Transaction costs (commission, slippage) are modeled but default to zero for engine comparability.
_Avoid_: continuous position, fractional trade, signal-weighted trade

**Degenerate auto-recovery**:
When an HMM engine's pre-check detects that a 3-state fit will collapse (one state receiving < 5% of bars), the engine automatically downgrades to `n_states=2` before entering the walk-forward loop. The output uses the 2-state regime mapping (Bear/Bull, no Sideways). The original requested `n_states` and the auto-recovery event are recorded in `engine_info`. Per ADR-0018 (amended).
_Avoid_: auto-retry, fallback, degenerate retry

**Regime spooling**:
The threshold engine's method for mapping classified regimes to trading positions. At each bar, the regime (0/1/2) from `classify_regimes()` is mapped directly to a position via `{0: -1, 1: 0, 2: 1}`. No signal threshold or intermediate computation.
_Avoid_: signal-gating, conviction filtering

**BIC state selection** (`--n-states auto`):
Automatic selection of the number of HMM latent states via Bayesian Information Criterion. `select_n_states()` in `_hmm_engine.py` fits GaussianHMM for each candidate `k` in `[2, max_states]` with multiple restarts and returns the count with the lowest BIC. Default remains `--n-states 3`. The BIC penalty naturally guards against overfitting on short data windows.
_Avoid_: auto-tuning, state optimization

**PCA whitening**:
Optional dimensionality reduction applied after z-score normalization and before `model.fit()` inside `_fit_hmm_on_slice()`. Opt-in per engine via `pca_variance` parameter (e.g. 0.95 = keep 95% of variance). Reduces the generic engine's ~50 features to ~10-15 components. Component count is sticky — determined on first refit, reused on subsequent refits. Per ADR-0005.
_Avoid_: feature reduction, dimensionality reduction (use PCA whitening)

**Dwell-time filter** (`--dwell-bars N`):
Walk-forward filter that requires a regime to persist for N consecutive bars before the position changes. Lives in `walk_forward.py`, not inside engines (ADR-0003). Counter resets when the candidate regime changes. Default 0 = disabled.
_Avoid_: hold period, minimum hold

**Hysteresis filter** (`--hysteresis D`):
Walk-forward filter that only switches regimes when the posterior probability of the new regime exceeds the current regime's by a margin `> D`. Requires `posteriors` from `ClassifyResult` — no-op for the threshold engine (which returns `posteriors=None`). Lives in `walk_forward.py` (ADR-0003, ADR-0007). Default 0.0 = disabled.
_Avoid_: probability threshold, confidence filter

**Posteriors**:
Posterior probability array over regimes, returned by HMM engines via `ClassifyResult.posteriors`. Computed from `model.predict_proba()`. When `n_states > 3`, aggregated by regime bucket (Bear/Sideways/Bull). Used by the hysteresis filter. `None` for the threshold engine.
_Avoid_: probabilities, confidence scores

**Robust HMM engine** (`robust_hmm`):
An HMM engine that applies outlier-resistant emission estimation after standard GaussianHMM fitting. Two methods: `huber` (Huber IRLS correction of means and variances) and `mcd` (Minimum Covariance Determinant replacement of emission covariance). Opt-in via `--robust-method`. Uses the same ~50 generic features as the `hmm` engine.
_Avoid_: robust model, outlier engine

**FSHMM engine** (`fshmm`):
Feature Saliency HMM (Adams et al. 2016). Learns per-feature saliency weights ρ during EM, automatically identifying which of the ~50 generic features are most informative for regime detection. Features with ρ < `--saliency-threshold` (default 0.5) are masked as irrelevant. Outputs `feature_saliency` and `selected_features` in `engine_info`.
_Avoid_: saliency model, feature selection engine

**Engine config**:
A flat dataclass that encapsulates all constructor parameters for one engine. Each engine has its own config class (e.g. `ThresholdConfig`, `RobustHMMConfig`) with fields matching the engine's `__init__`. Configs also carry `name` (the registry key) and `features` (the feature-engineering mode label). The CLI constructs the right config from CLI args; pipeline and walk-forward never see engine-specific kwargs. Per ADR-0011.
_Avoid_: engine settings, engine params

**Regime transitions**:
Historical regime change events extracted from the classified regime sequence by `extract_transitions()` in `regime_transitions.py`. Each event is a `TransitionEvent` namedtuple with `date` (ISO), `from_regime`, `to_regime`, and `bar_index`. Walks adjacent regime pairs and emits one event per change. Always computed in `pipeline.run()` and included in `PipelineResult.regime_transitions`. In terminal output, displayed via `--transitions N` (N most recent, 0 = all); in JSON, always present.
_Avoid_: regime changes, regime history

**Duration forecast**:
Post-processing, now on by default, that estimates how long the current regime will persist. Two survival models: `weibull` (Weibull distribution fit to historical regime durations — default, no extra dependencies) and `cox` (Cox proportional hazards with realized-volatility and spell-return covariates — requires `lifelines`). Outputs expected remaining days, hazard rate, 50%-survival point, and Weibull shape/scale. The Cox model adds covariate-adjusted predictions (`cox_expected_remaining_days`, `cox_coefficients`, `concordance_index`). Opt-out via `duration_forecast=False` on `pipeline.run()`.
_Avoid_: regime length prediction, time-to-transition

**Verdict**:
A synthesized signal computed by `_compute_verdict()` in `pipeline.py`, always present in pipeline output. One of ``"bullish"``, ``"bearish"``, ``"neutral"``, ``"transition_bull"``, ``"transition_bear"``. For Bull/Bear regimes, compares the 20-step forecast probability to the current distribution — continuation if the forecast reinforces, transition if it reverses. For Sideways, uses ``|signal|`` against a dynamic threshold (see below). Includes a ``confidence`` field = ``abs(signal)``.
_Avoid_: prediction, outlook, recommendation

**Dynamic threshold**:
A regime-aging-adjusted threshold for the Sideways → transition verdict boundary. Computed by ``_compute_dynamic_threshold()`` from the duration forecast's Weibull fit. When ``days_in_regime > expected_total`` (Weibull unconditional mean), the threshold shrinks linearly from 1.0× at aging_ratio=1 down to 0.3× at aging_ratio=1.7. Makes the verdict more sensitive to transition signals as the regime ages past its historical norm. Falls back to 0.1 when duration data is unavailable.
_Avoid_: adaptive threshold, variable cutoff

## Engine suitability

| Engine | Best for | Weakness |
|---|---|---|
| `threshold` | Crypto, high-vol assets, close-only data | Whipsaw-heavy without dwell-bars; negative Sharpe on low-vol equities |
| `hmm` (generic) | Low-vol equities (e.g. KO, SPY), OHLCV-rich data | No universal PCA setting; per-ticker tuning needed; CRM unsalvageable across all configs |
| `messina` | Medium-vol assets, OHLCV-rich data | Requires PCA whitening (0.95) to overcome feature multicollinearity; without PCA the engine degenerates to 1-trade fits |
| `robust_hmm` | Data with outlier bars, broad indices (SPY) | PCA=0.90 required for positive Sharpe on most tickers; 0700_HK and CRM uniformly negative |
| `fshmm` | Feature selection diagnostics, individual equities | Saliency weights (ρ) identify dead features but don't fix high-vol degeneration; negative Sharpe on SPY, KO, BTC at defaults |

**Rule of thumb**: If daily std > 2%, prefer `threshold` with dwell-bars ≥ 5. If daily std < 1.5% and OHLCV is available, `hmm` with n_states=3 often outperforms threshold. `n_states=auto` (BIC) is worth trying — it recovered BTC from +0.31 to +0.78 Sharpe on the generic engine and from +0.31 to +0.74 on robust_hmm. The `n_states=2` edge case is now handled correctly (bear/bull with no sideways). Per ADR-0019.

- **"state"**: Use **regime** for the labeled market condition (Bear/Sideways/Bull) and **HMM latent state** for the raw model output index. Never use "state" alone.
- **"method"**: Use **engine** for the selected analysis pipeline. The old term "method" appears in the JSON `params.method` field; this is legacy from when threshold was the only engine and should migrate to `engine_info.method`.

## Example dialogue

> **Dev**: When the user runs `--engine hmm`, does the walk-forward backtest use the HMM's transition matrix or the threshold's?
>
> **Domain expert**: The HMM engine computes everything from the HMM. Each engine is self-contained. The HMM engine fits an HMM on the expanding feature window, predicts the regime at `t-1`, and maps that regime to a trade position. The transition matrix in the output comes from the HMM model parameters, not from threshold regime counts.
>
> **Dev**: And the signal field — is that still `P(bull) - P(bear)`?
>
> **Domain expert**: Yes, same formula, same range [-1, 1]. But the probabilities come from the HMM's transition matrix, not the threshold's. The formula is engine-agnostic; the source matrix differs.
>
> **Dev**: What happens if someone runs `--engine messina` on a CSV that only has close prices?
>
> **Domain expert**: It errors. Messina and HMM engines require OHLCV. The threshold engine is the only one that works with close-only data.
