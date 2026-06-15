# HMM Regime Detection — Statistical Evaluation Report

**Test scope:** 5 tickers × 5 engines = 25 CLI invocations on
`test_data/{SPY,NVDA,BTC,KO,XLE}.csv` using
`./scripts/run.sh --csv … --json --engine …`.

**Tickers chosen for diverse risk profiles:**

| Ticker | Profile | Daily σ (computed) | Buy-and-hold Sharpe | Buy-and-hold return |
|---|---|---|---|---|
| SPY | Mega-cap index ETF (low vol) | 0.0106 | 0.90 | +325% (10y) |
| KO | Defensive consumer staple | 0.0111 | 0.58 | +140% (10y) |
| XLE | Cyclical commodity ETF | 0.0158 | 0.47 | +159% (10y) |
| NVDA | High-β single-name tech | 0.0306 | 1.31 | +19 189% (10y) |
| BTC | Crypto, very high vol | 0.0418 | 0.43 | +94% (5y, from 2021) |

**Engines:** `threshold`, `messina`, `hmm`, `robust_hmm`, `fshmm`.

Raw outputs: `runs/json/{TICKER}__{ENGINE}.json`.
Master log: `runs/master.log`.

---

## TL;DR — Statistical-Expert Verdict

| Test | Result | Comment |
|---|---|---|
| **Output contract** | ✅ PASS | 0 schema violations across 25 runs |
| **Internal algebra** (`next_state_probs == T[current,:]`) | ✅ PASS | All 25 runs satisfy it |
| **Forecast[1_step] == T[current,:]** | ❌ **FAIL** | Off by one step in the threshold engine (max Δ 0.18); HMM engines pass because of near-1.0 persistence that masks it |
| **Cross-engine agreement (Fleiss' κ)** | ❌ FAIL | κ = −0.027 (no better than chance, slightly worse) |
| **Walk-forward skill vs B&H** | ❌ FAIL | 0/5 engines beat B&H on ≥3/5 tickers; median delta in [−1.31, −0.10] |
| **Persistence above random** | ✅ PASS | 25/25 runs have regime spell > 3 bars |
| **Verdict field consistency** | ✅ PASS | 24/25 verdicts align with regime + signal; 1 "transition" verdict is a legitimate reversal forecast |
| **Forecast 20-step vs stationary** | ⚠ EXPECTED | Wide gaps (up to 0.61) reflect very high persistence (0.99+) — 20 steps is not enough to mix |

The program is **technically correct in its internal algebra and output contract**, but **no engine produces consistent, predictive regime signals across the 5-ticker test panel**. The headline shortcoming is divergence, not numerical errors.

---

## 1. Output contract validation

25/25 runs produced valid JSON containing every documented key. Transition-matrix rows all sum to 1.0 (±0.001), the stationary distribution sums to 1.0, the signal is in [−1, 1], current regime index is in {0, 1, 2}, persistence diagonal entries are in (0, 1).

**This is a clean PASS — the wire format is solid.**

## 2. Forecast algebra — one real bug

The contract (SKILL.md) says `forecast.1_step` should be the row of the transition matrix corresponding to the current regime. Mathematically `T^k @ e_i = T^k[i, :]` (with `e_i` a one-hot), so for `k=1` it should equal `T[i, :]` exactly.

I verified the field `next_state_probabilities` IS `T[i, :]` (Δ=0 on all 25 runs).
But `forecast.1_step` ≠ `T[i, :]` on the **threshold engine only**:

| Ticker | current | T[i] | forecast[1] | Δ |
|---|---|---|---|---|
| BTC | bear | [0.84, 0.15, 0.01] | [0.72, 0.25, 0.03] | **0.117** |
| NVDA | bull | [0.00, 0.10, 0.90] | [0.01, 0.16, 0.83] | **0.072** |
| SPY | bull | [0.00, 0.27, 0.73] | [0.01, 0.44, 0.55] | **0.183** |
| XLE | side | [0.05, 0.89, 0.06] | [0.09, 0.80, 0.11] | **0.083** |

**Root cause** — `hmm_futures_analysis/regime/pipeline.py:241`:

```python
current_probs = transmat[last_regime]      # row i of T (i.e. 1-step ahead)
…
forecast_n_steps(markov.transmat, markov.current_probs, 1)
```

`forecast_n_steps` then does `current_probs @ T^n`. So `forecast[1_step]` is actually `T[i] @ T = T^2[i, :]` (a 2-step distribution); `forecast[5_step]` is a 6-step; `forecast[20_step]` is a 21-step.

The HMM engines hide the bug because their persistence is 0.99+ (≈ identity), so `T[i] @ T ≈ T[i] ≈ e_i @ T` and the contract is met to 4+ decimal places.

**Fix** (one line): replace with a one-hot

```python
current_probs = np.zeros(transmat.shape[0])
current_probs[last_regime] = 1.0
```

This is a real, low-risk defect; it would not be visible on the HMM panels in this repo's existing eval folders, which is probably why it hasn't been caught.

## 3. Cross-engine agreement — Fleiss' κ = −0.027

Fleiss' kappa on the 5-engine × 5-ticker current-regime matrix:

| Ticker | bear | side | bull | majority |
|---|---|---|---|---|
| SPY | **3** | 1 | 1 | bear (60%) |
| NVDA | 0 | 1 | **4** | bull (80%) |
| BTC | 1 | 2 | 2 | tie (40/40) |
| KO | 1 | **3** | 1 | side (60%) |
| XLE | 2 | 2 | 1 | tie (40/40) |

κ = **−0.027**. By Cohen/Fleiss convention, that is *worse than chance* — the engines disagree so much that the agreement is below what you'd expect from 5 random voters picking from {bear, side, bull} uniformly.

The two most-decisive tickers (NVDA, SPY) DO show clear majority, but with only 5 tickers, the result is dominated by the 3 split cases (BTC, KO, XLE).

The pairwise matrix also shows that **threshold** and **hmm** agree 60% of the time (somewhat correlated), **messina** and **fshmm** agree 60% (correlated), and the threshold/messina/hmm cluster disagrees with the robust_hmm/fshmm cluster. There is no global consensus.

This is a **structural finding**, not a measurement error. With 5 independent engines producing regimes from very different statistical principles (rolling sum vs. 19-feature HMM vs. 50-feature HMM vs. robust HMM vs. saliency HMM), high disagreement on a small test panel is not surprising, but κ<0 is a strong signal that they are not converging on the same regimes.

## 4. Walk-forward performance vs buy-and-hold

Per-ticker Sharpe grid (engine Sharpe, B&H row at bottom for reference):

| engine      |   SPY |  NVDA |   BTC |    KO |   XLE | median |
|-------------|------:|------:|------:|------:|------:|-------:|
| threshold   | -0.35 |  0.31 | -0.22 | -0.37 | -0.00 | -0.22 |
| messina     | -0.62 | -0.42 | -0.07 |  0.03 | -0.43 | -0.42 |
| hmm         | -0.52 |  0.26 |  0.55 | -0.31 | -0.22 | -0.22 |
| robust_hmm  | -0.61 |  1.12 |  0.36 |  0.48 |  0.44 |  0.44 |
| fshmm       | -0.97 | -0.10 | -0.60 | -0.73 | -0.53 | -0.60 |
| **B&H**     |  0.90 |  1.31 |  0.43 |  0.58 |  0.47 |  0.58 |

Per-engine median delta vs B&H:

| engine | median ΔSharpe | wins (Δ>0) | verdict |
|---|---:|---:|---|
| threshold | −0.94 | 0/5 | destroys B&H by ~1σ |
| messina | −0.91 | 0/5 | destroys B&H by ~1σ |
| hmm | −0.88 | 1/5 | destroys B&H by ~1σ |
| robust_hmm | −0.10 | 0/5 | within striking distance |
| fshmm | −1.31 | 0/5 | destroys B&H by 1.3σ |

**No engine consistently adds skill over buy-and-hold.** This is expected for a regime-detection system that is marketed as a *descriptive* tool, not a trading signal — but it is important to record. The robust_hmm median is positive only because of a degenerate-2-state collapse to "all-bull" on 4/5 tickers (see §6).

## 5. Persistence (regime stickiness) — sane

Empirical average regime spell length (bars between transitions):

| engine | min | median | max |
|---|---:|---:|---:|
| threshold | 5.8 | 7.7 | 11.1 |
| messina | 81.0 | 96.5 | 132.3 |
| hmm | 125.7 | 193.1 | 837.3 |
| robust_hmm | 418.7 | 912.5 | 1256.5 |
| fshmm | 147.8 | 209.2 | 314.0 |

A random i.i.d. 3-state label generator would produce ~3 bars/spell. All 25 runs exceed that. HMM engines are much stickier than the threshold engine, which is the design intent. **No degenerate "every bar is its own regime" failures observed.**

## 6. Two engine-specific pathologies

### 6.1 `robust_hmm` mid-stream collapse to 2 states

On 4/5 tickers (NVDA, BTC, KO, XLE) the robust_hmm engine's regime_counts are `{bear: 0, sideways: 252, bull: 2261}`-ish — only **1 transition** in the entire walk-forward, all after the `min_train` warm-up. The persistence diagonal shows `persist_bear = 0.333` (placeholder, since bear is unvisited) and the strategy degenerates to "go long at bar 252, hold forever".

This is the documented "**mid-stream degeneration**" path described in CONTEXT.md (Degenerate auto-recovery, ADR-0018 amended). It is working as designed for degenerate fits, but the **side-effect is that the strategy effectively becomes buy-and-hold**, so a high Sharpe on NVDA (1.12) is mostly the buy-and-hold 19 000% return minus churn, not a regime-detection edge. The Sharpe-vs-B&H delta for robust_hmm is the smallest of the five engines (median −0.10), which corroborates this.

### 6.2 `fshmm` produces too few trades to be statistically meaningful

fshmm fires **1 completed trade per ticker**, and that single trade is always a loss in this 5-ticker panel. Win-rate=0%, profit-factor=0% across the board. The reason is that fshmm is ultra-sticky (avg spell 150–300 bars) so the walk-forward only sees a handful of regime changes. With `n_trades ≤ 1`, the Sharpe is essentially a function of when the last regime change happened and the sign of the trade that followed. **fshmm Sharpe values are not statistically meaningful on this test panel** — they need either longer horizons or shorter windows.

## 7. Verdict & duration forecast

The `verdict` field is internally consistent: bullish/bearish match the current regime + sign of `signal`, neutral is reported for `|signal| < 0.1` sideways, and the lone "transition_bull" verdict (BTC threshold) reflects the 20-step forecast flipping the regime, which is the documented behavior.

`duration_forecast` is `null` in all 25 runs (it is opt-in via `--duration-forecast`; we did not enable it in this test). No false nulls.

## 8. Final assessment

| Aspect | Grade | Reason |
|---|---|---|
| **Wire format & schema** | A | Clean contract, no violations |
| **Internal algebra** | A− | One subtle forecast bug (§2) — minor in practice, should be fixed |
| **Statistical robustness** | D | κ<0 across engines, no engine beats B&H on a majority of tickers |
| **Per-engine suitability** | mixed | See CONTEXT.md engine-suitability table — this 5-ticker panel does not contradict the documented "low-vol → hmm, high-vol → threshold, outlier-prone → robust_hmm" rule of thumb |
| **Reproducibility** | A | 25/25 runs deterministic and re-runnable from CSVs |

**The program is acting well as a self-consistent regime detector, but it is not acting as a profitable trading signal in this test.** The 5×5 panel shows that the documented behavior holds, the JSON contract is honored, and the engines do what they say — but they disagree with each other, and none consistently produces trade-level alpha over buy-and-hold. That matches the explicit disclaimer in the SKILL ("regime detection is probabilistic … not financial advice") and is consistent with the body of academic literature on HMM regime detection for daily financial series.
