# HMM Adapter Scout Report

## 1. What `run_hmm_regime()` Returns — Full Dict Structure

File: `scripts/regime/hmm_adapter.py`

**Success response:**
```python
{
    "available": True,
    "regimes": ["bear", "sideways", "bull", "bear", ...],       # list of str, one per row
    "transition_matrix": [[0.7, 0.2, 0.1], ...],                # 3×3 float, labeled order
    "stationary_distribution": {
        "bear": 0.25,
        "sideways": 0.45,
        "bull": 0.30,
    },
    "feature_mode": "messina"  # or "generic"
    "caveat": "HMM state labels are inferred from ascending mean return; labels may swap on re-fit",
}
```

**Failure response:**
```python
{"available": False, "reason": "<error message>"}
```

### Key internal details:
- State ordering is inferred by sorting states by their mean log-return: lowest → bear, middle → sideways, highest → bull (lines 72-73)
- `transition_matrix` is reordered via `np.ix_(order, order)` so it aligns with labeled bear/sideways/bull order (line 78)
- The `regimes` list length = `len(df_clean)`, which is typically **shorter** than the input `prices` after NaN drops from feature engineering (line 42: `df_clean = df[numeric_cols].dropna()`)

---

## 2. Does it produce regime labels for backtesting?

**Yes, but with alignment problems.**

`run_hmm_regime()` returns a flat list of string labels (`"bear"`/`"sideways"`/`"bull"`) that could theoretically be converted to integer positions and fed into a backtest. However:

| Issue | Detail |
|-------|--------|
| **Length mismatch** | Output length = `len(df_clean)` which is shorter than `len(prices)` |
| **No index alignment** | The returned `regimes` list has no index attached — it's a bare Python list of strings |
| **Label instability** | The explicit `caveat` warns labels may swap on re-fit |
| **Full-data fit** | HMM currently fits on all data at once — not walk-forward compatible |

---

## 3. Does it produce a signal that could replace `compute_signal()`?

**No.** `run_hmm_regime()` does **not** call `compute_signal()` or return a signal value.

The threshold-based `compute_signal()` in `markov_chain.py` (line 66-68):
```python
def compute_signal(next_state_probs: np.ndarray) -> float:
    """Compute directional signal: P(bull) - P(bear) in [-1, 1]."""
    return float(next_state_probs[2] - next_state_probs[0])
```

The HMM adapter returns a labeled `transition_matrix` and `stationary_distribution`, but never applies the same `P(bull) - P(bear)` formula. To get a signal from HMM:
- Use HMM's last-state row of its transition matrix as `next_state_probs`
- Call `compute_signal()` on that
- Or compute from HMM's stationary distribution

---

## 4. Is there any existing code path connecting HMM to backtesting?

**No.** The HMM result is purely an informational/display overlay in all three code paths:

| Code path | HMM usage |
|-----------|-----------|
| `pipeline.run()` | Returns HMM result as `output["hmm"]` dict key — no effect on signal, walk-forward, or forecasts |
| `cli.py` `_build_output()` | Same — HMM is stuffed into `output["hmm"]`, backtest uses only threshold regimes |
| `walk_forward_backtest()` | Uses `classify_regimes()`, `build_transition_matrix()`, `compute_signal()` — zero HMM involvement |
| Terminal/JSON output | HMM section is printed below the backtest section with a note about availability |

The `walk_forward_backtest()` function in `walk_forward.py` is entirely self-contained with threshold classification. It never imports or references `run_hmm_regime`.

---

## 5. The `hmm_source` vs `prices` split

### In `cli.py` — **split already exists**

The `_build_output()` function (line 75) accepts `hmm_source` as a separate parameter. The CLI's `main()` (line ~195-202) passes a full OHLCV DataFrame when `--ticker` is used:

```python
hmm_source = ohlcv_raw[["open", "high", "low", "close", "volume"]]
```

And `_build_output()` passes it through:
```python
hmm_data = hmm_source if hmm_source is not None else prices
hmm_result = run_hmm_regime(hmm_data, n_states=n_states, use_messina=use_messina)
```

**But this only works for `--ticker` mode.** For `--csv` mode, `hmm_source` is never set (stays `None`), so HMM gets just the price Series.

### In `pipeline.run()` — **no split exists**

`pipeline.run()` (line 46) has no `hmm_source` parameter and no `use_messina` parameter. It always passes `prices` (a Series) directly to `run_hmm_regime()`:

```python
hmm_result = run_hmm_regime(prices, n_states=n_states)
```

The `use_messina` flag is also absent from `pipeline.run()` — it's only in `cli.py`.

---

## 6. `use_hmm` and `use_messina` parameter flow

| Parameter | `cli.py` | `pipeline.run()` |
|-----------|----------|------------------|
| `use_hmm` | ✅ Fully plumbed via `--hmm`/`--no-hmm` | ✅ Present, defaults to `True` |
| `n_states` | ✅ Plumbed | ✅ Plumbed |
| `use_messina` | ✅ Plumbed via `--messina` | ❌ **Not present** — `pipeline.run()` never passes it |
| `hmm_source` | ✅ Available via `--ticker` (OHLCV), `None` for `--csv` | ❌ **Not present** — always just `prices` (Series) |

---

## 7. What a backtest-integration would need

### Minimum to swap threshold → HMM regimes:

1. **Index alignment** — `run_hmm_regime()` must return regimes aligned to the input index so they can be merged back to the price DataFrame.

2. **A HMM-based signal** — Either:
   - Add `compute_signal()` to the HMM result, or
   - Build a `hmm_transition_signal()` that feeds HMM's last-regime row into `compute_signal()`

### Minimum to walk-forward HMM:

3. **Per-window HMM fitting** — `run_hmm_regime()` fits on all data. For walk-forward, it needs to accept a training subset and produce (or update) a signal per new bar.
   - This implies calling `run_hmm_regime()` inside a loop with progressively larger data slices (like `walk_forward_backtest` does with `classify_regimes`)

4. **State-label stability** — The instability caveat means HMM refits on each window may shuffle state labels. Need to enforce consistent ordering (already done by log-return sorting, but re-fit on a small window could produce different state means).

### Plumbing gaps:

5. **Add `hmm_source` and `use_messina` to `pipeline.run()`** — currently missing, so API consumers can't pass rich OHLCV data or Messina features.

6. **Add `--csv` HMM source** — `hmm_source` is `None` for CSV path; `load_from_csv()` returns a full DataFrame that could be used.

---

## 8. Files that would need changes

| File | What to change |
|------|----------------|
| `scripts/regime/hmm_adapter.py` | Add index-preserving return; optionally add signal computation |
| `scripts/regime/pipeline.py` | Add `hmm_source`, `use_messina` params; optionally wire HMM into backtest path |
| `scripts/regime/walk_forward.py` | Add HMM-based option inside the loop (per-window fit) |
| `scripts/regime/markov_chain.py` | No changes needed — `compute_signal()` already works with any transition matrix |
| `scripts/cli.py` | CSV path could set `hmm_source`; otherwise well-plumbed |

---

## Summary Table

| Capability | Status |
|---|---|
| Produces bear/sideways/bull labels | ✅ Yes, but unaligned to index |
| Produces a signal [-1, 1] | ❌ No |
| Connected to backtest | ❌ No — purely display/analysis |
| Walk-forward compatible | ❌ No — full-data fit only |
| `hmm_source` split exists | ✅ In CLI, ⚠️ Not in `pipeline.run()` |
| `use_messina` flows through | ✅ In CLI, ❌ Not in `pipeline.run()` |
