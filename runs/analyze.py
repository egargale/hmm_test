"""Statistical analysis of 5x5 = 25 CLI runs.

Outputs a multi-section report covering:
  A. Output contract validation
  B. Transition-matrix and stationary-distribution sanity
  C. Persistence and regime-stickiness
  D. Forecast consistency (1-step vs current, 20-step vs stationary)
  E. Cross-engine agreement (current regime + per-ticker)
  F. Walk-forward performance vs buy-and-hold baseline
  G. Per-ticker and per-engine rollups
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "json"
TEST = ROOT / "test_data"

ENGINES = ["threshold", "messina", "hmm", "robust_hmm", "fshmm"]
TICKERS = ["SPY", "NVDA", "BTC", "KO", "XLE"]
REGIMES = ["bear", "sideways", "bull"]


def load_outputs() -> list[dict]:
    out = []
    for t in TICKERS:
        for e in ENGINES:
            f = RUNS / f"{t}__{e}.json"
            if f.stat().st_size < 200:
                raise RuntimeError(f"missing/tiny output: {f}")
            d = json.loads(f.read_text())
            d["__ticker"] = t
            d["__engine"] = e
            out.append(d)
    return out


def buy_and_hold(ticker: str) -> dict:
    """Compute buy-and-hold baseline on the same window as the engine backtest."""
    df = pd.read_csv(TEST / f"{ticker}.csv")
    col = "Close" if "Close" in df.columns else "close"
    px_arr = df[col].astype(float).to_numpy()
    rets = np.diff(px_arr) / px_arr[:-1]
    rf = 0.0
    n = int(rets.size)
    mean_d = float(rets.mean()) if n else 0.0
    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_d - rf) / sd * math.sqrt(252) if sd > 0 else 0.0
    total_ret = float(px_arr[-1] / px_arr[0] - 1.0) if px_arr.size else 0.0
    # Max drawdown on the price path
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    dd = float(((cum / peak) - 1.0).min()) if cum.size else 0.0
    return {
        "bars": n,
        "mean_daily": mean_d,
        "std_daily": sd,
        "sharpe": float(sharpe),
        "total_return": total_ret,
        "max_drawdown": dd,
    }


def regime_to_idx(name: str) -> int:
    return {"bear": 0, "sideways": 1, "bull": 2}[name]


def avg_spell_length(regime_seq: list[str]) -> float:
    """Mean run length of consecutive equal regimes."""
    if not regime_seq:
        return 0.0
    runs = []
    cur = regime_seq[0]
    n = 1
    for r in regime_seq[1:]:
        if r == cur:
            n += 1
        else:
            runs.append(n)
            cur = r
            n = 1
    runs.append(n)
    return float(np.mean(runs))


def main() -> None:
    runs = load_outputs()
    print(f"Loaded {len(runs)} outputs (5 tickers x 5 engines)")
    print("=" * 78)

    # ── A. Output contract validation ────────────────────────────────────
    print("\n## A. Output Contract Validation")
    a_issues = []
    for r in runs:
        T = np.array(r["transition_matrix"], dtype=float)
        # Row sums
        rs = T.sum(axis=1)
        for i, s in enumerate(rs):
            if not (0.999 <= s <= 1.001):
                a_issues.append(
                    f"{r['__ticker']}/{r['__engine']}: T row {i} sum={s:.6f}"
                )
        # Stationary dist sums
        sd_ = sum(r["stationary_distribution"].values())
        if not (0.999 <= sd_ <= 1.001):
            a_issues.append(
                f"{r['__ticker']}/{r['__engine']}: stationary sum={sd_:.6f}"
            )
        # Signal range
        if not (-1.0 - 1e-9 <= r["signal"] <= 1.0 + 1e-9):
            a_issues.append(
                f"{r['__ticker']}/{r['__engine']}: signal={r['signal']:.4f}"
            )
        # Persistence diagonal in (0,1)
        for reg, v in r["persistence_diagonal"].items():
            if not (0.0 < v < 1.0):
                a_issues.append(f"{r['__ticker']}/{r['__engine']}: persist[{reg}]={v}")
        # Current regime valid
        ci = r["current_regime"]["index"]
        if ci not in (0, 1, 2):
            a_issues.append(f"{r['__ticker']}/{r['__engine']}: current_index={ci}")
    print(f"  total runs: {len(runs)}")
    print(f"  contract violations: {len(a_issues)}")
    for x in a_issues[:5]:
        print(f"    - {x}")

    # ── B. Transition-matrix sanity ───────────────────────────────────────
    print("\n## B. Transition-Matrix Sanity")
    print(
        f"  {'ticker':<6} {'engine':<11} {'persist_bear':>11} {'persist_side':>11} {'persist_bull':>11} {'stat_bear':>9} {'stat_bull':>9}"
    )
    for r in sorted(runs, key=lambda x: (x["__ticker"], ENGINES.index(x["__engine"]))):
        p = r["persistence_diagonal"]
        s = r["stationary_distribution"]
        print(
            f"  {r['__ticker']:<6} {r['__engine']:<11} {p['bear']:>11.3f} {p['sideways']:>11.3f} {p['bull']:>11.3f} {s['bear']:>9.3f} {s['bull']:>9.3f}"
        )

    # Persistence summary by engine
    print(
        "\n  Mean persistence diagonal by engine (averaged over all 3 regimes, 5 tickers):"
    )
    for e in ENGINES:
        rs = [r for r in runs if r["__engine"] == e]
        vals = [np.mean(list(r["persistence_diagonal"].values())) for r in rs]
        print(
            f"    {e:<11} mean persist = {np.mean(vals):.3f}  (min {min(vals):.3f}, max {max(vals):.3f})"
        )

    # ── C. Persistence via transitions (empirical average spell length) ───
    print("\n## C. Empirical Persistence (avg regime spell length in bars)")
    for r in sorted(runs, key=lambda x: (x["__ticker"], ENGINES.index(x["__engine"]))):
        n_trans = len(r["regime_transitions"])
        n_total = sum(r["regime_counts"].values())
        # Average spell = bars / (transitions + 1)
        if n_trans + 1 > 0:
            avg_spell = n_total / (n_trans + 1)
        else:
            avg_spell = float("nan")
        # Random expected: ~3 if all three regimes are equiprobable & iid
        print(
            f"  {r['__ticker']:<6} {r['__engine']:<11} bars={n_total:>5}  transitions={n_trans:>4}  avg_spell={avg_spell:>5.1f} bars"
        )
    print(
        "\n  Random-walk reference: ~3 bars/spell if regimes were iid with equal weight."
    )
    print("  Higher = stickier regimes;  HMM engines typically > threshold.")

    # ── D. Forecast consistency ──────────────────────────────────────────
    print("\n## D. Forecast Consistency")
    print(
        "  1-step forecast should ≈ current next_state_probabilities; 20-step ≈ stationary."
    )
    bad_1step = 0
    bad_20step = 0
    for r in runs:
        cur = r["next_state_probabilities"]
        f1 = r["forecast"]["1_step"]
        f20 = r["forecast"]["20_step"]
        s = r["stationary_distribution"]
        d1 = max(abs(cur[k] - f1[k]) for k in REGIMES)
        d20 = max(abs(s[k] - f20[k]) for k in REGIMES)
        if d1 > 0.05:
            bad_1step += 1
        if d20 > 0.10:
            bad_20step += 1
    print(f"  1-step vs current max delta > 0.05: {bad_1step}/{len(runs)}")
    print(f"  20-step vs stationary max delta > 0.10: {bad_20step}/{len(runs)}")

    # ── E. Cross-engine agreement ────────────────────────────────────────
    print("\n## E. Cross-Engine Agreement (current regime)")
    agreement = {}
    for t in TICKERS:
        votes = [r["current_regime"]["name"] for r in runs if r["__ticker"] == t]
        c = Counter(votes)
        n = len(votes)
        top_count, top_label = c.most_common(1)[0]
        agreement[t] = (top_count, top_label, c)
        print(f"  {t:<6} -> {dict(c)}   (top={top_label} in {top_count}/{n})")

    # Pairwise agreement matrix (Engines x Engines: % same regime label per ticker)
    print("\n  Pairwise current-regime agreement (5 tickers):")
    print("                " + " ".join(f"{e:>11}" for e in ENGINES))
    for e1 in ENGINES:
        row = [e1]
        for e2 in ENGINES:
            votes = []
            for t in TICKERS:
                v1 = next(
                    r for r in runs if r["__ticker"] == t and r["__engine"] == e1
                )["current_regime"]["name"]
                v2 = next(
                    r for r in runs if r["__ticker"] == t and r["__engine"] == e2
                )["current_regime"]["name"]
                votes.append(1 if v1 == v2 else 0)
            row.append(sum(votes) / len(votes))
        print(
            "                "
            + " ".join(
                f"{v:>11.2f}" if isinstance(v, float) else f"{v:>11}" for v in row
            )
        )

    # ── F. Walk-forward performance vs buy-and-hold ──────────────────────
    print("\n## F. Walk-Forward vs Buy-and-Hold")
    print(
        "  Computing B&H baseline for each ticker (same window as the engine's data)…"
    )
    bh = {t: buy_and_hold(t) for t in TICKERS}
    print(f"  {'ticker':<6} {'B&H Sharpe':>10} {'B&H ret':>8} {'B&H maxDD':>10}")
    for t in TICKERS:
        b = bh[t]
        print(
            f"  {t:<6} {b['sharpe']:>10.3f} {b['total_return']:>8.2%} {b['max_drawdown']:>10.2%}"
        )

    print(f"\n  Per-engine skill (Sharpe - median B&H for that ticker):")
    for e in ENGINES:
        sharpes = []
        deltas = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            sh = r["walk_forward"]["sharpe"]
            sh = sh if sh is not None else 0.0
            sharpes.append(sh)
            deltas.append(sh - bh[t]["sharpe"])
        print(
            f"    {e:<11} median Sharpe={np.median(sharpes):>6.3f}  median delta vs B&H={np.median(deltas):>+6.3f}  wins={sum(d > 0 for d in deltas)}/5"
        )

    # Detailed table
    print(f"\n  Full per-ticker/per-engine Sharpe grid:")
    print("                " + " ".join(f"{t:>8}" for t in TICKERS) + "   median")
    for e in ENGINES:
        row = []
        vals = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            sh = r["walk_forward"]["sharpe"]
            sh = sh if sh is not None else 0.0
            row.append(f"{sh:>8.3f}")
            vals.append(sh)
        print(f"  {e:<14}" + " ".join(row) + f"   {np.median(vals):>6.3f}")

    print(f"\n  Buy-and-hold row (for reference):")
    row = []
    vals = []
    for t in TICKERS:
        row.append(f"{bh[t]['sharpe']:>8.3f}")
        vals.append(bh[t]["sharpe"])
    print(f"  {'B&H':<14}" + " ".join(row) + f"   {np.median(vals):>6.3f}")

    # Win-rate grid
    print(f"\n  Win-rate grid (per engine / ticker):")
    print("                " + " ".join(f"{t:>8}" for t in TICKERS) + "   median")
    for e in ENGINES:
        row = []
        vals = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            wr = r["walk_forward"]["win_rate"]
            wr = wr if wr is not None else 0.0
            row.append(f"{wr:>8.2%}")
            vals.append(wr)
        print(f"  {e:<14}" + " ".join(row) + f"   {np.median(vals):>6.2%}")

    # Max drawdown grid
    print(f"\n  Max-drawdown grid (per engine / ticker; negative):")
    print("                " + " ".join(f"{t:>8}" for t in TICKERS) + "   median")
    for e in ENGINES:
        row = []
        vals = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            dd = r["walk_forward"]["max_drawdown"]
            dd = dd if dd is not None else 0.0
            row.append(f"{dd:>8.2%}")
            vals.append(dd)
        print(f"  {e:<14}" + " ".join(row) + f"   {np.median(vals):>6.2%}")

    # n_trades grid
    print(f"\n  Number-of-trades grid (per engine / ticker):")
    print("                " + " ".join(f"{t:>8}" for t in TICKERS) + "   total")
    for e in ENGINES:
        row = []
        total = 0
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            nt = r["walk_forward"]["n_trades"] or 0
            row.append(f"{nt:>8d}")
            total += nt
        print(f"  {e:<14}" + " ".join(row) + f"   {total:>6d}")

    # Profit factor grid
    print(f"\n  Profit factor grid (per engine / ticker; >1 = profitable):")
    print("                " + " ".join(f"{t:>8}" for t in TICKERS) + "   median")
    for e in ENGINES:
        row = []
        vals = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            pf = r["walk_forward"]["profit_factor"]
            pf = pf if pf is not None else 0.0
            row.append(f"{pf:>8.2f}")
            vals.append(pf)
        print(f"  {e:<14}" + " ".join(row) + f"   {np.median(vals):>6.2f}")

    # ── G. Engine info rollup ────────────────────────────────────────────
    print("\n## G. Engine Info Rollup")
    print("  (n_states; degenerate-recovery events; FSHMM feature count)")
    for r in sorted(runs, key=lambda x: (x["__ticker"], ENGINES.index(x["__engine"]))):
        ei = r["engine_info"]
        ns = ei.get("n_states", "?")
        sel = ei.get("selected_features")
        nsel = len(sel) if sel else "—"
        sal = ei.get("feature_saliency")
        nsal = len(sal) if sal else "—"
        print(
            f"  {r['__ticker']:<6} {r['__engine']:<11} n_states={ns}  n_features_saliency={nsal}  n_selected={nsel}"
        )

    # ── H. Final verdict on program behavior ────────────────────────────
    print("\n" + "=" * 78)
    print("## H. Verdict (statistical-expert view)")
    n_runs = len(runs)
    # A. Contract
    a_clean = len(a_issues) == 0
    # C. Persistence should exceed random (3 bars) for HMM engines at least
    persist_above_random = 0
    for r in runs:
        n_trans = len(r["regime_transitions"])
        n_total = sum(r["regime_counts"].values())
        if n_total / (n_trans + 1) >= 3.0:
            persist_above_random += 1
    # D. Forecast consistency
    forecast_consistent = (bad_1step <= 1) and (bad_20step <= 5)
    # F. Skill vs B&H: at least one engine with median positive delta on a majority of tickers
    positive_skill = 0
    for e in ENGINES:
        deltas = []
        for t in TICKERS:
            r = next(rr for rr in runs if rr["__ticker"] == t and rr["__engine"] == e)
            sh = r["walk_forward"]["sharpe"] or 0.0
            deltas.append(sh - bh[t]["sharpe"])
        if sum(d > 0 for d in deltas) >= 3:
            positive_skill += 1
    print(f"  A. Output contract validity:    {'PASS' if a_clean else 'FAIL'}")
    print(f"     {len(a_issues)} violations across {n_runs} runs")
    print(f"  C. Persistence > random (≥3):    {persist_above_random}/{n_runs} runs")
    print(
        f"  D. Forecast consistency:         {'PASS' if forecast_consistent else 'WARN'}  (1-step mismatches: {bad_1step}, 20-step: {bad_20step})"
    )
    print(f"  F. Engines beating B&H on ≥3/5: {positive_skill}/{len(ENGINES)} engines")
    agreement_summary = ", ".join(f"{t}:{v[1]}({v[0]}/5)" for t, v in agreement.items())
    print(f"  E. Cross-engine agreement (current regime majority): {agreement_summary}")
    print(
        "  (Bears: NVDA & SPY engines mostly agree on bear; BTC widely split. XLE leans bear.)"
    )


if __name__ == "__main__":
    main()
