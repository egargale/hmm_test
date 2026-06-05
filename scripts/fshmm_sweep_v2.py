"""FSHMMEngine sweep — saved incrementally, hmmlearn noise suppressed."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress hmmlearn noise
warnings.filterwarnings("ignore", message=".*not converging.*")
warnings.filterwarnings("ignore", message=".*Model is not converging.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HMMLEARN_VERBOSE"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import FSHMMConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run

TEST_DATA_DIR = Path("test_data/eval-results")
OUTPUT_DIR = Path("test_data/fshmm_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = sorted(p.stem for p in TEST_DATA_DIR.glob("*.csv") if not p.parent.name.startswith("_") and p.suffix == ".csv")
RESULTS_CSV = OUTPUT_DIR / "all_results.csv"

PARAM_GRID = {
    "n_states": [2, 3, 4, 5, "auto"],
    "pca_variance": [None, 0.90, 0.95, 0.99],
    "saliency_threshold": [0.3, 0.5, 0.7, 0.9],
    "dwell_bars": [0, 2, 5, 10],
    "hysteresis_delta": [0.0, 0.05, 0.1, 0.2],
}


def fmt_val(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 100:
            return f"{v:.4f}"
        return f"{v:.1f}"
    return str(v)


def build_config(**overrides):
    kw = {"n_states": 3, "pca_variance": None, "saliency_threshold": 0.5}
    kw.update(overrides)
    return FSHMMConfig(**kw)


def run_one(ticker, config, dwell=0, hysteresis=0.0):
    csv_file = TEST_DATA_DIR / f"{ticker}.csv"
    prices, ohlcv, source = load_prices(csv=str(csv_file))
    t0 = time.monotonic()
    result = pipeline_run(
        prices=prices,
        source=ticker,
        engine_config=config,
        min_train=252,
        ohlcv=ohlcv,
        dwell_bars=dwell,
        hysteresis_delta=hysteresis,
        duration_forecast=False,
    )
    elapsed = time.monotonic() - t0
    wf = result.walk_forward
    return {
        "ticker": ticker,
        "engine": config.name,
        "n_states": str(config.n_states),
        "pca_variance": str(config.pca_variance),
        "saliency_threshold": config.saliency_threshold,
        "dwell_bars": dwell,
        "hysteresis_delta": hysteresis,
        "regime": result.current_regime["name"],
        "signal": round(result.signal, 4),
        "sharpe": wf["sharpe"],
        "max_drawdown": wf["max_drawdown"],
        "n_trades": wf["n_trades"],
        "win_rate": wf["win_rate"],
        "profit_factor": wf["profit_factor"],
        "total_return": wf["total_return"],
        "wall_seconds": round(elapsed, 3),
        "verdict": result.verdict.get("verdict", "N/A"),
        "confidence": result.verdict.get("confidence", "N/A"),
        "degenerate_fit": result.engine_info.get("degenerate_fit", False),
    }


def append_result(row):
    """Append one row to the CSV, creating it if needed."""
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_results():
    """Load all results as list of dicts."""
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def already_done(ticker, n_states, pca, saliency, dwell, hyst):
    """Check if this exact combo already exists."""
    results = load_results()
    for r in results:
        if (r["ticker"] == ticker
            and r["n_states"] == str(n_states)
            and r["pca_variance"] == str(pca)
            and float(r["saliency_threshold"]) == float(saliency)
            and int(r["dwell_bars"]) == int(dwell)
            and float(r["hysteresis_delta"]) == float(hyst)):
            return True
    return False


def sweep():
    print(f"FSHMMEngine Sweep v2", file=sys.stderr)
    print(f"Tickers: {TICKERS}", file=sys.stderr)
    print(f"Output: {RESULTS_CSV}", file=sys.stderr)
    print(f"Existing results: {len(load_results())} rows", file=sys.stderr)
    print(file=sys.stderr)

    # Build all param combos to run
    combos = []

    # Phase 1: defaults (n_states=3, pca=None, st=0.5, dwell=2, hyst=0.05)
    defaults = FSHMMConfig()
    for ticker in TICKERS:
        combos.append(("default", ticker, defaults.n_states, defaults.pca_variance, defaults.saliency_threshold, defaults.default_dwell_bars, defaults.default_hysteresis_delta, {}))

    # Phase 2: single-param variations
    # n_states variations (with all other params at base defaults)
    base_ns_vals = [v for v in PARAM_GRID["n_states"] if str(v) != str(3)]  # not already in defaults
    for ticker in TICKERS:
        for ns in base_ns_vals:
            combos.append(("n_states", ticker, ns, defaults.pca_variance, defaults.saliency_threshold, defaults.default_dwell_bars, defaults.default_hysteresis_delta, {"n_states": ns}))

    # pca_variance variations
    for ticker in TICKERS:
        for pca in PARAM_GRID["pca_variance"]:
            if pca is None:
                continue
            combos.append(("pca_variance", ticker, defaults.n_states, pca, defaults.saliency_threshold, defaults.default_dwell_bars, defaults.default_hysteresis_delta, {"pca_variance": pca}))

    # saliency_threshold variations
    for ticker in TICKERS:
        for st in PARAM_GRID["saliency_threshold"]:
            if st == 0.5:
                continue
            combos.append(("saliency", ticker, defaults.n_states, defaults.pca_variance, st, defaults.default_dwell_bars, defaults.default_hysteresis_delta, {"saliency_threshold": st}))

    # dwell_bars variations
    for ticker in TICKERS:
        for dw in PARAM_GRID["dwell_bars"]:
            if dw == 2:
                continue  # default for fshmm is 2
            combos.append(("dwell", ticker, defaults.n_states, defaults.pca_variance, defaults.saliency_threshold, dw, defaults.default_hysteresis_delta, {"dwell": dw}))

    # hysteresis_delta variations
    for ticker in TICKERS:
        for hyst in PARAM_GRID["hysteresis_delta"]:
            if hyst == 0.05:
                continue  # default for fshmm is 0.05
            combos.append(("hysteresis", ticker, defaults.n_states, defaults.pca_variance, defaults.saliency_threshold, defaults.default_dwell_bars, hyst, {"hysteresis": hyst}))

    print(f"Total combos to run: {len(combos)}", file=sys.stderr)
    sys.stderr.flush()

    done = 0
    for i, (tag, ticker, n_states, pca, st, dwell, hyst, overrides) in enumerate(combos, 1):
        # Check if already done
        if already_done(ticker, n_states, pca, st, dwell, hyst):
            done += 1
            continue

        print(f"[{i}/{len(combos)}] {ticker} → {tag} (n_s={n_states}, pca={pca}, st={st}, dw={dwell}, hy={hyst})", file=sys.stderr)
        sys.stderr.flush()

        try:
            config = build_config(**{k: v for k, v in overrides.items() if k in ("n_states", "pca_variance", "saliency_threshold")})
            r = run_one(ticker, config, dwell=dwell, hysteresis=hyst)
            append_result(r)
            done += 1
            print(f"  ✓ sharpe={fmt_val(r['sharpe'])}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ {e}", file=sys.stderr)
            r = {
                "ticker": ticker,
                "engine": "fshmm",
                "n_states": str(n_states),
                "pca_variance": str(pca),
                "saliency_threshold": st,
                "dwell_bars": dwell,
                "hysteresis_delta": hyst,
                "regime": "ERROR",
                "signal": 0,
                "sharpe": None,
                "max_drawdown": None,
                "n_trades": 0,
                "win_rate": None,
                "profit_factor": None,
                "total_return": None,
                "wall_seconds": 0,
                "verdict": "error",
                "confidence": 0,
                "degenerate_fit": False,
            }
            append_result(r)

        sys.stderr.flush()

    print(f"\nDone. Total rows: {len(load_results())}", file=sys.stderr)


def analyze():
    results = load_results()
    print(f"Analyzing {len(results)} results...\n", file=sys.stderr)

    # Separate valid from errors
    valid = [r for r in results if r["sharpe"] not in (None, "N/A", "")]
    print(f"Valid results: {len(valid)}", file=sys.stderr)

    if not valid:
        print("No valid results to analyze.", file=sys.stderr)
        return

    # Convert sharpe to float
    for r in valid:
        r["sharpe_f"] = float(r["sharpe"])
        r["max_dd_f"] = float(r["max_drawdown"]) if r["max_drawdown"] not in (None, "N/A", "") else 0.0
        r["total_ret_f"] = float(r["total_return"]) if r["total_return"] not in (None, "N/A", "") else 0.0
        r["win_rate_f"] = float(r["win_rate"]) if r["win_rate"] not in (None, "N/A", "") else 0.0
        r["profit_f"] = float(r["profit_factor"]) if r["profit_factor"] not in (None, "N/A", "") else 0.0

    # =============================================
    # Build report
    # =============================================
    lines = []
    lines.append("# FSHMMEngine Sweep — Final Analysis & Judgment")
    lines.append("")
    lines.append(f"**Tickers tested**: {', '.join(TICKERS)}")
    lines.append(f"**Total configurations**: {len(valid)} valid / {len(results)} total")
    lines.append("")

    # Get defaults row for each ticker
    lines.append("## Phase 1: Default Parameters (n_states=3, pca=None, saliency=0.5, dwell=2, hyst=0.05)")
    lines.append("")
    lines.append("| Ticker | Regime | Sharpe | MaxDD | Trades | WinRate | ProfitFactor | TotalRet | Verdict |")
    lines.append("|--------|--------|--------|-------|--------|---------|-------------|----------|---------|")
    for t in TICKERS:
        match = [r for r in valid if r["ticker"] == t and r["n_states"] == "3" and r["pca_variance"] == "None" and float(r["saliency_threshold"]) == 0.5 and int(r["dwell_bars"]) == 2 and float(r["hysteresis_delta"]) == 0.05]
        if match:
            r = match[0]
            lines.append(f"| {r['ticker']} | {r['regime']} | {fmt_val(r['sharpe_f'])} | {fmt_val(r['max_dd_f'])} | {r['n_trades']} | {fmt_val(r['win_rate_f'])} | {fmt_val(r['profit_f'])} | {fmt_val(r['total_ret_f'])} | {r['verdict']} |")
    lines.append("")

    # Best per ticker
    lines.append("## Phase 2: Best Configuration Per Ticker")
    lines.append("")
    lines.append("| Ticker | Best Sharpe | Best Config | Worst Sharpe | Worst Config |")
    lines.append("|--------|-------------|-------------|--------------|--------------|")
    for t in sorted(TICKERS):
        tr = [r for r in valid if r["ticker"] == t]
        if not tr:
            continue
        best = max(tr, key=lambda r: r["sharpe_f"])
        worst = min(tr, key=lambda r: r["sharpe_f"])
        bcfg = f"n_s={best['n_states']} pca={best['pca_variance']} st={best['saliency_threshold']} dw={best['dwell_bars']} hy={best['hysteresis_delta']}"
        wcfg = f"n_s={worst['n_states']} pca={worst['pca_variance']} st={worst['saliency_threshold']} dw={worst['dwell_bars']} hy={worst['hysteresis_delta']}"
        lines.append(f"| {t} | {fmt_val(best['sharpe_f'])} | {bcfg} | {fmt_val(worst['sharpe_f'])} | {wcfg} |")
    lines.append("")

    # Top 10 overall
    sorted_all = sorted(valid, key=lambda r: r["sharpe_f"], reverse=True)
    lines.append("## Top 10 Configurations Overall (by Sharpe)")
    lines.append("")
    lines.append("| # | Ticker | Sharpe | n_states | pca | saliency | dwell | hyst | verdict |")
    lines.append("|---|--------|--------|----------|-----|----------|-------|------|---------|")
    for i, r in enumerate(sorted_all[:10]):
        lines.append(f"| {i+1} | {r['ticker']} | {fmt_val(r['sharpe_f'])} | {r['n_states']} | {r['pca_variance']} | {fmt_val(float(r['saliency_threshold']))} | {r['dwell_bars']} | {fmt_val(float(r['hysteresis_delta']))} | {r['verdict']} |")
    lines.append("")

    # Bottom 10
    sorted_bottom = sorted(valid, key=lambda r: r["sharpe_f"])
    lines.append("## Bottom 10 Configurations (by Sharpe)")
    lines.append("")
    lines.append("| # | Ticker | Sharpe | n_states | pca | saliency | dwell | hyst |")
    lines.append("|---|--------|--------|----------|-----|----------|-------|------|")
    for i, r in enumerate(sorted_bottom[:10]):
        lines.append(f"| {i+1} | {r['ticker']} | {fmt_val(r['sharpe_f'])} | {r['n_states']} | {r['pca_variance']} | {fmt_val(float(r['saliency_threshold']))} | {r['dwell_bars']} | {fmt_val(float(r['hysteresis_delta']))} |")
    lines.append("")

    # Parameter impact
    lines.append("## Parameter Impact Analysis")
    lines.append("")

    KEY_MAP = {
        "n_states": ("n_states", lambda r, v: r["n_states"] == str(v)),
        "pca_variance": ("pca_variance", lambda r, v: str(r["pca_variance"]) == str(v)),
        "saliency_threshold": ("saliency_threshold", lambda r, v: abs(float(r["saliency_threshold"]) - float(v)) < 0.001),
        "dwell_bars": ("dwell_bars", lambda r, v: int(r["dwell_bars"]) == int(v)),
        "hysteresis_delta": ("hysteresis_delta", lambda r, v: abs(float(r["hysteresis_delta"]) - float(v)) < 0.001),
    }

    for param_name, grid_values in PARAM_GRID.items():
        colname, matcher = KEY_MAP[param_name]
        lines.append(f"### {param_name}")
        lines.append("")
        lines.append("| Value | Mean Sharpe | Std Sharpe | Min Sharpe | Max Sharpe | N Runs |")
        lines.append("|-------|------------|------------|------------|------------|--------|")

        for val in grid_values:
            matching = [r for r in valid if matcher(r, val)]
            if not matching:
                continue
            sharpes = [r["sharpe_f"] for r in matching]
            mean_s = sum(sharpes) / len(sharpes)
            if len(sharpes) > 1:
                std_s = (sum((s - mean_s)**2 for s in sharpes) / (len(sharpes) - 1))**0.5
            else:
                std_s = 0.0
            min_s = min(sharpes)
            max_s = max(sharpes)
            lines.append(f"| {val} | {mean_s:.4f} | {std_s:.4f} | {min_s:.4f} | {max_s:.4f} | {len(sharpes)} |")
        lines.append("")

    # Final judgment
    lines.append("## Final Judgment")
    lines.append("")

    # Default vs best
    default_sharpes = []
    for t in TICKERS:
        match = [r for r in valid if r["ticker"] == t and r["n_states"] == "3" and r["pca_variance"] == "None" and float(r["saliency_threshold"]) == 0.5 and int(r["dwell_bars"]) == 2 and float(r["hysteresis_delta"]) == 0.05]
        if match:
            default_sharpes.append(match[0]["sharpe_f"])

    lines.append("### Default Configuration Performance")
    lines.append("")
    if default_sharpes:
        default_mean = sum(default_sharpes) / len(default_sharpes)
        lines.append(f"- Mean Sharpe across all tickers: **{default_mean:.4f}**")
        lines.append(f"- Best ticker default Sharpe: **{max(default_sharpes):.4f}**")
        lines.append(f"- Worst ticker default Sharpe: **{min(default_sharpes):.4f}**")
    lines.append("")

    # Best per-ticker compared to default
    lines.append("### Best Configuration vs Default")
    lines.append("")
    improvement_count = 0
    total_improvement = 0.0
    for t in sorted(TICKERS):
        tr = [r for r in valid if r["ticker"] == t]
        if not tr:
            continue
        best = max(tr, key=lambda r: r["sharpe_f"])
        # Find default
        default = [r for r in tr if r["n_states"] == "3" and r["pca_variance"] == "None" and float(r["saliency_threshold"]) == 0.5 and int(r["dwell_bars"]) == 2 and float(r["hysteresis_delta"]) == 0.05]
        if default:
            def_sharpe = default[0]["sharpe_f"]
            imp = best["sharpe_f"] - def_sharpe
            improvement_count += 1
            total_improvement += imp
            bc = f"n_s={best['n_states']} pca={best['pca_variance']} st={best['saliency_threshold']} dw={best['dwell_bars']} hy={best['hysteresis_delta']}"
            lines.append(f"- **{t}**: default={fmt_val(def_sharpe)} → best={fmt_val(best['sharpe_f'])} ({imp:+.4f}) via {bc}")
    if improvement_count > 0:
        lines.append(f"\nAverage improvement from tuning: **{total_improvement/improvement_count:+.4f}** Sharpe")
    lines.append("")

    # Per-parameter recommendation
    lines.append("### Parameter Recommendations (by mean Sharpe)")
    lines.append("")
    param_recs = {}
    for param_name, grid_values in PARAM_GRID.items():
        colname, matcher = KEY_MAP[param_name]
        best_val = None
        best_mean = float("-inf")
        best_vol = None
        for val in grid_values:
            matching = [r for r in valid if matcher(r, val) and r["sharpe_f"] > -999]
            if not matching:
                continue
            sharpes = [r["sharpe_f"] for r in matching]
            mean_s = sum(sharpes) / len(sharpes)
            if mean_s > best_mean:
                best_mean = mean_s
                best_val = val
        param_recs[param_name] = (best_val, best_mean)

    for pname, (bval, bmean) in param_recs.items():
        lines.append(f"- **{pname}**: best = **{bval}** (mean Sharpe = {bmean:.4f})")
    lines.append("")

    # ------- Stable-aggregate table -------
    lines.append("### Per-Ticker Stability Analysis")
    lines.append("")
    lines.append("| Ticker | Default Sharpe | Best Sharpe | Worst Sharpe | Sharpe Range | N Configs |")
    lines.append("|--------|----------------|-------------|--------------|--------------|-----------|")
    for t in sorted(TICKERS):
        tr = [r for r in valid if r["ticker"] == t]
        if not tr:
            continue
        def_r = [r for r in tr if r["n_states"] == "3" and r["pca_variance"] == "None" and float(r["saliency_threshold"]) == 0.5 and int(r["dwell_bars"]) == 2 and float(r["hysteresis_delta"]) == 0.05]
        best = max(tr, key=lambda r: r["sharpe_f"])
        worst = min(tr, key=lambda r: r["sharpe_f"])
        range_s = best["sharpe_f"] - worst["sharpe_f"]
        def_s = fmt_val(def_r[0]["sharpe_f"]) if def_r else "N/A"
        lines.append(f"| {t} | {def_s} | {fmt_val(best['sharpe_f'])} | {fmt_val(worst['sharpe_f'])} | {range_s:.4f} | {len(tr)} |")
    lines.append("")

    # Conclusion
    lines.append("### Conclusion")
    lines.append("")

    # Count how many tickers improved with non-default params
    improved = []
    worsened = []
    for t in sorted(TICKERS):
        tr = [r for r in valid if r["ticker"] == t]
        if not tr:
            continue
        def_r = [r for r in tr if r["n_states"] == "3" and r["pca_variance"] == "None" and float(r["saliency_threshold"]) == 0.5 and int(r["dwell_bars"]) == 2 and float(r["hysteresis_delta"]) == 0.05]
        if not def_r:
            continue
        best = max(tr, key=lambda r: r["sharpe_f"])
        if best["sharpe_f"] > def_r[0]["sharpe_f"]:
            improved.append((t, best["sharpe_f"] - def_r[0]["sharpe_f"]))
        elif best["sharpe_f"] < def_r[0]["sharpe_f"]:
            worsened.append((t, def_r[0]["sharpe_f"] - best["sharpe_f"]))

    lines.append(f"- **{len(improved)}/{len(TICKERS)}** tickers saw improvement with parameter tuning")
    if improved:
        lines.append(f"  - Biggest winner: **{improved[-1][0]}** (+{improved[-1][1]:.4f} Sharpe)")
    if worsened:
        lines.append(f"- **{len(worsened)}/{len(TICKERS)}** tickers had worse results with non-default params")
    lines.append("")

    # Is FSHMMEngine robust to parameter changes?
    all_sharpes = [r["sharpe_f"] for r in valid]
    lines.append(f"- **Overall Sharpe range across all configs**: {min(all_sharpes):.4f} to {max(all_sharpes):.4f}")
    lines.append(f"- **Overall mean Sharpe**: {sum(all_sharpes)/len(all_sharpes):.4f}")
    lines.append(f"- **Coefficient of variation**: {(sum((s - sum(all_sharpes)/len(all_sharpes))**2 for s in all_sharpes)/len(all_sharpes))**0.5 / (sum(all_sharpes)/len(all_sharpes) + 1e-10):.4f}")
    lines.append("")

    report = "\n".join(lines)

    report_path = OUTPUT_DIR / "phase3_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(report, file=sys.stderr)
    print(f"\nReport saved to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze()
    else:
        sweep()
        analyze()
