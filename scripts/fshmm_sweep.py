"""FSHMMEngine sweep: defaults → parameter grid → final judgment.

Phase 1: Run FSHMMEngine with default parameters on every CSV in test_data/eval-results/.
Phase 2: Sweep all available FSHMMEngine parameters.
Phase 3: Analyze results and make final judgment.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmm_futures_analysis.data_processing.csv_auto_detect import load_prices
from hmm_futures_analysis.regime.engine_configs import FSHMMConfig
from hmm_futures_analysis.regime.pipeline import run as pipeline_run
from hmm_futures_analysis.regime.engine_protocol import resolve_engine


TEST_DATA_DIR = Path("test_data/eval-results")
OUTPUT_DIR = Path("test_data/fshmm_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = sorted(p.stem for p in TEST_DATA_DIR.glob("*.csv") if not p.parent.name.startswith("_"))

# ── Parameter grid for Phase 2 ──────────────────────────────────────────
PARAM_GRID = {
    "n_states": [2, 3, 4, 5, "auto"],
    "pca_variance": [None, 0.90, 0.95, 0.99],
    "saliency_threshold": [0.3, 0.5, 0.7, 0.9],
    "dwell_bars": [0, 2, 5, 10],
    "hysteresis_delta": [0.0, 0.05, 0.1, 0.2],
}


def build_config(**overrides) -> FSHMMConfig:
    """Build FSHMMConfig with default-override semantics."""
    kw = {
        "n_states": 3,
        "pca_variance": None,
        "saliency_threshold": 0.5,
    }
    kw.update(overrides)
    return FSHMMConfig(**kw)


def run_one(ticker: str, config: FSHMMConfig, dwell: int = 0, hysteresis: float = 0.0) -> dict:
    """Run pipeline for one ticker + config combo, return summary dict."""
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
        "n_features_used": result.engine_info.get("n_features", None),
    }


def fmt_val(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 100:
            return f"{v:.4f}"
        return f"{v:.1f}"
    return str(v)


def format_results(results: list[dict], sort_key: str = "sharpe") -> str:
    """Format results as markdown table, sorted by sort_key descending."""
    if not results:
        return "No results."

    headers = [
        "ticker", "regime", "n_states", "pca", "saliency",
        "dwell", "hyst", "sharpe", "max_dd", "trades",
        "win_rate", "profit", "total_ret", "verdict", "conf", "wall_s"
    ]

    def get_val(r, key):
        v = r.get(key, "N/A")
        if v is None or v == "N/A":
            return -999  # sort NaN to bottom
        return float(v)

    sorted_results = sorted(
        results,
        key=lambda r: get_val(r, sort_key),
        reverse=True,
    )

    rows = []
    for r in sorted_results:
        rows.append([
            r["ticker"],
            r["regime"],
            r["n_states"],
            r["pca_variance"],
            fmt_val(r["saliency_threshold"]),
            str(r["dwell_bars"]),
            fmt_val(r["hysteresis_delta"]),
            fmt_val(r["sharpe"]),
            fmt_val(r["max_drawdown"]),
            str(r["n_trades"]),
            fmt_val(r["win_rate"]),
            fmt_val(r["profit_factor"]),
            fmt_val(r["total_return"]),
            r["verdict"],
            fmt_val(r["confidence"]),
            fmt_val(r["wall_seconds"]),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    return "\n".join(lines)


def save_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        if not results:
            return
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Defaults on all tickers
# ═══════════════════════════════════════════════════════════════════════
def phase1() -> list[dict]:
    print("=" * 72, file=sys.stderr)
    print("PHASE 1: FSHMMEngine defaults on all tickers", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    results = []
    defaults = FSHMMConfig()

    for ticker in TICKERS:
        print(f"  {ticker} ...", file=sys.stderr)
        try:
            r = run_one(ticker, defaults, dwell=defaults.default_dwell_bars, hysteresis=defaults.default_hysteresis_delta)
            results.append(r)
            print(f"    → sharpe={fmt_val(r['sharpe'])}, regime={r['regime']}, verdict={r['verdict']}", file=sys.stderr)
        except Exception as e:
            print(f"    ✗ FAILED: {e}", file=sys.stderr)
            results.append({
                "ticker": ticker,
                "engine": "fshmm",
                "n_states": str(defaults.n_states),
                "pca_variance": str(defaults.pca_variance),
                "saliency_threshold": defaults.saliency_threshold,
                "dwell_bars": defaults.default_dwell_bars,
                "hysteresis_delta": defaults.default_hysteresis_delta,
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
                "n_features_used": None,
            })

    path = OUTPUT_DIR / "phase1_defaults.csv"
    save_csv(results, path)
    print(f"\nPhase 1 results → {path}", file=sys.stderr)
    print(format_results(results), file=sys.stderr)
    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Full parameter sweep
# ═══════════════════════════════════════════════════════════════════════
def phase2(defaults_results: list[dict]) -> list[dict]:
    print("\n" + "=" * 72, file=sys.stderr)
    print("PHASE 2: FSHMMEngine parameter sweep", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    # Build parameter list (pairs that don't conflict)
    param_list = []

    # n_states alone (with all else default)
    for ns in PARAM_GRID["n_states"]:
        param_list.append(("n_states", ns))

    # pca_variance alone
    for pca in PARAM_GRID["pca_variance"]:
        if pca is None:
            continue  # already covered by defaults
        param_list.append(("pca_variance", pca))

    # saliency_threshold alone
    for st in PARAM_GRID["saliency_threshold"]:
        if st == 0.5:
            continue  # default
        param_list.append(("saliency_threshold", st))

    # dwell_bars alone
    for dw in PARAM_GRID["dwell_bars"]:
        if dw == 0:
            continue  # no-dwell is default-like (engine default is 2)
        param_list.append(("dwell_bars", dw))

    # hysteresis_delta alone
    for hyst in PARAM_GRID["hysteresis_delta"]:
        if hyst == 0.0:
            continue  # no-hysteresis is default-like
        param_list.append(("hysteresis_delta", hyst))

    all_results = list(defaults_results)

    total = len(TICKERS) * len(param_list)
    done = 0

    for param_name, param_val in param_list:
        for ticker in TICKERS:
            done += 1
            print(f"  [{done}/{total}] {ticker} → {param_name}={param_val}", file=sys.stderr)

            defaults = FSHMMConfig()
            dwell = defaults.default_dwell_bars
            hysteresis = defaults.default_hysteresis_delta

            config_kw = {}

            if param_name == "n_states":
                config_kw["n_states"] = param_val
            elif param_name == "pca_variance":
                config_kw["pca_variance"] = param_val
            elif param_name == "saliency_threshold":
                config_kw["saliency_threshold"] = param_val
            elif param_name == "dwell_bars":
                dwell = param_val
            elif param_name == "hysteresis_delta":
                hysteresis = param_val

            config = build_config(**config_kw)

            try:
                r = run_one(ticker, config, dwell=dwell, hysteresis=hysteresis)
                all_results.append(r)
            except Exception as e:
                print(f"    ✗ FAILED: {e}", file=sys.stderr)
                all_results.append({
                    "ticker": ticker,
                    "engine": "fshmm",
                    "n_states": str(config.n_states),
                    "pca_variance": str(config.pca_variance),
                    "saliency_threshold": config.saliency_threshold,
                    "dwell_bars": dwell,
                    "hysteresis_delta": hysteresis,
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
                    "n_features_used": None,
                })

    path = OUTPUT_DIR / "phase2_sweep.csv"
    save_csv(all_results, path)
    print(f"\nPhase 2 results → {path}", file=sys.stderr)
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Analysis & Judgment
# ═══════════════════════════════════════════════════════════════════════
def phase3(all_results: list[dict]) -> str:
    print("\n" + "=" * 72, file=sys.stderr)
    print("PHASE 3: Analysis & Final Judgment", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    # Find best and worst per ticker
    per_ticker_best = {}
    for r in all_results:
        t = r["ticker"]
        if t not in per_ticker_best:
            per_ticker_best[t] = {"best": None, "worst": None}
        s = r.get("sharpe")
        if s is None:
            continue
        b = per_ticker_best[t]["best"]
        w = per_ticker_best[t]["worst"]
        if b is None or s > b["sharpe"]:
            per_ticker_best[t]["best"] = {**r, "sharpe": s}
        if w is None or s < w["sharpe"]:
            per_ticker_best[t]["worst"] = {**r, "sharpe": s}

    # Build analysis markdown
    lines = []
    lines.append("# FSHMMEngine Sweep — Final Analysis & Judgment")
    lines.append("")
    lines.append(f"Run against {len(TICKERS)} tickers: {', '.join(TICKERS)}")
    lines.append("")
    lines.append(f"Total configurations tested: {len(all_results)}")
    lines.append("")

    lines.append("## Phase 1: Default Parameters")
    lines.append("")
    lines.append("| Ticker | Regime | Sharpe | MaxDD | Trades | WinRate | ProfitFactor | TotalRet | Verdict |")
    lines.append("|--------|--------|--------|-------|--------|---------|-------------|----------|---------|")
    defaults = [r for r in all_results if r["saliency_threshold"] == 0.5 and r["n_states"] == "3" and r["pca_variance"] == "None" and r["dwell_bars"] == 2 and r["hysteresis_delta"] == 0.05]
    for r in defaults:
        lines.append(f"| {r['ticker']} | {r['regime']} | {fmt_val(r['sharpe'])} | {fmt_val(r['max_drawdown'])} | {r['n_trades']} | {fmt_val(r['win_rate'])} | {fmt_val(r['profit_factor'])} | {fmt_val(r['total_return'])} | {r['verdict']} |")
    lines.append("")

    lines.append("## Phase 2: Best & Worst Configuration Per Ticker")
    lines.append("")
    lines.append("| Ticker | Best Sharpe | Best Config | Worst Sharpe | Worst Config |")
    lines.append("|--------|-------------|-------------|--------------|--------------|")
    for t in sorted(per_ticker_best.keys()):
        b = per_ticker_best[t]["best"]
        w = per_ticker_best[t]["worst"]
        bcfg = f"n_s={b['n_states']} pca={b['pca_variance']} st={b['saliency_threshold']} dw={b['dwell_bars']} hy={b['hysteresis_delta']}" if b else "N/A"
        wcfg = f"n_s={w['n_states']} pca={w['pca_variance']} st={w['saliency_threshold']} dw={w['dwell_bars']} hy={w['hysteresis_delta']}" if w else "N/A"
        lines.append(f"| {t} | {fmt_val(b['sharpe']) if b else 'N/A'} | {bcfg} | {fmt_val(w['sharpe']) if w else 'N/A'} | {wcfg} |")
    lines.append("")

    # Top 10 overall
    sorted_all = sorted([r for r in all_results if r.get("sharpe") is not None], key=lambda r: r["sharpe"], reverse=True)
    lines.append("## Top 10 Configurations Overall (by Sharpe)")
    lines.append("")
    lines.append("| # | Ticker | Sharpe | n_states | pca | saliency | dwell | hyst | verdict |")
    lines.append("|---|--------|--------|----------|-----|----------|-------|------|---------|")
    for i, r in enumerate(sorted_all[:10]):
        lines.append(f"| {i+1} | {r['ticker']} | {fmt_val(r['sharpe'])} | {r['n_states']} | {r['pca_variance']} | {fmt_val(r['saliency_threshold'])} | {r['dwell_bars']} | {fmt_val(r['hysteresis_delta'])} | {r['verdict']} |")
    lines.append("")

    # Bottom 10
    lines.append("## Bottom 10 Configurations (by Sharpe)")
    lines.append("")
    sorted_bottom = sorted([r for r in all_results if r.get("sharpe") is not None], key=lambda r: r["sharpe"])
    lines.append("| # | Ticker | Sharpe | n_states | pca | saliency | dwell | hyst |")
    lines.append("|---|--------|--------|----------|-----|----------|-------|------|")
    for i, r in enumerate(sorted_bottom[:10]):
        lines.append(f"| {i+1} | {r['ticker']} | {fmt_val(r['sharpe'])} | {r['n_states']} | {r['pca_variance']} | {fmt_val(r['saliency_threshold'])} | {r['dwell_bars']} | {fmt_val(r['hysteresis_delta'])} |")
    lines.append("")

    # Per-parameter impact analysis
    lines.append("## Parameter Impact Analysis")
    lines.append("")
    lines.append("For each parameter, we compare aggregate Sharpe across all tickers.")
    lines.append("")

    for param_name, grid_values in PARAM_GRID.items():
        lines.append(f"### {param_name}")
        lines.append("")
        lines.append(f"| Value | Mean Sharpe | Std Sharpe | N Runs |")
        lines.append("|-------|------------|------------|--------|")
        # Group by single-param variation
        for val in grid_values:
            if param_name == "n_states":
                key = f"n_states={val}"
                matching = [r for r in all_results if str(r.get("n_states")) == str(val)]
            elif param_name == "pca_variance":
                matching = [r for r in all_results if str(r.get("pca_variance")) == str(val)]
            elif param_name == "saliency_threshold":
                matching = [r for r in all_results if r.get("saliency_threshold") == val]
            elif param_name == "dwell_bars":
                matching = [r for r in all_results if r.get("dwell_bars") == val]
            elif param_name == "hysteresis_delta":
                matching = [r for r in all_results if r.get("hysteresis_delta") == val]

            if not matching:
                continue

            sharpes = [r["sharpe"] for r in matching if r["sharpe"] is not None]
            if not sharpes:
                continue
            mean_s = sum(sharpes) / len(sharpes)
            if len(sharpes) > 1:
                std_s = (sum((s - mean_s)**2 for s in sharpes) / (len(sharpes) - 1))**0.5
            else:
                std_s = 0.0
            lines.append(f"| {val} | {mean_s:.4f} | {std_s:.4f} | {len(sharpes)} |")
        lines.append("")

    # Judgment
    lines.append("## Final Judgment")
    lines.append("")

    # Compute which param values are best on average
    judgment_lines = []
    judgment_lines.append("**Parameter Recommendations:**")
    judgment_lines.append("")

    for param_name, grid_values in PARAM_GRID.items():
        best_val = None
        best_mean = float("-inf")
        for val in grid_values:
            if param_name == "n_states":
                matching = [r for r in all_results if str(r.get("n_states")) == str(val) and r.get("sharpe") is not None]
            elif param_name == "pca_variance":
                matching = [r for r in all_results if str(r.get("pca_variance")) == str(val) and r.get("sharpe") is not None]
            elif param_name == "saliency_threshold":
                matching = [r for r in all_results if r.get("saliency_threshold") == val and r.get("sharpe") is not None]
            elif param_name == "dwell_bars":
                matching = [r for r in all_results if r.get("dwell_bars") == val and r.get("sharpe") is not None]
            elif param_name == "hysteresis_delta":
                matching = [r for r in all_results if r.get("hysteresis_delta") == val and r.get("sharpe") is not None]
            if matching:
                mean_s = sum(r["sharpe"] for r in matching) / len(matching)
                if mean_s > best_mean:
                    best_mean = mean_s
                    best_val = val
        judgment_lines.append(f"- **{param_name}**: best value = **{best_val}** (mean Sharpe = {best_mean:.4f})")
    judgment_lines.append("")

    judgment_lines.append("**Default vs Best Comparison:**")
    judgment_lines.append("")
    default_sharpes = [r["sharpe"] for r in defaults if r["sharpe"] is not None]
    if default_sharpes:
        default_mean = sum(default_sharpes) / len(default_sharpes)
        judgment_lines.append(f"- Default configuration mean Sharpe: **{default_mean:.4f}**")
    best_overall = sorted_all[0] if sorted_all else None
    if best_overall:
        judgment_lines.append(f"- Best overall Sharpe: **{fmt_val(best_overall['sharpe'])}** ({best_overall['ticker']}, n_states={best_overall['n_states']}, pca={best_overall['pca_variance']}, st={best_overall['saliency_threshold']}, dwell={best_overall['dwell_bars']}, hyst={best_overall['hysteresis_delta']})")
    judgment_lines.append("")

    # Final summary
    judgment_lines.append("**Conclusion:**")
    judgment_lines.append("")
    judgment_lines.append("After running FSHMMEngine across all tickers with default and swept parameters:")
    judgment_lines.append("")

    lines.extend(judgment_lines)

    report = "\n".join(lines)

    path = OUTPUT_DIR / "phase3_report.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nPhase 3 report → {path}", file=sys.stderr)
    return report


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"FSHMMEngine Sweep", file=sys.stderr)
    print(f"Tickers: {TICKERS}", file=sys.stderr)
    print(f"Output: {OUTPUT_DIR}/", file=sys.stderr)
    print(file=sys.stderr)

    p1 = phase1()
    p2 = phase2(p1)
    report = phase3(p2)

    print("\n" + "=" * 72, file=sys.stderr)
    print("DONE. Full report:", file=sys.stderr)
    print(report)
