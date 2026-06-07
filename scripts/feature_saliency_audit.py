"""Feature Saliency Audit — ranks generic features by FSHMM saliency.

Generates synthetic OHLCV data with known regime structure, runs the full
feature engineering pipeline, fits FSHMM, and produces a ranked report
of feature importance based on average saliency weights (rho_k).

Usage:
    python scripts/feature_saliency_audit.py

Output:
    docs/research/feature-saliency-audit.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _generate_regime_ohlcv(
    n_bars: int = 2000,
    n_regimes: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with distinct market regimes.

    Each regime has different drift and volatility characteristics:
    - Regime 0 (Bear): negative drift, high volatility
    - Regime 1 (Sideways): near-zero drift, moderate volatility
    - Regime 2 (Bull): positive drift, low volatility
    """
    rng = np.random.default_rng(seed)
    bars_per_regime = n_bars // n_regimes
    remainder = n_bars - bars_per_regime * n_regimes

    regime_params = [
        {"drift": -0.002, "vol": 0.025, "label": "bear"},
        {"drift": 0.000, "vol": 0.015, "label": "sideways"},
        {"drift": 0.003, "vol": 0.010, "label": "bull"},
    ]

    close_prices = [100.0]
    regimes = []

    for r_idx, params in enumerate(regime_params):
        size = bars_per_regime + (1 if r_idx < remainder else 0)
        daily_returns = rng.normal(params["drift"], params["vol"], size)
        for ret in daily_returns:
            close_prices.append(close_prices[-1] * (1 + ret))
        regimes.extend([r_idx] * size)

    close = pd.Series(close_prices[1:])

    # Build plausible OHLCV from close prices
    n = len(close)
    noise = rng.uniform(0.002, 0.008, n)
    open_price = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_price, close.shift(-1).fillna(close.iloc[-1])) * (1 + noise)
    low = np.minimum(open_price, close.shift(-1).fillna(close.iloc[-1])) * (1 - noise)
    high = np.maximum(high, open_price)
    low = np.minimum(low, open_price)

    dates = pd.bdate_range("2020-01-01", periods=n)
    ohlcv = pd.DataFrame(
        {
            "open": open_price.values,
            "high": high,
            "low": low,
            "close": close.values,
            "volume": rng.integers(500_000, 2_000_000, n),
        },
        index=dates,
    )
    ohlcv["true_regime"] = regimes[:n]

    return ohlcv


def _run_audit(
    ohlcv: pd.DataFrame, n_runs: int = 5
) -> dict[str, list[float]]:
    """Run FSHMM multiple times and collect per-feature saliency weights.

    Returns a dict mapping feature_name → list of saliency values across runs.
    """
    from hmm_futures_analysis.data_processing.feature_engineering import add_features
    from hmm_futures_analysis.regime.engines.fshmm import FSHMMEngine

    # Engineer generic features
    features_df = add_features(ohlcv, min_periods=10)
    # Drop OHLCV columns — keep only engineered features
    ohlcv_set = {"open", "high", "low", "close", "volume", "true_regime"}
    feature_cols = [c for c in features_df.columns if c not in ohlcv_set]
    # Filter to numeric columns only
    numeric_df = features_df[feature_cols].select_dtypes(include=[np.number])
    # Drop columns with all NaN
    numeric_df = numeric_df.dropna(axis=1, how="all")
    # Forward-fill and drop remaining NaN
    numeric_df = numeric_df.bfill().dropna()

    all_saliencies: dict[str, list[float]] = {}

    for run_idx in range(n_runs):
        engine = FSHMMEngine(
            n_states=3,
            saliency_threshold=0.0,  # collect all, regardless of relevance
            random_state=42 + run_idx,
            max_iter=40,
        )
        try:
            result = engine.classify(numeric_df)
            if result.feature_saliency is not None:
                rho = result.feature_saliency
                for feat_name, sal in zip(numeric_df.columns, rho):
                    all_saliencies.setdefault(feat_name, []).append(float(sal))
                print(
                    f"  Run {run_idx + 1}/{n_runs}: {len(rho)} features, "
                    f"mean ρ={np.mean(rho):.3f}, "
                    f"# ρ≥0.5 = {np.sum(rho >= 0.5)}"
                )
            else:
                print(f"  Run {run_idx + 1}/{n_runs}: saliency is None (skipping)")
        except Exception as exc:
            print(f"  Run {run_idx + 1}/{n_runs}: FAILED — {exc}")
            continue

    return all_saliencies


def _categorize_feature(name: str) -> str:
    """Map a feature name to a category for grouping."""
    lower = name.lower()
    if any(x in lower for x in ("log_ret", "simple_ret")):
        return "Returns"
    if any(x in lower for x in ("ma_", "sma_", "ema_", "tma_", "wma_", "hma_")):
        return "Moving Averages"
    if any(x in lower for x in ("atr", "volatility", "bb_", "kc_", "keltner_",
                                 "donchian_", "chaikin_vol", "hv_")):
        return "Volatility"
    if any(x in lower for x in ("rsi", "roc", "macd", "stoch", "williams",
                                 "cci", "mfi", "mtm_", "proc")):
        return "Momentum"
    if any(x in lower for x in ("volume_", "obv", "vwap", "adl", "vpt", "eom_")):
        return "Volume"
    if any(x in lower for x in ("adx", "aroon", "di_", "dpo")):
        return "Trend"
    if any(x in lower for x in ("price_position", "hl_ratio")):
        return "Price Patterns"
    if any(x in lower for x in ("day_of", "month", "quarter", "hour",
                                 "minute", "session", "is_", "_sin", "_cos")):
        return "Time-based"
    return "Other"


def _generate_report(
    all_saliencies: dict[str, list[float]],
    output_path: Path,
    n_runs: int,
) -> None:
    """Generate a Markdown report from collected saliency weights."""
    # Compute average saliency per feature
    rankings = []
    for feat_name, values in all_saliencies.items():
        vals = np.array(values, dtype=np.float64)
        rankings.append({
            "feature": feat_name,
            "mean_rho": float(np.mean(vals)),
            "std_rho": float(np.std(vals)),
            "min_rho": float(np.min(vals)),
            "max_rho": float(np.max(vals)),
            "category": _categorize_feature(feat_name),
            "n_runs": len(values),
        })

    rankings.sort(key=lambda r: r["mean_rho"], reverse=True)

    # Category summary
    from collections import defaultdict
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in rankings:
        cat_scores[r["category"]].append(r["mean_rho"])
    cat_summary = [
        (cat, float(np.mean(scores)), float(np.std(scores)), len(scores))
        for cat, scores in cat_scores.items()
    ]
    cat_summary.sort(key=lambda x: x[1], reverse=True)

    # Build markdown
    lines = []
    lines.append("# Feature Saliency Audit Report")
    lines.append("")
    lines.append(f"*Generated by `scripts/feature_saliency_audit.py`*")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- **Data**: 2000 bars of synthetic OHLCV with 3 known regimes "
        "(bear/sideways/bull), each with distinct drift and volatility."
    )
    lines.append(
        f"- **Engine**: FSHMM (n_states=3, saliency_threshold=0.0, max_iter=40)"
    )
    lines.append(
        f"- **Runs**: {n_runs} independent runs with different random seeds "
        f"(42–{42 + n_runs - 1})"
    )
    lines.append(
        "- **Features**: Full generic feature set via `add_features()` "
        "(see `feature_engineering.py`)"
    )
    lines.append(
        "- **Metric**: Saliency weight ρₖ ∈ [0, 1] — 1 = highly relevant, "
        "0 = irrelevant background"
    )
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Mean ρ | Std ρ | # Features |")
    lines.append("|----------|--------|-------|------------|")
    for cat, mean_r, std_r, count in cat_summary:
        lines.append(f"| {cat} | {mean_r:.4f} | {std_r:.4f} | {count} |")
    lines.append("")
    lines.append("## Feature Rankings")
    lines.append("")
    lines.append(
        f"Ranked by mean ρ across {n_runs} runs (descending):"
    )
    lines.append("")
    lines.append("| Rank | Feature | Category | Mean ρ | Std ρ | Min ρ | Max ρ | Runs |")
    lines.append("|------|---------|----------|--------|-------|-------|-------|------|")
    for idx, r in enumerate(rankings, 1):
        lines.append(
            f"| {idx} | `{r['feature']}` | {r['category']} | "
            f"{r['mean_rho']:.4f} | {r['std_rho']:.4f} | "
            f"{r['min_rho']:.4f} | {r['max_rho']:.4f} | {r['n_runs']} |"
        )
    lines.append("")

    # Top/bottom observations
    top_10 = rankings[:10]
    bottom_10 = rankings[-10:]

    lines.append("## Observations")
    lines.append("")

    # Category observations
    lines.append("### By Category")
    lines.append("")
    for cat, mean_r, std_r, count in cat_summary:
        lines.append(f"- **{cat}** ({count} features): mean ρ = {mean_r:.4f}")

    lines.append("")
    lines.append("### Top 10 Features")
    lines.append("")
    for r in top_10:
        lines.append(f"- `{r['feature']}`: ρ = {r['mean_rho']:.4f} ± {r['std_rho']:.4f} ({r['category']})")

    lines.append("")
    lines.append("### Bottom 10 Features")
    lines.append("")
    for r in bottom_10:
        lines.append(f"- `{r['feature']}`: ρ = {r['mean_rho']:.4f} ± {r['std_rho']:.4f} ({r['category']})")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "- Features with mean ρ < 0.3 across multiple runs are candidates "
        "for trimming."
    )
    lines.append(
        "- Time-based features (day_of_week, hour, minute, session, etc.) "
        "should be evaluated per data frequency — on daily data they are "
        "constant or near-constant and add no signal."
    )
    lines.append(
        "- Before removing any features, re-run this audit on real market data "
        "(e.g. BTC/USD, ES futures) as synthetic data may not capture all "
        "real-world relationships."
    )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Synthetic data**: The regimes are simple drift/vol changes. "
        "Real markets exhibit more complex dynamics."
    )
    lines.append(
        "- **No PCA**: This audit was run without PCA whitening, so features "
        "are evaluated in their original space."
    )
    lines.append(
        "- **Single asset class**: Results may differ for equities vs crypto "
        "vs FX."
    )
    lines.append(
        "- **Feature collinearity**: Highly correlated features may split "
        "saliency weight, depressing individual rankings."
    )
    lines.append(
        "- **This is an AUDIT report**. No features have been removed. "
        "Use this evidence to inform future trimming decisions."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nReport written to {output_path}")


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "docs" / "research" / "feature-saliency-audit.md"
    n_runs = 5

    print("=" * 60)
    print("Feature Saliency Audit")
    print("=" * 60)
    print()

    print("Step 1: Generating synthetic OHLCV data...")
    ohlcv = _generate_regime_ohlcv(n_bars=2000, n_regimes=3, seed=42)
    print(f"  Generated {len(ohlcv)} bars, {len(ohlcv.columns)} columns")
    print(f"  Regime distribution: {ohlcv['true_regime'].value_counts().to_dict()}")

    print(f"\nStep 2: Running feature engineering...")
    from hmm_futures_analysis.data_processing.feature_engineering import add_features
    features_df = add_features(ohlcv, min_periods=10)
    ohlcv_set = {"open", "high", "low", "close", "volume", "true_regime"}
    feature_cols = [c for c in features_df.columns if c not in ohlcv_set]
    numeric_df = features_df[feature_cols].select_dtypes(include=[np.number])
    print(f"  Feature columns: {len(numeric_df.columns)}")
    print(f"  Feature columns (list): {list(numeric_df.columns)}")

    print(f"\nStep 3: Running FSHMM × {n_runs}...")
    all_saliencies = _run_audit(ohlcv, n_runs=n_runs)

    if not all_saliencies:
        print("ERROR: No saliency data collected. Aborting.")
        sys.exit(1)

    print(f"\nStep 4: Generating report...")
    _generate_report(all_saliencies, output_path, n_runs=n_runs)

    # Print summary to stdout
    rankings = sorted(
        [
            (name, float(np.mean(vals)))
            for name, vals in all_saliencies.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\nTop 5 features by mean ρ:")
    for name, score in rankings[:5]:
        print(f"  {score:.4f}  {name}")
    print(f"\nBottom 5 features by mean ρ:")
    for name, score in rankings[-5:]:
        print(f"  {score:.4f}  {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
