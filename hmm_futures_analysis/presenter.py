"""Presenter layer: format PipelineResult and eval output as strings.

This module owns the "how it looks" seam.  It does not write to streams,
parse arguments, construct configs, or touch engine internals.

Four public functions:

- ``format_pipeline()``  → terminal-formatted string for a single run
- ``serialize_pipeline()`` → JSON-compatible dict for a single run
- ``format_eval()``      → markdown table, statistical table, or JSON
  string for eval results
- ``limit_transitions()`` → pure filter/reverse for regime transitions
"""

from __future__ import annotations

import json
from typing import Any, Literal

from .regime.pipeline import PipelineResult

_STATE_NAMES = ("bear", "sideways", "bull")


# ── Public surface ──────────────────────────────────────────────────


def format_pipeline(
    result: PipelineResult,
    *,
    transitions_limit: int | None = None,
) -> str:
    """Format a single PipelineResult as a human-readable terminal string.

    Parameters
    ----------
    result : PipelineResult
        Output from ``pipeline.run()``.
    transitions_limit : int | None
        Controls the regime-transitions section:
        - ``None``  → omit transitions entirely.
        - ``0``     → show all transitions, newest first.
        - ``N > 0`` → show at most N, newest first.

    Returns
    -------
    str
        Formatted output.  Ends with a newline.  The caller decides
        where to write it (stdout, stderr, file, HTTP response, …).

    Raises
    ------
    TypeError
        If *result* is not a PipelineResult.

    Invariants
    ----------
    - Stateless. Same inputs → same string every time.
    - Does not read or write any file, stream, or env var.
    """
    lines: list[str] = []
    width = 54
    sep = "─" * width

    def header(title: str) -> None:
        lines.append(f"\n{sep}")
        lines.append(f"  {title}")
        lines.append(sep)

    sr = result
    header("REGIME DETECTION")
    lines.append(f"  Source      : {sr.source}")
    lines.append(f"  Engine      : {sr.engine}")
    lines.append(f"  Date range  : {sr.dates['start']} → {sr.dates['end']}")

    ei = sr.engine_info
    lines.append(f"  Method      : {ei['method']}")
    lines.append(f"  Features    : {ei['features']}")
    if "caveat" in ei:
        lines.append(f"  Caveat      : {ei['caveat']}")

    header("CURRENT REGIME")
    cr = sr.current_regime
    lines.append(f"  Regime      : {cr['name'].upper()} (index {cr['index']})")
    lines.append(f"  Signal      : {sr.signal:+.4f}")

    header("REGIME DISTRIBUTION")
    rc = sr.regime_counts
    for name in _STATE_NAMES:
        lines.append(f"  {name.capitalize():<12s}: {rc.get(name, 0):>6d}")

    header("NEXT-STATE PROBABILITIES")
    for name in _STATE_NAMES:
        prob = sr.next_state_probabilities[name]
        bar = "█" * int(prob * 30)
        lines.append(f"  {name.capitalize():<12s}: {prob:.3f}  {bar}")

    header("PERSISTENCE")
    for name in _STATE_NAMES:
        p = sr.persistence_diagonal[name]
        lines.append(f"  {name.capitalize():<12s}: {p:.3f}")

    header("TRANSITION MATRIX")
    names = [s.capitalize() for s in _STATE_NAMES]
    lines.append(f"  {'':>10s}  {names[0]:>8s}  {names[1]:>8s}  {names[2]:>8s}")
    for i, row in enumerate(sr.transition_matrix):
        cells = "  ".join(f"{v:8.3f}" for v in row)
        lines.append(f"  {names[i]:>10s}  {cells}")

    header("STATIONARY DISTRIBUTION")
    for name in _STATE_NAMES:
        prob = sr.stationary_distribution[name]
        lines.append(f"  {name.capitalize():<12s}: {prob:.3f}")

    header("FORECAST (1-step)")
    f = sr.forecast["1_step"]
    for name in _STATE_NAMES:
        lines.append(f"  {name.capitalize():<12s}: {f[name]:.3f}")

    header("WALK-FORWARD BACKTEST")
    wf = sr.walk_forward
    sharpe_str = f"{wf['sharpe']:.2f}" if wf["sharpe"] is not None else "N/A"
    dd_str = f"{wf['max_drawdown']:.2%}" if wf["max_drawdown"] is not None else "N/A"
    wr_str = f"{wf['win_rate']:.1%}" if wf["win_rate"] is not None else "N/A"
    pf_str = f"{wf['profit_factor']:.2f}" if wf["profit_factor"] is not None else "N/A"
    tr_str = f"{wf['total_return']:.2%}" if wf["total_return"] is not None else "N/A"

    lines.append(f"  Sharpe ratio   : {sharpe_str}")
    lines.append(f"  Max drawdown   : {dd_str}")
    lines.append(f"  Total return   : {tr_str}")
    lines.append(f"  Trades         : {wf['n_trades']}")
    lines.append(f"  Win rate       : {wr_str}")
    lines.append(f"  Profit factor  : {pf_str}")

    # Lookahead bias warning for --reverse-classify (Issue #102)
    if sr.engine_info.get("lookahead_bias_warning"):
        lines.append(
            "  ⚠  LOOKAHEAD BIAS: walk-forward backtest contains lookahead bias."
        )
        lines.append("     Regime at bar t is partially informed by data from t+1…n.")
        lines.append(
            "     Use reverse-classify for display only, not backtest decisions."
        )

    if sr.duration_forecast is not None:
        df = sr.duration_forecast
        header("DURATION FORECAST")
        lines.append(f"  Current regime     : {df['current_regime'].upper()}")
        lines.append(f"  Days in regime     : {df['days_in_regime']}")
        if df["expected_remaining_days"] is not None:
            lines.append(
                f"  Expected remaining  : {df['expected_remaining_days']:.1f} days"
            )
            lines.append(f"  Hazard rate         : {df['hazard_rate']:.4f}")
            lines.append(f"  Median survival     : {df['survival_50pct']:.1f} days")
            lines.append(f"  Weibull shape       : {df['weibull_shape']:.4f}")
            lines.append(f"  Weibull scale       : {df['weibull_scale']:.2f}")
        else:
            lines.append("  (insufficient historical spells for fitting)")

    # --- Regime transitions (issue #63) ---
    if transitions_limit is not None and sr.regime_transitions:
        header("REGIME TRANSITIONS")
        all_transitions = sr.regime_transitions
        shown = limit_transitions(all_transitions, transitions_limit)
        for ev in shown:
            lines.append(
                f"  {ev['date']}  {ev['from_regime'].upper()} → {ev['to_regime'].upper()}"
            )
        if transitions_limit > 0 and len(all_transitions) > transitions_limit:
            lines.append(f"  ... ({len(all_transitions) - transitions_limit} more)")

    header("DISCLAIMER")
    lines.append(f"  {sr.disclaimer}")
    lines.append(sep)

    return "\n".join(lines) + "\n"


def serialize_pipeline(
    result: PipelineResult,
    *,
    transitions_limit: int | None = None,
) -> dict[str, Any]:
    """Convert PipelineResult to a JSON-compatible dict.

    Thin wrapper around ``result._asdict()`` with optional transition
    limiting applied to the ``"regime_transitions"`` key.

    Parameters
    ----------
    result : PipelineResult
        Output from ``pipeline.run()``.
    transitions_limit : int | None
        - ``None``  → include transitions as-is (no filtering).
        - ``0``     → all transitions, reversed (newest first).
        - ``N > 0`` → N most recent, newest first.

    Returns
    -------
    dict[str, Any]
        JSON-compatible dict.  Safe for ``json.dump(..., allow_nan=False)``.
        Returns a new dict every call (no shared mutable state).

    Invariants
    ----------
    - Stateless.
    - Does not import argparse, sys, csv, or any engine module.
    """
    d = result._asdict()
    if d.get("regime_transitions"):
        limit = transitions_limit if transitions_limit is not None else 0
        d["regime_transitions"] = limit_transitions(d["regime_transitions"], limit)
    return d


def format_eval(
    results: list[dict[str, Any]],
    *,
    fmt: Literal["table", "json", "statistical_table"] = "table",
) -> str:
    """Format eval-mode results as a string.

    Parameters
    ----------
    results : list[dict]
        List of summary dicts from ``run_eval_csv()`` or
        ``run_eval_tickers()``.
    fmt : "table", "json" or "statistical_table", default "table"
        - ``"table"`` → markdown-style table.
        - ``"json"`` → indented JSON array.
        - ``"statistical_table"`` → cross-engine agreement summary and
          buy-and-hold delta grid (issue #114, ADR-0028).

    Returns
    -------
    str
        Formatted output.  The caller decides where to write it.

    Raises
    ------
    ValueError
        If *fmt* is not one of the recognised literals.

    Invariants
    ----------
    - Stateless.
    """
    if fmt == "json":
        return json.dumps(results, indent=2, allow_nan=False) + "\n"
    if fmt == "table":
        return _format_eval_table(results)
    if fmt == "statistical_table":
        return _format_statistical_table(results)
    raise ValueError(f"Unknown format: {fmt!r}")


# ── Public utility ──────────────────────────────────────────────────


def limit_transitions(
    transitions: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Filter and reorder transitions for display/serialization.

    Parameters
    ----------
    transitions : list[dict]
        Chronological transition list (oldest first).
    limit : int | None
        ``None``  → passthrough (returns object unchanged).
        ``0``     → all transitions, reversed (newest first).
        ``N > 0`` → N most recent, newest first.

    Returns
    -------
    The original list object when *limit* is ``None``; a new list
    otherwise.
    """
    if limit is None:
        return transitions
    if not transitions:
        return transitions
    reversed_list = list(reversed(transitions))
    if limit == 0:
        return reversed_list
    return reversed_list[:limit]


# ── Private helpers ─────────────────────────────────────────────────


def _format_eval_table(results: list[dict[str, Any]]) -> str:
    """Format results as a markdown table (moved from eval.format_table)."""
    if not results:
        return "No results."

    headers = [
        "ticker",
        "engine",
        "regime",
        "signal",
        "sharpe",
        "max_dd",
        "trades",
        "win_rate",
        "pf",
        "total_ret",
        "wall_s",
    ]

    def fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            if abs(val) < 100:
                return f"{val:.3f}"
            return f"{val:.1f}"
        return str(val)

    rows: list[list[str]] = []
    for r in results:
        rows.append(
            [
                r["ticker"],
                r["engine"],
                r["regime"],
                fmt(r["signal"]),
                fmt(r["sharpe"]),
                fmt(r["max_drawdown"]),
                str(r["n_trades"]),
                fmt(r["win_rate"]),
                fmt(r["profit_factor"]),
                fmt(r["total_return"]),
                fmt(r["wall_seconds"]),
            ]
        )

    # Column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    def separator() -> str:
        return "  ".join("-" * w for w in widths)

    lines = [
        fmt_row(headers),
        separator(),
    ]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)



def _format_statistical_table(results: list[dict[str, Any]]) -> str:
    """Format results as the statistical report (issue #114, Slice 3).

    Two grids: the cross-engine agreement summary (one row per ticker,
    engine vote columns, plus a report-only majority column) and the
    buy-and-hold delta grid (engine Sharpe minus B&H Sharpe, one row per
    engine).  Pure: no IO.  (ADR-0028.)
    """
    if not results:
        return "No results."

    ticker_order: list[str] = []
    engine_order: list[str] = []
    votes: dict[tuple[str, str], str] = {}
    deltas: dict[tuple[str, str], Any] = {}
    kappa_present = "fleiss_kappa" in results[0]
    kappa = results[0].get("fleiss_kappa")
    for r in results:
        ticker, engine = r["ticker"], r["engine"]
        if ticker not in ticker_order:
            ticker_order.append(ticker)
        if engine not in engine_order:
            engine_order.append(engine)
        votes[(ticker, engine)] = r["regime"]
        deltas[(ticker, engine)] = r.get("bnh_delta")

    def majority(ticker: str) -> str:
        """Modal vote across engines; 'tie' when the top count is shared.

        Descriptive agreement only — never an ensemble position signal
        (.out-of-scope/ensemble-engine.md).
        """
        tally: dict[str, int] = {}
        for engine in engine_order:
            vote = votes.get((ticker, engine))
            if vote is not None:
                tally[vote] = tally.get(vote, 0) + 1
        if not tally:
            return "N/A"
        top = max(tally.values())
        leaders = sorted(v for v, n in tally.items() if n == top)
        return leaders[0] if len(leaders) == 1 else "tie"

    def render(headers: list[str], rows: list[list[str]]) -> list[str]:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        lines = [
            "  ".join(c.ljust(widths[i]) for i, c in enumerate(headers)),
            "  ".join("-" * w for w in widths),
        ]
        for row in rows:
            lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        return lines

    lines = ["Cross-engine agreement (current_regime):"]
    rows = [
        [
            ticker,
            *[
                str(votes[(ticker, engine)])
                if (ticker, engine) in votes
                else "N/A"
                for engine in engine_order
            ],
            majority(ticker),
        ]
        for ticker in ticker_order
    ]
    lines.extend(render(["ticker", *engine_order, "majority"], rows))

    if kappa_present:
        kappa_str = "n/a" if kappa is None else f"{kappa:.6f}"
        lines.append("")
        lines.append(f"Fleiss' kappa = {kappa_str}")

    lines.append("")
    lines.append("Buy-and-hold delta Sharpe (engine Sharpe - B&H Sharpe):")
    rows = [
        [
            engine,
            *[
                "N/A"
                if deltas.get((ticker, engine)) is None
                else f"{deltas[(ticker, engine)]:.3f}"
                for ticker in ticker_order
            ],
        ]
        for engine in engine_order
    ]
    lines.extend(render(["engine", *ticker_order], rows))

    return "\n".join(lines)
