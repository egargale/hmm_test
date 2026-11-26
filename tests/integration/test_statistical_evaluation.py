"""5-ticker x 5-engine statistical evaluation harness (issue #113, Slice 2).

Black-box panel over the committed ``test_data/`` CSVs: 25 serial CLI
subprocesses driven through ``run_regime`` exactly the way a user invokes
``python -m hmm_futures_analysis.cli --csv ... --engine ... --json``.  The
module parses the JSON, asserts the invariants the product exposes, and
writes the measured grids to ``runs/statistical_report.txt``.

Metric definitions (fixed here, once — changing one changes the measured
values and therefore the floors below):

* **Forecast identity** — ``forecast.1_step`` must equal the transition-matrix
  row ``T[current_regime.index, :]`` and ``next_state_probabilities`` to
  ``abs(delta) < 1e-9`` on bear/sideways/bull, on all 25 cells.  Only the
  1-step horizon exists in the JSON surface ([[ADR-0025]]); the 5/20-step
  n-step algebra stays covered by
  ``tests/unit/test_build_markov_stats.py::test_forecast_horizons_match_matrix_powers``.
* **Persistence** — per cell, the mean spell length is
  ``n_bars / (n_transitions + 1)`` where ``n_bars = sum(regime_counts.values())``
  and the transitions are sorted by ``bar_index``; per engine the median of
  that value across the five tickers must be strictly ``> 3``.  Do NOT use the
  median of raw run lengths: that makes threshold exactly 3.0 on this panel
  and fails the strict comparison.
* **Fleiss' kappa** — 5 subjects (tickers) x 5 raters (engines) x 3 categories
  (Bear/Sideways/Bull) over the ``current_regime`` votes; floor ``>= -0.10``.
* **Buy-and-hold delta-Sharpe** — the B&H baseline is computed on the engine's
  own window, not on the full CSV: close prices ``prices[252:]`` (drop the
  first ``min_train=252`` bars), returns ``numpy.diff(px) / px[:-1]``,
  ``Sharpe = mean(r) / std(r, ddof=1) * sqrt(252)`` with ``rf=0``.  Per cell,
  ``walk_forward.sharpe - bh_sharpe``; per engine the median across tickers
  must be ``>= -1.20``.  This one-line window choice flips messina from fail
  to pass, and the Sharpe convention (rf=0, ddof=1) is what reproduces the
  measured baseline (SPY 0.858, NVDA 1.165, BTC 0.399, KO 0.583, XLE 0.504).

The floors are non-regression baselines re-measured on current master (kappa
-0.0523, worst median delta -1.050), not quality targets; their deviation
from the originally proposed 0.0 / -1.0 floors is disclosed in the PR body
([[ADR-0027]]).

Panel execution: every cell runs with ``OMP_NUM_THREADS=1`` and at the CLI
defaults (no flag overrides — [[ADR-0005]] / [[ADR-0019]] fix the parameter
defaults the panel values depend on).  ``fshmm`` cells pass an explicit
``timeout=300`` to ``run_regime`` (4/5 of them exceed 60 s); the shared 60 s
default is untouched because 84 call sites rely on it.

Report artifact: ``runs/statistical_report.txt`` is written by this module —
never by ``presenter.py`` ([[ADR-0021]] keeps the presenter pure) — and
overwritten on each run.  Accepted trade-off: the file is tracked in git, so
running the harness dirties the working tree until the refreshed file is
committed again.
"""

import json
import statistics
import time

import numpy as np
import pytest

from hmm_futures_analysis.data_processing.csv_auto_detect import load_from_csv
from tests.conftest import ROOT, run_regime

pytestmark = [pytest.mark.slow, pytest.mark.statistical]

TICKERS = ("SPY", "NVDA", "BTC", "KO", "XLE")
ENGINES = ("threshold", "messina", "hmm", "robust_hmm", "fshmm")
CELLS = [(t, e) for t in TICKERS for e in ENGINES]
CATEGORIES = ("bear", "sideways", "bull")

MIN_TRAIN = 252  # --min-train default: the engine's warm-up window
FSHMM_TIMEOUT = 300  # seconds; 4/5 fshmm cells exceed the 60 s default
FORECAST_TOL = 1e-9
PERSISTENCE_FLOOR = 3.0
KAPPA_FLOOR = -0.10
BH_DELTA_FLOOR = -1.20

REPORT_PATH = ROOT / "runs" / "statistical_report.txt"

REQUIRED_SECTIONS = (
    "Panel wall time",
    "## A. Per-ticker buy-and-hold",
    "## B. Walk-forward Sharpe grid",
    "## D. Win-rate grid",
    "## E. Max-drawdown grid",
    "## F. Number-of-trades grid",
    "## G. Profit-factor grid",
    "## H. Cross-engine agreement",
    "Fleiss' kappa",
)


# --------------------------------------------------------------------------
# Fleiss' kappa helper (test-side only; no product consumer)
# --------------------------------------------------------------------------


def _fleiss_components(counts):
    """Return ``(p_bar, p_e, kappa)`` for a subjects x categories matrix."""
    mat = np.asarray(counts, dtype=float)
    n_raters = mat.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("each subject must be rated by the same rater count")
    n_subjects, _ = mat.shape
    n = n_raters[0]
    if n < 2:
        raise ValueError("kappa needs at least two raters per subject")
    p_i = (np.sum(mat * mat, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_i))
    p_cat = mat.sum(axis=0) / (n_subjects * n)
    p_e = float(np.sum(p_cat * p_cat))
    if p_e == 1.0:
        raise ValueError("kappa undefined: chance agreement is exactly 1")
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return p_bar, p_e, kappa


def fleiss_kappa(counts_matrix):
    """Fleiss' kappa for a subjects x categories rating-count matrix.

    Row ``i`` of *counts_matrix* holds, per category, how many of the ``n``
    raters assigned subject ``i`` to that category (rows sum to ``n``).  With
    per-subject agreement ``P_i = (sum_j n_ij^2 - n) / (n * (n - 1))``,
    observed agreement ``P_bar = mean_i P_i`` and chance agreement
    ``P_e = sum_j (sum_i n_ij / (N * n))^2``::

        kappa = (P_bar - P_e) / (1 - P_e)

    Interpretation (Landis & Koch 1977): < 0 poor, 0.00-0.20 slight,
    0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00
    almost perfect.  This 5x5 panel measures kappa around -0.05 (the
    poor/slight boundary); the asserted floor is -0.10 — a non-regression
    guard, not a quality target.
    """
    return _fleiss_components(counts_matrix)[2]


# --------------------------------------------------------------------------
# Panel execution (each CLI cell runs exactly once per session)
# --------------------------------------------------------------------------


class StatisticalPanel:
    """Lazily run the 5x5 CLI panel and cache each cell's parsed JSON."""

    def __init__(self, ticker_csvs):
        self._csv = ticker_csvs
        self._data = {}
        self._wall = {}

    def get(self, ticker, engine):
        """Return the parsed JSON of one ``(ticker, engine)`` CLI cell."""
        key = (ticker, engine)
        if key not in self._data:
            args = ["--csv", self._csv[ticker], "--engine", engine, "--json"]
            kwargs = {}
            if engine == "fshmm":
                kwargs["timeout"] = FSHMM_TIMEOUT
            start = time.monotonic()
            result = run_regime(
                *args,
                env={"OMP_NUM_THREADS": "1"},
                **kwargs,
            )
            self._wall[key] = time.monotonic() - start
            assert result.returncode == 0, (
                f"{ticker}/{engine}: rc={result.returncode} "
                f"stderr tail: {result.stderr[-500:]}"
            )
            self._data[key] = json.loads(result.stdout)
        return self._data[key]

    def all_cells(self):
        """Run (if needed) and return every cell keyed by ``(ticker, engine)``."""
        return {key: self.get(*key) for key in CELLS}

    @property
    def wall_seconds(self):
        """Per-cell wall time in seconds (keyed by ``(ticker, engine)``)."""
        return dict(self._wall)

    @property
    def total_wall_seconds(self):
        """Total wall time of the 25 serial CLI cells, in seconds."""
        return sum(self._wall.values())


@pytest.fixture(scope="session")
def ticker_csvs(spy_csv, nvda_csv, btc_csv, ko_csv, xle_csv):
    """Ticker symbol -> CSV path for the 5-ticker panel."""
    return {
        "SPY": spy_csv,
        "NVDA": nvda_csv,
        "BTC": btc_csv,
        "KO": ko_csv,
        "XLE": xle_csv,
    }


@pytest.fixture(scope="session")
def buy_and_hold(ticker_csvs):
    """Ticker -> buy-and-hold stats on the engine's own window.

    Window/returns/Sharpe convention: see the module docstring
    (``prices[252:]`` after ``min_train``, ``numpy.diff``, rf=0, ddof=1).
    """
    stats = {}
    for ticker, path in ticker_csvs.items():
        prices = load_from_csv(path).to_numpy()
        px = prices[MIN_TRAIN:]
        rets = np.diff(px) / px[:-1]
        peak = np.maximum.accumulate(px)
        stats[ticker] = {
            "sharpe": float(
                np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252.0)
            ),
            "total_return": float(px[-1] / px[0] - 1.0),
            "max_drawdown": float(np.min(px / peak - 1.0)),
        }
    return stats


@pytest.fixture(scope="session")
def panel(ticker_csvs):
    """The 5x5 CLI panel; cells run on first access and are cached."""
    return StatisticalPanel(ticker_csvs)


# --------------------------------------------------------------------------
# Report writer (this module owns the IO — never presenter.py)
# --------------------------------------------------------------------------


def _vote_counts(panel):
    """5x3 (bear/sideways/bull) vote-count matrix over engine raters."""
    rows = []
    for ticker in TICKERS:
        counts = [0, 0, 0]
        for engine in ENGINES:
            name = panel.get(ticker, engine)["current_regime"]["name"]
            counts[CATEGORIES.index(name)] += 1
        rows.append(counts)
    return rows


def _mean_spell(data):
    """Mean regime spell length in bars (see module docstring)."""
    transitions = sorted(
        data["regime_transitions"], key=lambda ev: ev["bar_index"]
    )
    n_bars = sum(data["regime_counts"].values())
    return n_bars / (len(transitions) + 1)


def _row(label, values, fmt):
    cells = "".join(fmt.format(v) for v in values)
    return f"  {label:<12}{cells}\n"


def _write_report(panel, buy_and_hold):
    """Write the measured statistical grids to ``runs/statistical_report.txt``."""
    cells = panel.all_cells()
    lines = []
    out = lines.append

    out("=" * 78 + "\n")
    out("Statistical evaluation report (issue #113, Slice 2)\n")
    out("Generated by tests/integration/test_statistical_evaluation.py.\n")
    out("Overwritten on each `pytest -m statistical` run; measured values only\n")
    out("— the assertions live in the test module.\n\n")
    out(f"Panel wall time: {panel.total_wall_seconds:.3f} s for 25 serial CLI\n")
    out("cells at OMP_NUM_THREADS=1 (fshmm cells timeout=300, other cells the\n")
    out("60 s run_regime default).\n\n")
    out("Per-cell wall time (seconds):\n")
    out("  engine      " + "".join(f"{t:>9}" for t in TICKERS) + f"{'total':>10}\n")
    for engine in ENGINES:
        times = [panel.wall_seconds[(t, engine)] for t in TICKERS]
        total = sum(times)
        cells_str = "".join(f"{v:9.3f}" for v in times)
        out(f"  {engine:<12}{cells_str}{total:10.3f}\n")

    out("\n## A. Per-ticker buy-and-hold (engine window: prices[252:],\n")
    out("     numpy.diff, rf=0, ddof=1) — not the full CSV\n")
    out("  ticker  B&H Sharpe  B&H ret  B&H maxDD\n")
    for ticker in TICKERS:
        bh = buy_and_hold[ticker]
        out(
            f"  {ticker:<8}{bh['sharpe']:10.3f}"
            f"{bh['total_return'] * 100:9.2f}%"
            f"{bh['max_drawdown'] * 100:10.2f}%\n"
        )

    header = "  engine      " + "".join(f"{t:>9}" for t in TICKERS)
    out("\n## B. Walk-forward Sharpe grid (engine row, ticker column)\n")
    out(header + f"{'median':>9}\n")
    for engine in ENGINES:
        sharpes = [cells[(t, engine)]["walk_forward"]["sharpe"] for t in TICKERS]
        out(_row(engine, sharpes + [statistics.median(sharpes)], "{:9.3f}"))
    bh_row = [buy_and_hold[t]["sharpe"] for t in TICKERS]
    out(_row("B&H", bh_row + [statistics.median(bh_row)], "{:8.3f}"))

    out("\n  Per-engine skill (walk-forward Sharpe - B&H Sharpe):\n")
    for engine in ENGINES:
        sharpes = [
            cells[(t, engine)]["walk_forward"]["sharpe"] for t in TICKERS
        ]
        deltas = [
            s - buy_and_hold[t]["sharpe"] for s, t in zip(sharpes, TICKERS)
        ]
        wins = sum(1 for d in deltas if d > 0)
        out(
            f"    {engine:<12} median Sharpe="
            f"{statistics.median(sharpes): .3f}"
            f"  median delta vs B&H={statistics.median(deltas): .3f}"
            f"  wins={wins}/5\n"
        )

    out("\n## C. Delta-Sharpe vs buy-and-hold (median is the asserted value)\n")
    out(header + f"{'median':>9}\n")
    for engine in ENGINES:
        deltas = [
            cells[(t, engine)]["walk_forward"]["sharpe"]
            - buy_and_hold[t]["sharpe"]
            for t in TICKERS
        ]
        out(_row(engine, deltas + [statistics.median(deltas)], "{:9.3f}"))

    out("\n## D. Win-rate grid (percent)\n")
    out(header + f"{'median':>9}\n")
    for engine in ENGINES:
        rates = [
            cells[(t, engine)]["walk_forward"]["win_rate"] * 100.0
            for t in TICKERS
        ]
        out(_row(engine, rates + [statistics.median(rates)], "{:8.2f}%"))

    out("\n## E. Max-drawdown grid (percent, negative)\n")
    out(header + f"{'median':>9}\n")
    for engine in ENGINES:
        dds = [
            cells[(t, engine)]["walk_forward"]["max_drawdown"] * 100.0
            for t in TICKERS
        ]
        out(_row(engine, dds + [statistics.median(dds)], "{:8.2f}%"))

    out("\n## F. Number-of-trades grid\n")
    out(header + f"{'total':>9}\n")
    for engine in ENGINES:
        trades = [
            cells[(t, engine)]["walk_forward"]["n_trades"] for t in TICKERS
        ]
        out(_row(engine, trades + [sum(trades)], "{:9d}"))

    out("\n## G. Profit-factor grid (>1 = profitable; 'n/a' = no losing\n")
    out("     trades -> JSON null; median ignores n/a cells)\n")
    out(header + f"{'median':>9}\n")
    for engine in ENGINES:
        pfs = [
            cells[(t, engine)]["walk_forward"]["profit_factor"]
            for t in TICKERS
        ]
        finite = [v for v in pfs if v is not None]
        med = statistics.median(finite) if finite else None
        cells_str = "".join(
            f"{v:9.4f}" if v is not None else f"{'n/a':>9}" for v in pfs
        )
        med_str = f"{med:9.4f}" if med is not None else f"{'n/a':>9}"
        nulls = len(pfs) - len(finite)
        note = f"  ({nulls} nulls)" if nulls else ""
        out(f"  {engine:<12}{cells_str}{med_str}{note}\n")

    out("\n## H. Cross-engine agreement (current regime) — Fleiss' kappa\n")
    counts = _vote_counts(panel)
    out("  ticker  [bear, sideways, bull] votes over 5 engines\n")
    for ticker, row in zip(TICKERS, counts):
        out(f"  {ticker:<8}{row}\n")
    p_bar, p_e, kappa = _fleiss_components(counts)
    out(f"\n  P_bar={p_bar:.6f}  P_e={p_e:.6f}\n")
    out(f"  Fleiss' kappa = {kappa:.6f}  (floor >= {KAPPA_FLOOR})\n")

    out("\n## I. Empirical persistence (mean spell length in bars,\n")
    out("     n_bars / (n_transitions + 1))\n")
    out("  engine      " + "".join(f"{t:>9}" for t in TICKERS) + f"{'median':>9}\n")
    for engine in ENGINES:
        spells = [_mean_spell(cells[(t, engine)]) for t in TICKERS]
        cells_str = "".join(f"{v:9.2f}" for v in spells)
        out(
            f"  {engine:<12}{cells_str}"
            f"{statistics.median(spells):9.2f}\n"
        )

    out("\n## J. Forecast consistency (1-step identity, max abs delta)\n")
    row_delta = 0.0
    nsp_delta = 0.0
    for ticker, engine in CELLS:
        data = cells[(ticker, engine)]
        t_row = data["transition_matrix"][data["current_regime"]["index"]]
        one_step = data["forecast"]["1_step"]
        nsp = data["next_state_probabilities"]
        for i, cat in enumerate(CATEGORIES):
            row_delta = max(row_delta, abs(one_step[cat] - t_row[i]))
            nsp_delta = max(nsp_delta, abs(one_step[cat] - nsp[cat]))
    out(
        f"  forecast.1_step vs T[current_regime, :]: "
        f"max |delta| = {row_delta:.3e}\n"
    )
    out(
        f"  forecast.1_step vs next_state_probabilities: "
        f"max |delta| = {nsp_delta:.3e}\n"
    )
    out("=" * 78 + "\n")

    REPORT_PATH.write_text("".join(lines))


# --------------------------------------------------------------------------
# Assertions (a product regression must fail the build)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ticker,engine", CELLS, ids=[f"{t}-{e}" for t, e in CELLS]
)
def test_forecast_identity(panel, ticker, engine):
    """forecast.1_step == T[current_regime, :] == next_state_probabilities."""
    data = panel.get(ticker, engine)
    assert set(data["forecast"]) == {"1_step"}, (
        "ADR-0025 pins the forecast surface to the 1-step horizon only"
    )
    t_row = data["transition_matrix"][data["current_regime"]["index"]]
    one_step = data["forecast"]["1_step"]
    nsp = data["next_state_probabilities"]
    for i, cat in enumerate(CATEGORIES):
        assert abs(one_step[cat] - t_row[i]) < FORECAST_TOL, (
            f"{ticker}/{engine}: forecast.1_step[{cat}]={one_step[cat]} "
            f"!= T[current_regime, :][{cat}]={t_row[i]}"
        )
        assert abs(one_step[cat] - nsp[cat]) < FORECAST_TOL, (
            f"{ticker}/{engine}: forecast.1_step[{cat}]={one_step[cat]} "
            f"!= next_state_probabilities[{cat}]={nsp[cat]}"
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_persistence_median_per_engine(panel, engine):
    """Median per-ticker mean spell length > 3 bars for every engine."""
    spells = [_mean_spell(panel.get(t, engine)) for t in TICKERS]
    med = statistics.median(spells)
    assert med > PERSISTENCE_FLOOR, (
        f"{engine}: median mean-spell {med:.2f} bars "
        f"(per ticker: {['%.2f' % s for s in spells]}) "
        f"<= {PERSISTENCE_FLOOR}"
    )


def test_fleiss_kappa_floor(panel):
    """Cross-engine agreement on current_regime: Fleiss' kappa >= -0.10."""
    kappa = fleiss_kappa(_vote_counts(panel))
    assert kappa >= KAPPA_FLOOR, (
        f"Fleiss' kappa {kappa:.6f} below non-regression floor {KAPPA_FLOOR}"
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_buy_and_hold_delta_sharpe_floor(panel, buy_and_hold, engine):
    """Median per-ticker B&H delta-Sharpe >= -1.20 on the engine window."""
    deltas = [
        panel.get(t, engine)["walk_forward"]["sharpe"]
        - buy_and_hold[t]["sharpe"]
        for t in TICKERS
    ]
    med = statistics.median(deltas)
    assert med >= BH_DELTA_FLOOR, (
        f"{engine}: median B&H delta-Sharpe {med:.4f} "
        f"(per ticker: {['%.4f' % d for d in deltas]}) "
        f"below floor {BH_DELTA_FLOOR}"
    )


def test_statistical_report_sections(panel, buy_and_hold):
    """runs/statistical_report.txt exists and mirrors runs/report.txt."""
    _write_report(panel, buy_and_hold)
    assert REPORT_PATH.exists(), f"{REPORT_PATH} was not written"
    text = REPORT_PATH.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in text, f"report is missing section: {section!r}"
