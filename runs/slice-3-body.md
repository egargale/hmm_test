## Parent

#111

## What to build

Two deliverables: (1) expose the Slice 2 harness as a CLI flag so a user can reproduce the 5×5 report from a clean clone without invoking pytest, and (2) write ADR-0025 documenting the forecast-algebra invariant the fix in Slice 1 establishes. The CLI work adds a new `--eval-statistical` mode to the existing `hmm-regime` entry point; the ADR work captures the design record so future contributors do not reintroduce the off-by-one.

End-to-end behaviour after this slice: running `hmm-regime --eval-statistical SPY,NVDA,BTC,KO,XLE` (or any ticker list) invokes the same harness code from Slice 2 and prints a markdown table to stderr matching the format of `runs/statistical_report.txt`. ADR-0025 lives in `docs/adr/0025-forecast-algebra-invariant.md` and is cross-referenced from `docs/adr/README.md` and the existing `CONTEXT.md` glossary (the latter only if a new domain term was introduced).

## Acceptance criteria

- [ ] A new `--eval-statistical` flag is added to the `hmm-regime` CLI. It accepts a comma-separated ticker list (e.g. `--eval-statistical SPY,NVDA,BTC,KO,XLE`) and writes the same markdown report that the Slice 2 harness writes. The flag is mutually exclusive with `--ticker`, `--csv`, `--eval-tickers`, and `--eval-csv` (consistent with the existing eval-mode exclusivity).
- [ ] A new function `run_eval_statistical(tickers, engines)` is added to the eval module. It returns the same dict shape as `run_eval_tickers` but with two additional fields per result row: `fleiss_kappa` (a single float for the panel) and `bnh_delta` (the engine Sharpe minus the buy-and-hold Sharpe for the same ticker and window).
- [ ] The presenter exposes a new `fmt="statistical_table"` mode for `format_eval` that renders the cross-engine agreement summary (one row per ticker, columns = engine votes, plus a majority column) and the buy-and-hold delta grid. The existing `fmt="table"` and `fmt="json"` paths are unchanged.
- [ ] The CLI behaviour matches the harness: running `hmm-regime --eval-statistical SPY,NVDA,BTC,KO,XLE` produces a report that is identical to what `pytest -m statistical` writes to `runs/statistical_report.txt`, modulo timestamps and any stochastic content.
- [ ] ADR-0025 exists at `docs/adr/0025-forecast-algebra-invariant.md`. The ADR follows the project's ADR format (Status / Context / Decision / Consequences). It documents: (a) the algebraic invariant that `MarkovStats.current_probs` is a one-hot for the current regime, (b) the rationale (the prior code path produced `T^2[i,:]` not `T[i,:]`), (c) the high-persistence consequence (HMM engines with persistence ≈ 1.0 hid the bug), and (d) cross-references to ADR-0011 (engine dispatch consolidation, because the fix lives at the dispatch-adjacent seam) and ADR-0018 (degenerate fit recovery, because the 2×2 case must be handled).
- [ ] `docs/adr/README.md` (or equivalent index) is updated to list ADR-0025 in the table of ADRs.
- [ ] Running `hmm-regime --help` shows the new flag in the eval-mode group with usage text matching the format of the other eval flags.

## Blocked by

- #112
- #113
