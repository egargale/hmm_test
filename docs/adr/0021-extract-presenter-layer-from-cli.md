# 0021: Extract Presenter Layer from CLI

## Status

Accepted

## Context

`cli.py` grew to 621 lines and 25 changes (second-hottest file in the repo), accumulating seven distinct responsibilities: argument parsing, engine config assembly, transition limiting, terminal rendering, JSON serialization, eval mode dispatch, and saliency CSV output. The file was on track to breach 700 lines with the next feature drop.

The terminal rendering (`_print_terminal`, ~100 lines) and JSON serialization (~30 lines inline in `main()`) are pure presentation logic that has no business touching `sys.stderr`/`sys.stdout` directly. They can't be tested without subprocessing the entire CLI, and every formatting change forces a touch to the dispatch module.

The `_build_engine_config` function maps argparse Namespace → engine config dataclass. It's pure plumbing that belongs with the config types, not in the CLI dispatch shell.

The `_apply_transitions_limit` helper is used in both terminal and JSON paths but lived in `cli.py`, forcing both paths to know about it.

The `format_table` function in `eval.py` is presentation logic that doesn't belong in the computation harness.

## Decision

Extract a **presenter layer** into a new `presenter.py` module with four public functions:

- `format_pipeline(result, *, transitions_limit=None) -> str` — terminal-formatted string
- `serialize_pipeline(result, *, transitions_limit=None) -> dict` — JSON-compatible dict
- `format_eval(results, *, fmt="table"|"json") -> str` — eval table or JSON string
- `limit_transitions(transitions, limit) -> list` — pure filter/reverse

All functions are stateless, return strings or dicts, and never touch IO streams. The caller owns stdout/stderr.

Extract `_build_engine_config` into `regime/engine_config_builder.py`.

Move `format_table` from `eval.py` to `presenter.py` (with a backwards-compatible re-export in `eval.py`).

Keep `_write_saliency_csv` in `cli.py` — it's an fshmm-specific side effect, not a general presenter concern.

## Consequences

- **cli.py drops from 621 → 399 lines** (36% reduction), becoming a thin dispatch shell: parse args → build config → run pipeline → call presenter.
- **Presenter functions are testable without subprocessing**: construct a fake PipelineResult dict, call `format_pipeline()`, assert on the string.
- **Library users** can call `format_pipeline()` or `serialize_pipeline()` directly without importing argparse or sys.
- **Adding a new output format** (HTML, CSV report) means adding a function to `presenter.py`, not touching `cli.py`.
- **`eval.format_table` remains available** via backwards-compatible re-export.
- **One new architectural concept** ("Presenter") added to CONTEXT.md.
