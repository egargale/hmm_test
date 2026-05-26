# ADR-004: Extract CLI Data Loading Seam

**Status**: Proposed
**Date**: 2026-05-26
**Priority**: Worth Exploring

## Context

`cli.py` does three things in `main()`: argument parsing, data loading (yfinance download + CSV parsing inline), and output formatting. The data loading logic (lines 176–195) is particularly entangled — yfinance download, MultiIndex column flattening, and OHLCV extraction are all inline in `main()`.

This matters because adding a new data source (database, REST API, Parquet file) means editing `main()` directly rather than adding an adapter at a seam.

## Current Structure

```python
# cli.py main() — data loading section (simplified)
if args.csv is not None:
    source = args.csv
    prices = load_from_csv(args.csv)
else:
    source = args.ticker
    ohlcv_raw = yfinance.download(args.ticker, period="10y", progress=False)
    if isinstance(ohlcv_raw.columns, pd.MultiIndex):
        ohlcv_raw.columns = [c[0].lower() for c in ohlcv_raw.columns]
    else:
        ohlcv_raw.columns = [c.lower() for c in ohlcv_raw.columns]
    prices = ohlcv_raw["close"]
    if args.engine in ("messina", "hmm"):
        ohlcv = ohlcv_raw[["open", "high", "low", "close", "volume"]]
```

### Problems

1. **yfinance is a hard dependency in the CLI path** — the import is inline (`import yfinance`) which is good for optional dependency handling, but the column-flattening logic belongs to the yfinance adapter, not the orchestrator.

2. **OHLCV conditionality leaks into CLI** — the CLI knows that messina/hmm engines need OHLCV and conditionally extracts it. This is engine knowledge leaking into the data-loading layer.

3. **No testable data-loading seam** — to test CLI with a mock data source, you'd have to mock `yfinance.download` or `load_from_csv` separately.

## Proposed Design

Extract a `load_prices()` function that encapsulates data source resolution:

```python
def load_prices(
    source: str,
    *,
    needs_ohlcv: bool = False,
) -> tuple[pd.Series, pd.DataFrame | None, str]:
    """Load prices (and optionally OHLCV) from a data source.

    Returns (prices, ohlcv_or_none, source_label).
    """
    if source.endswith(".csv") or Path(source).exists():
        prices = load_from_csv(source)
        return prices, None, source
    else:
        # yfinance ticker
        ohlcv_raw = yfinance.download(source, period="10y", progress=False)
        # flatten, normalise columns...
        prices = ohlcv_raw["close"]
        ohlcv = ohlcv_raw[["open", "high", "low", "close", "volume"]] if needs_ohlcv else None
        return prices, ohlcv, source
```

`cli.py` becomes:

```python
prices, ohlcv, source_label = load_prices(args.csv or args.ticker, needs_ohlcv=args.engine in ("messina", "hmm"))
```

Future data sources add an adapter in `load_prices()`, not in `main()`.

## Consequences

**Positive:**
- Adding a data source = edit `load_prices()`, not `main()`
- Testable: mock `load_prices` in CLI tests
- yfinance column-flattening logic is contained in one function
- Locality: data source specifics stop leaking into orchestration

**Negative:**
- Small additional indirection
- `needs_ohlcv` param is still engine-awareness at the loading layer (could be eliminated if ADR-002 engine protocol handles OHLCV internally)

## Related

- [[ADR-002]] Engine seam — with `RegimeEngine` protocol, each engine could own its data requirements, making `needs_ohlcv` unnecessary
- [[ADR-001]] Dead weight removal — `load_from_yfinance` is already exported from `data_processing/__init__.py` but dead; this PR would revive the concept in a different form
