# CLI data loading seam

The CLI orchestrator (`cli.py`) inlined yfinance downloading, MultiIndex flattening, and OHLCV extraction directly in `main()`. Adding a new data source meant editing the orchestrator. We extract a unified `load_prices()` entry point in the existing `csv_auto_detect.py` module so the CLI's data-loading block reduces to a single call.

## Considered options

### A) New `loader.py` module

Create a separate module for the unified loader. Clean break, descriptive name.

**Rejected because**: it would duplicate yfinance download logic already in `csv_auto_detect.py`. The existing module already handles both CSV and yfinance — it just needs a unified entry point returning `(prices, ohlcv, source_label)`.

### B) Extend `csv_auto_detect.py` with `load_prices()` (chosen)

Add `load_prices(*, csv=None, ticker=None)` to the existing module. The yfinance path always returns full OHLCV; the CSV path returns `ohlcv=None`. The function owns mutual-exclusivity validation and source-label derivation. `load_from_yfinance()` becomes an internal helper. `load_price_series()` is deprecated and removed from `__all__`. `load_from_csv()` remains public.

**Chosen because**: one module, one download path, no duplication. The function is source-agnostic — it has no knowledge of engines. ADR-0003's self-containment contract already establishes that each engine decides what data it needs; the loader just provides everything it can.

### C) Add `engine` parameter to `load_prices()`

Let the loader decide whether to populate OHLCV based on which engine was selected.

**Rejected because**: it couples the loader to engine-specific knowledge. The yfinance path already downloads all columns — returning OHLCV costs nothing. Each engine already validates its own data requirements per ADR-0003.
