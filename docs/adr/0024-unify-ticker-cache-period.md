# 0024: Unify yfinance period in ticker_cache no-cache path

## Status

Accepted

## Context

`get_ticker_data()` used different `period` values for the cached vs no-cache paths:

- **Cached path** (cache miss): `yfinance.download(ticker, period="max")`
- **No-cache path** (`no_cache=True`): `yfinance.download(ticker, period="10y")`

This is a silent correctness bug. For tickers with more than 10 years of history (e.g. SPY, most major indices), `--no-cache` returns fewer bars than the cached path. Walk-forward backtest results diverge depending on the cache flag — but `--no-cache` is documented as a performance flag, not a data-semantics flag.

The existing tests mock `yfinance.download` and assert `period="10y"` for the no-cache path, codifying the inconsistency.

## Decision

Unify both paths to `period="max"`. The `--no-cache` flag controls whether the disk cache is read/written, not how much data is fetched.

## Consequences

- **Both paths now fetch the same data window** — `--no-cache` no longer silently truncates to 10 years.
- **Existing test updated** — `TestNoCache` now asserts `period="max"` instead of `period="10y"`.
- **New regression test** — `TestPeriodConsistency` verifies both paths call `yfinance.download` with the same period.
- **No API change** — `get_ticker_data()` signature unchanged.
