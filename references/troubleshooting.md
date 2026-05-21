# Troubleshooting

## "HMM fails to converge"

**Symptom**: HMM output shows `"available": false` with reason containing "convergence" or no states returned.

**Causes and fixes:**

1. **Insufficient data**: HMM needs at least `n_states + 1` clean rows after feature engineering. With `n_states=3`, you need at least ~100 bars.
   - **Fix**: Use a larger dataset, or switch to threshold-only (`--no-hmm`).

2. **Too many features relative to data**: The covariance matrix becomes singular when `n_features > n_samples`.
   - **Fix**: Reduce features. Use `--no-hmm` for quick analysis, or trim the feature set.

3. **Outliers**: Extreme price moves can destabilise EM.
   - **Fix**: The pipeline applies standard scaling and outlier detection by default. If you see convergence failures, check for data errors (e.g. price of 0, negative prices).

4. **Convergence threshold too tight**: Default `tol=1e-3`. With noisy financial data, tight convergence may cause infinite loops.
   - **Fix**: Increase `tol` to `1e-2` in `run_hmm_regime()` call.

5. **Random initialisation**: Some random seeds produce poor initialisations.
   - **Fix**: Increase `num_restarts` (default 3) or try a different `random_state`.

## "NaN in output"

**Symptom**: Sharpe ratio = `null`, max_drawdown = `null`, or other fields are `null` in JSON.

**Causes:**

1. **Insufficient data for walk-forward**: `len(prices) < min_train + 1`.
   - **Fix**: Reduce `--min-train` (e.g. `--min-train 50`) or provide more data.

2. **Stationary distribution computation failed**: All eigenvalues are complex or the eigenvalue-1 search fails.
   - **This is expected for degenerate transition matrices** (e.g. all states identical). Check regime diversity — if all bars are classified as one regime, the transition matrix has no useful structure.

3. **Short equity curve**: Fewer than 2 valid bars in the walk-forward equity curve.
   - **Fix**: Lower `--min-train` or use more data.

## "File not found for CSV"

**Symptom**: `FileNotFoundError: CSV file not found: BTC.csv`

**Causes:**

1. **Relative path resolved from wrong directory**: The `--csv` path resolves relative to the current working directory, not the script location.
   - **Fix**: Use an absolute path (`--csv /full/path/to/data.csv`) or run from the directory containing the file.

2. **Filename case mismatch**: `BTC.csv` vs `btc.csv`.
   - **Fix**: Check exact filename with `ls`.

## "yfinance returns empty"

**Symptom**: `ValueError: No data returned for ticker: ES=F` or empty output.

**Causes:**

1. **Invalid ticker symbol**: yfinance tickers differ from broker symbols. Yahoo Finance uses specific formats.
   - **Fix**: Check the ticker on finance.yahoo.com. Common mappings:
     - S&P 500 E-mini futures: `ES=F`
     - Bitcoin USD: `BTC-USD`
     - S&P 500 ETF: `SPY`
     - Gold futures: `GC=F`

2. **yfinance not installed**: Optional dependency.
   - **Fix**: `pip install yfinance` or `uv sync --extra yfinance`.

3. **Rate limiting**: Yahoo Finance may throttle repeated requests.
   - **Fix**: Add delays between ticker requests. Use `--csv` with pre-downloaded data for batch analysis.

4. **No data for date range**: The ticker may have been delisted or data may not start until a certain date.
   - **Fix**: Try a shorter period using `load_from_yfinance(ticker, period="5y")`.

## "Signal always 0"

**Symptom**: `signal` output is consistently 0.0 or near-zero.

**Causes:**

1. **Bull and bear probabilities are equal**: `P(bull) ≈ P(bear)`.
   - This is **normal** in sideways markets. The signal correctly indicates no directional bias.

2. **Transition matrix is uniform**: All rows are approximately `[1/3, 1/3, 1/3]`.
   - The regime classification isn't finding structure. **Try**:
     - Adjust `--threshold` (lower for more regime differentiation).
     - Adjust `--window` (shorter for more responsive classification).
     - Check that the input data has meaningful trends (not pure noise).

3. **All bars classified as the same state**: If `classify_regimes` returns all sideways (1), the transition matrix will be degenerate.
   - **Fix**: The `--threshold` is too high relative to the return distribution. Try a lower value.

## "Module not found" when running regime.py

**Symptom**: `ModuleNotFoundError: No module named 'regime'`

**Cause**: Python path does not include the `scripts/` directory.

**Fix**: Run from the project root with `PYTHONPATH` set, or use the built-in path injection:

```bash
# From project root:
python scripts/regime.py --csv BTC.csv --json

# Or set PYTHONPATH explicitly:
PYTHONPATH=scripts python scripts/regime.py --csv BTC.csv --json
```

The script automatically adds its parent directory (`scripts/`) to `sys.path` on startup, but this only works when executed directly. For programmatic use:

```python
import sys
sys.path.insert(0, "/path/to/hmm_test/scripts")
from regime.markov_chain import classify_regimes
```

## "No numeric column found in CSV"

**Symptom**: `ValueError: No numeric column found in file.csv`

**Cause**: The CSV auto-detection could not find a column that looks like a close price.

**Fix**: The CSV must have at minimum a date/timestamp column and a numeric price column. Common formats auto-detected:
- `Date,Open,High,Low,Close,Volume`
- `DateTime,Open,High,Low,Close,Volume`
- `Date,Time,Open,High,Low,Last,Volume`

If your format differs, use `load_from_csv(path, date_col="MyDate", close_col="MyPrice")` programmatically.

## Slow HMM fitting on large datasets

**Symptom**: `run_hmm_regime()` takes minutes or hours.

**Fixes:**

1. **Reduce features**: Pass only essential columns (returns, volatility) instead of full feature set.
2. **Downsample**: Use daily instead of hourly/minute data.
3. **Use diagonal covariance**: `covariance_type="diag"` is significantly faster than `"full"`.
4. **Increase tolerance**: `tol=1e-2` converges faster than `tol=1e-3`.
5. **Use threshold mode**: `--no-hmm` for quick checks; HMM only when depth needed.

## ImportError: hmmlearn

**Symptom**: `ImportError: No module named 'hmm'` (hmmlearn imports as `hmm`).

**Fix**: Install hmmlearn:
```bash
pip install hmmlearn
```
Or use `--no-hmm` to skip HMM analysis entirely.
