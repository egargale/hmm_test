# Summary of Fixes Made to main.py

## Issues Identified and Fixed:

1. **Column Name Parsing Issue**:
   - The CSV file had columns with leading spaces that weren't being handled correctly
   - Fixed by stripping whitespace from column names and handling both "DateTime"/"Close" and "Date"/"Time"/"Last" column formats

2. **Datetime Parsing Issue**:
   - The script was trying to use `parse_dates={"DateTime": ["Date", "Time"]}` incorrectly
   - Fixed by reading the CSV without parse_dates first and then combining Date/Time columns after reading

3. **Backtest Function Issues**:
   - The backtest function was recalculating log returns instead of using the existing 'log_ret' column
   - Fixed by using the existing 'log_ret' column and shifting it appropriately

4. **Performance Metrics Calculation**:
   - The perf_metrics function was calculating Sharpe ratio incorrectly
   - Fixed by properly calculating returns from the series differences

## Files Generated:
- BTC.hmm_states.csv: Contains the original data with added HMM states
- BTC.backtest.csv: Contains backtest results
- BTC.png: Visualization of the HMM states overlaid on price data

## Verification:
The script now runs successfully with the BTC.csv file and generates all expected output files with proper data.