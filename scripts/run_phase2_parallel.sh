#!/usr/bin/env bash
# Run Phase 2 sweep for all tickers in parallel
set -e
cd /home/enrico/.herdr/worktrees/hmm_test/worktree-brave-meadow-7468
mkdir -p test_data/eval-results/messina_sweep

tickers=(0700_HK BTC CRM KO SPY)

echo "Starting parallel sweep for ${#tickers[@]} tickers..."
echo "Each ticker: 192 param combos × ~30s = ~96 min"
echo "Running 5 in parallel: ~96 min total wall time"
echo "---"
date

for t in "${tickers[@]}"; do
    echo "Starting $t..."
    timeout 6000 python scripts/messina_sweep_phase2.py "$t" \
        > test_data/eval-results/messina_sweep/log_${t}.txt 2>&1 &
done

echo "All tickers launched. Waiting..."
wait
echo "---"
date
echo "All done!"

# Check results
for t in "${tickers[@]}"; do
    if [ -f "test_data/eval-results/messina_sweep/phase2_${t}.json" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('test_data/eval-results/messina_sweep/phase2_${t}.json'))))")
        echo "  $t: $count results ✓"
    else
        echo "  $t: NOT FOUND ✗"
    fi
done
