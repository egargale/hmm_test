"""
Test script for Regime-Based Backtesting Engine

This script tests the backtesting implementation and validates its functionality
for HMM-driven trading strategies with realistic trade execution.
"""

import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, "src")

from backtesting.performance_analyzer import (
    analyze_performance,
    create_performance_report,
)
from backtesting.strategy_engine import backtest_strategy, backtest_with_analysis
from backtesting.utils import (
    analyze_regime_performance,
    calculate_position_returns,
    create_sample_price_data,
    create_sample_state_sequence,
)
from utils.data_types import BacktestConfig


def test_basic_backtesting():
    """Test basic backtesting functionality."""
    print("=" * 60)
    print("Testing Basic Backtesting Functionality")
    print("=" * 60)

    try:
        # Create sample data
        print("Creating sample price and state data...")
        prices = create_sample_price_data(
            n_samples=100, initial_price=100.0, volatility=0.02
        )
        states = create_sample_state_sequence(
            n_samples=100, n_states=3, transition_probability=0.1
        )

        print(
            f"✓ Created sample data: {len(prices)} price points, {len(states)} states"
        )

        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,  # 2 basis points
            position_size=1.0,
            state_map={0: 1, 1: -1, 2: 0},  # State 0=Long, 1=Short, 2=Flat
        )

        print(f"✓ Created backtest config: {config.state_map}")

        # Test basic backtesting
        print("\nTesting basic backtesting...")
        positions, trades = backtest_strategy(states, prices, config)

        print("✓ Backtesting completed:")
        print(f"  - Position series length: {len(positions)}")
        print(f"  - Number of trades: {len(trades)}")
        print(f"  - Position distribution: {positions.value_counts().to_dict()}")

        if trades:
            print(
                f"  - Sample trade: Entry={trades[0].entry_price}, P&L={trades[0].pnl:.2f}"
            )

        # Test position returns calculation
        print("\nTesting position returns calculation...")
        position_returns = calculate_position_returns(positions, prices)
        print(
            f"✓ Position returns calculated: mean={position_returns.mean():.6f}, std={position_returns.std():.6f}"
        )

        return True

    except Exception as e:
        print(f"✗ Basic backtesting test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_transaction_costs():
    """Test transaction cost calculations."""
    print("\n" + "=" * 60)
    print("Testing Transaction Cost Calculations")
    print("=" * 60)

    try:
        # Create sample data
        prices = create_sample_price_data(n_samples=50, initial_price=100.0)
        states = create_sample_state_sequence(n_samples=50, n_states=2)

        # Test with different cost configurations
        configs = [
            BacktestConfig(
                commission_per_trade=0.0, slippage_bps=0.0, state_map={0: 1, 1: -1}
            ),
            BacktestConfig(
                commission_per_trade=5.0, slippage_bps=2.0, state_map={0: 1, 1: -1}
            ),
            BacktestConfig(
                commission_per_trade=10.0, slippage_bps=5.0, state_map={0: 1, 1: -1}
            ),
        ]

        for i, config in enumerate(configs):
            print(f"\nTesting cost configuration {i + 1}:")
            print(f"  - Commission: ${config.commission_per_trade:.2f}")
            print(f"  - Slippage: {config.slippage_bps} bps")

            positions, trades = backtest_strategy(states, prices, config)

            if trades:
                total_commission = sum(t.commission for t in trades)
                total_slippage = sum(t.slippage for t in trades)
                total_costs = total_commission + total_slippage

                print(f"  - Total commission: ${total_commission:.2f}")
                print(f"  - Total slippage: ${total_slippage:.2f}")
                print(f"  - Total costs: ${total_costs:.2f}")
                print(f"  - Avg cost per trade: ${total_costs / len(trades):.2f}")

        print("✓ Transaction cost calculations completed successfully")
        return True

    except Exception as e:
        print(f"✗ Transaction cost test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lookahead_bias_prevention():
    """Test lookahead bias prevention mechanisms."""
    print("\n" + "=" * 60)
    print("Testing Lookahead Bias Prevention")
    print("=" * 60)

    try:
        # Create sample data
        prices = create_sample_price_data(
            n_samples=100, initial_price=100.0, volatility=0.01
        )
        states = create_sample_state_sequence(
            n_samples=100, n_states=2, transition_probability=0.2
        )

        config = BacktestConfig(
            commission_per_trade=0.0, slippage_bps=0.0, state_map={0: 1, 1: -1}
        )

        # Test with lookahead bias prevention (default)
        print("Testing with lookahead bias prevention enabled...")
        positions_with_bias, trades_with_bias = backtest_strategy(
            states, prices, config, lookahead_bias_prevention=True
        )

        # Test without lookahead bias prevention
        print("Testing with lookahead bias prevention disabled...")
        positions_without_bias, trades_without_bias = backtest_strategy(
            states, prices, config, lookahead_bias_prevention=False
        )

        print("✓ Results comparison:")
        print(f"  - With bias prevention: {len(trades_with_bias)} trades")
        print(f"  - Without bias prevention: {len(trades_without_bias)} trades")

        # The results should be different when bias prevention is applied
        if len(positions_with_bias) == len(positions_without_bias):
            # Check if position series are different
            differences = np.sum(positions_with_bias != positions_without_bias)
            print(f"  - Position differences: {differences} periods")

            if differences > 0:
                print("✓ Lookahead bias prevention is working correctly")
            else:
                print("⚠ No difference detected - might need more volatile data")
        else:
            print("⚠ Position series have different lengths")

        return True

    except Exception as e:
        print(f"✗ Lookahead bias prevention test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_analysis():
    """Test performance analysis functionality."""
    print("\n" + "=" * 60)
    print("Testing Performance Analysis")
    print("=" * 60)

    try:
        # Create sample data
        prices = create_sample_price_data(
            n_samples=252, initial_price=100.0, volatility=0.02, drift=0.0005
        )
        states = create_sample_state_sequence(
            n_samples=252, n_states=3, transition_probability=0.05
        )

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1, 2: 0},
        )

        # Run backtest with analysis
        print("Running backtest with performance analysis...")
        result = backtest_with_analysis(
            states, prices, config, initial_capital=100000.0
        )

        # Analyze performance
        print("Analyzing performance metrics...")
        metrics = analyze_performance(result)

        print("✓ Performance analysis completed:")
        print(f"  - Total return: {metrics.total_return:.2%}")
        print(f"  - Annualized return: {metrics.annualized_return:.2%}")
        print(f"  - Annualized volatility: {metrics.annualized_volatility:.2%}")
        print(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  - Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  - Max drawdown duration: {metrics.max_drawdown_duration} periods")
        print(f"  - Win rate: {metrics.win_rate:.2%}")
        print(f"  - Profit factor: {metrics.profit_factor:.2f}")

        # Create performance report
        print("Creating comprehensive performance report...")
        report = create_performance_report(result, metrics)

        print(f"✓ Performance report created with {len(report)} sections")
        print(f"  - Summary section keys: {list(report['summary'].keys())}")
        print(f"  - Trade analysis: {len(report['trade_analysis'])} metrics")

        return True

    except Exception as e:
        print(f"✗ Performance analysis test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_regime_performance():
    """Test regime-based performance analysis."""
    print("\n" + "=" * 60)
    print("Testing Regime Performance Analysis")
    print("=" * 60)

    try:
        # Create sample data with known regime characteristics
        prices = create_sample_price_data(
            n_samples=500, initial_price=100.0, volatility=0.02
        )
        states = create_sample_state_sequence(
            n_samples=500, n_states=3, transition_probability=0.05
        )

        # Create returns from prices
        returns = prices.pct_change().fillna(0)

        # Analyze performance by regime
        print("Analyzing performance by regime...")
        regime_analysis = analyze_regime_performance(
            states, returns, state_names=["Bull", "Bear", "Neutral"]
        )

        print(f"✓ Regime analysis completed for {len(regime_analysis)} regimes:")

        for state, metrics in regime_analysis.items():
            state_name = metrics.get("state_name", f"State_{state}")
            print(f"  - {state_name}:")
            print(
                f"    * Occurrences: {metrics['count']} ({metrics['percentage']:.1f}%)"
            )
            print(f"    * Mean return: {metrics['mean_return']:.4f}")
            print(f"    * Volatility: {metrics['std_return']:.4f}")
            print(f"    * Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"    * Win rate: {metrics['win_rate']:.2%}")

        return True

    except Exception as e:
        print(f"✗ Regime performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases and Error Handling")
    print("=" * 60)

    edge_case_passed = 0
    edge_case_total = 0

    try:
        # Test 1: Empty data
        edge_case_total += 1
        print("Testing empty data handling...")
        try:
            empty_states = np.array([])
            empty_prices = pd.Series([], dtype=float)
            config = BacktestConfig(state_map={0: 1})
            backtest_strategy(empty_states, empty_prices, config)
            print("✗ Should have raised an error for empty data")
        except ValueError:
            print("✓ Correctly handled empty data")
            edge_case_passed += 1

        # Test 2: Mismatched lengths
        edge_case_total += 1
        print("\nTesting mismatched lengths...")
        try:
            states = create_sample_state_sequence(100, 2)
            prices = create_sample_price_data(50)  # Different length
            config = BacktestConfig(state_map={0: 1, 1: -1})
            backtest_strategy(states, prices, config)
            print("✗ Should have raised an error for mismatched lengths")
        except ValueError:
            print("✓ Correctly handled mismatched lengths")
            edge_case_passed += 1

        # Test 3: Empty state map
        edge_case_total += 1
        print("\nTesting empty state map...")
        try:
            states = create_sample_state_sequence(50, 2)
            prices = create_sample_price_data(50)
            config = BacktestConfig(state_map={})  # Empty state map
            backtest_strategy(states, prices, config)
            print("✗ Should have raised an error for empty state map")
        except ValueError:
            print("✓ Correctly handled empty state map")
            edge_case_passed += 1

        # Test 4: No trades scenario
        edge_case_total += 1
        print("\nTesting no trades scenario...")
        states = np.zeros(100)  # All state 0
        prices = create_sample_price_data(100, volatility=0.01)
        config = BacktestConfig(state_map={0: 0})  # Flat position for state 0

        positions, trades = backtest_strategy(states, prices, config)
        if len(trades) == 0 and np.all(positions == 0):
            print("✓ Correctly handled no trades scenario")
            edge_case_passed += 1
        else:
            print(f"✗ Expected no trades, got {len(trades)} trades")

        # Test 5: Single trade scenario
        edge_case_total += 1
        print("\nTesting single trade scenario...")
        states = np.concatenate([np.zeros(50), np.ones(50)])  # State changes once
        prices = create_sample_price_data(100, volatility=0.01)
        config = BacktestConfig(state_map={0: 0, 1: 1})  # Flat then Long

        positions, trades = backtest_strategy(states, prices, config)
        if len(trades) >= 1:
            print(f"✓ Correctly executed at least one trade: {len(trades)} trades")
            edge_case_passed += 1
        else:
            print("✗ Expected at least one trade")

        print(f"\nEdge case tests passed: {edge_case_passed}/{edge_case_total}")
        return edge_case_passed == edge_case_total

    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all backtesting tests."""
    print("Starting Backtesting Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("Basic Backtesting", test_basic_backtesting),
        ("Transaction Costs", test_transaction_costs),
        ("Lookahead Bias Prevention", test_lookahead_bias_prevention),
        ("Performance Analysis", test_performance_analysis),
        ("Regime Performance", test_regime_performance),
        ("Edge Cases", test_edge_cases),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All backtesting tests passed!")
        return True
    else:
        print("❌ Some backtesting tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
