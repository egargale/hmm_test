"""
Test script for Performance Metrics and Bias Prevention (Task 7)

This script tests the advanced performance metrics calculation and
lookahead bias detection and prevention mechanisms.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from utils.data_types import BacktestConfig
from backtesting.performance_metrics import (
    calculate_performance, infer_trading_frequency,
    get_annualization_factor, validate_performance_metrics
)
from backtesting.bias_prevention import (
    detect_lookahead_bias, validate_backtest_realism,
    apply_bias_prevention, create_bias_prevention_report
)
from backtesting.utils import create_sample_price_data, create_sample_state_sequence

def test_core_performance_metrics():
    """Test core performance metrics calculation."""
    print("=" * 60)
    print("Testing Core Performance Metrics")
    print("=" * 60)

    try:
        # Create sample equity curve
        print("Creating sample equity curve...")
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns with some drift
        equity_curve = pd.Series(100000 * (1 + np.cumsum(returns)), index=dates, name='equity')

        print(f"✓ Created equity curve: {len(equity_curve)} days")
        print(f"  - Initial value: {equity_curve.iloc[0]:.2f}")
        print(f"  - Final value: {equity_curve.iloc[-1]:.2f}")
        print(f"  - Total return: {(equity_curve.iloc[-1]/equity_curve.iloc[0] - 1):.2%}")

        # Test frequency inference
        print("\nTesting frequency inference...")
        inferred_freq = infer_trading_frequency(equity_curve)
        print(f"✓ Inferred frequency: {inferred_freq}")

        # Test annualization factors
        print("\nTesting annualization factors...")
        test_frequencies = ['daily', 'weekly', 'monthly', 'hourly']
        for freq in test_frequencies:
            factor = get_annualization_factor(freq)
            print(f"  - {freq}: {factor}")

        # Test core performance calculation
        print("\nTesting core performance calculation...")
        metrics = calculate_performance(equity_curve, risk_free_rate=0.02)

        print(f"✓ Core performance metrics calculated:")
        print(f"  - Total return: {metrics.total_return:.2%}")
        print(f"  - Annualized return: {metrics.annualized_return:.2%}")
        print(f"  - Annualized volatility: {metrics.annualized_volatility:.2%}")
        print(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  - Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  - Max drawdown duration: {metrics.max_drawdown_duration} periods")
        print(f"  - Calmar ratio: {metrics.calmar_ratio:.2f}")

        # Test metrics validation
        print("\nTesting metrics validation...")
        validation = validate_performance_metrics(metrics)
        print(f"✓ Validation passed: {validation['valid']}")
        if validation['warnings']:
            print(f"  - Warnings: {len(validation['warnings'])}")
        if validation['errors']:
            print(f"  - Errors: {len(validation['errors'])}")

        return True

    except Exception as e:
        print(f"✗ Core performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_performance_metrics():
    """Test advanced performance metrics with trade data."""
    print("\n" + "=" * 60)
    print("Testing Advanced Performance Metrics")
    print("=" * 60)

    try:
        # Create a backtest result with trades
        print("Creating backtest result with trades...")
        from backtesting.strategy_engine import backtest_with_analysis

        # Create sample data
        prices = create_sample_price_data(n_samples=252, volatility=0.02, drift=0.0005)
        states = create_sample_state_sequence(n_samples=252, n_states=3, transition_probability=0.05)

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1, 2: 0}
        )

        # Run backtest
        result = backtest_with_analysis(states, prices, config)

        print(f"✓ Backtest completed: {len(result.trades)} trades, {len(result.equity_curve)} periods")

        # Test comprehensive performance analysis
        print("\nTesting comprehensive performance analysis...")
        from backtesting.performance_analyzer import analyze_performance
        metrics = analyze_performance(result, risk_free_rate=0.02)

        print(f"✓ Advanced performance metrics:")
        print(f"  - Total return: {metrics.total_return:.2%}")
        print(f"  - Annualized return: {metrics.annualized_return:.2%}")
        print(f"  - Annualized volatility: {metrics.annualized_volatility:.2%}")
        print(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  - Sortino ratio: {metrics.sortino_ratio:.2f}")
        print(f"  - Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  - Calmar ratio: {metrics.calmar_ratio:.2f}")
        print(f"  - Win rate: {metrics.win_rate:.2%}")
        print(f"  - Profit factor: {metrics.profit_factor:.2f}")

        # Create performance report
        print("\nTesting performance report generation...")
        from backtesting.performance_analyzer import create_performance_report
        report = create_performance_report(result, metrics)

        print(f"✓ Performance report created with {len(report)} sections")
        print(f"  - Summary: {len(report['summary'])} fields")
        print(f"  - Returns: {len(report['returns'])} fields")
        print(f"  - Risk metrics: {len(report['risk_metrics'])} fields")

        return True

    except Exception as e:
        print(f"✗ Advanced performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bias_prevention():
    """Test lookahead bias detection and prevention."""
    print("\n" + "=" * 60)
    print("Testing Lookahead Bias Detection and Prevention")
    print("=" * 60)

    try:
        # Create sample data with known bias issues
        print("Creating sample data with potential bias issues...")
        prices = create_sample_price_data(n_samples=100, volatility=0.02)
        states = create_sample_state_sequence(n_samples=100, n_states=2, transition_probability=0.1)
        timestamps = pd.date_range('2020-01-01', periods=100, freq='D')

        # Create position mapping
        state_map = {0: 1, 1: -1}

        # Create positions without proper lag (simulating bias)
        biased_positions = np.array([state_map.get(s, 0) for s in states])
        positions_series = pd.Series(biased_positions, index=timestamps)

        print(f"✓ Created test data: {len(states)} periods")

        # Test bias detection
        print("\nTesting bias detection...")
        bias_result = detect_lookahead_bias(
            states=states,
            positions=positions_series,
            timestamps=timestamps,
            state_map=state_map,
            lag_periods=1
        )

        print(f"✓ Bias detection completed:")
        print(f"  - Has lookahead bias: {bias_result.has_lookahead_bias}")
        print(f"  - Overall risk score: {bias_result.overall_risk_score:.3f}")
        print(f"  - Timing violations: {len(bias_result.timing_violations)}")
        print(f"  - Position shift violations: {len(bias_result.position_shift_violations)}")
        print(f"  - Data leakage detected: {bias_result.data_leakage_detected}")

        # Test bias prevention report
        print("\nTesting bias prevention report...")
        report = create_bias_prevention_report(bias_result)
        print(f"✓ Bias prevention report generated ({len(report)} characters)")

        # Test bias prevention application
        print("\nTesting bias prevention application...")
        lagged_states, lagged_positions = apply_bias_prevention(
            states, positions_series, lag_periods=1
        )

        print(f"✓ Bias prevention applied:")
        print(f"  - Original states: {states[:5]}")
        print(f"  - Lagged states: {lagged_states[:5]}")
        print(f"  - Original positions: {positions_series.head().values}")
        print(f"  - Lagged positions: {lagged_positions.head().values}")

        # Re-run bias detection on corrected data
        corrected_result = detect_lookahead_bias(
            states=lagged_states,
            positions=lagged_positions,
            timestamps=timestamps,
            state_map=state_map,
            lag_periods=1
        )

        print(f"\n✓ Re-detection after prevention:")
        print(f"  - Has lookahead bias: {corrected_result.has_lookahead_bias}")
        print(f"  - Overall risk score: {corrected_result.overall_risk_score:.3f}")

        return True

    except Exception as e:
        print(f"✗ Bias prevention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_realism_validation():
    """Test backtest realism validation."""
    print("\n" + "=" * 60)
    print("Testing Backtest Realism Validation")
    print("=" * 60)

    try:
        # Create backtest results with various characteristics
        print("Creating backtest results for validation...")
        from backtesting.strategy_engine import backtest_with_analysis

        # Test 1: Normal realistic backtest
        print("\nTesting normal realistic backtest...")
        prices_normal = create_sample_price_data(n_samples=252, volatility=0.02, drift=0.0003)
        states_normal = create_sample_state_sequence(n_samples=252, n_states=2, transition_probability=0.1)

        config = BacktestConfig(
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1}
        )

        result_normal = backtest_with_analysis(states_normal, prices_normal, config)
        validation_normal = validate_backtest_realism(result_normal)

        print(f"✓ Normal backtest validation:")
        print(f"  - Is realistic: {validation_normal['is_realistic']}")
        print(f"  - Warnings: {len(validation_normal['warnings'])}")
        print(f"  - Errors: {len(validation_normal['errors'])}")

        # Test 2: Unrealistic backtest (high returns)
        print("\nTesting unrealistic backtest (high returns)...")
        # Create artificially high returns
        high_returns_prices = create_sample_price_data(n_samples=252, volatility=0.01, drift=0.01)  # 1% daily drift
        states_high = create_sample_state_sequence(n_samples=252, n_states=1, transition_probability=0.0)  # Single state
        config_high = BacktestConfig(state_map={0: 1})  # Always long

        result_high = backtest_with_analysis(states_high, high_returns_prices, config_high)
        validation_high = validate_backtest_realism(result_high)

        print(f"✓ High-return backtest validation:")
        print(f"  - Is realistic: {validation_high['is_realistic']}")
        print(f"  - Warnings: {len(validation_high['warnings'])}")
        if validation_high['warnings']:
            print(f"  - Sample warnings: {validation_high['warnings'][:2]}")

        # Test 3: Zero volatility backtest
        print("\nTesting zero volatility backtest...")
        flat_prices = pd.Series([100.0] * 252, index=pd.date_range('2020-01-01', periods=252, freq='D'))
        states_flat = create_sample_state_sequence(n_samples=252, n_states=1, transition_probability=0.0)
        result_flat = backtest_with_analysis(states_flat, flat_prices, BacktestConfig(state_map={0: 1}))
        validation_flat = validate_backtest_realism(result_flat)

        print(f"✓ Flat backtest validation:")
        print(f"  - Is realistic: {validation_flat['is_realistic']}")
        print(f"  - Warnings: {len(validation_flat['warnings'])}")

        # Test risk scores
        print("\nTesting risk score calculations...")
        for name, validation in [
            ("Normal", validation_normal),
            ("High-return", validation_high),
            ("Flat", validation_flat)
        ]:
            risk_scores = validation['risk_scores']
            print(f"  - {name} risk scores: trade_freq={risk_scores.get('trade_frequency', 0):.3f}, "
                  f"win_rate={risk_scores.get('win_rate', 0):.3f}")

        return True

    except Exception as e:
        print(f"✗ Realism validation test failed: {e}")
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
        # Test 1: Empty equity curve
        edge_case_total += 1
        print("Testing empty equity curve...")
        empty_equity = pd.Series([], dtype=float)
        try:
            metrics = calculate_performance(empty_equity)
            if metrics.total_return == 0.0:
                print("✓ Handled empty equity curve correctly")
                edge_case_passed += 1
            else:
                print("✗ Unexpected metrics for empty equity curve")
        except Exception as e:
            print(f"✗ Failed to handle empty equity curve: {e}")

        # Test 2: Single point equity curve
        edge_case_total += 1
        print("\nTesting single point equity curve...")
        single_equity = pd.Series([100000.0], index=[pd.Timestamp('2020-01-01')])
        try:
            metrics = calculate_performance(single_equity)
            if metrics.total_return == 0.0:
                print("✓ Handled single point equity curve correctly")
                edge_case_passed += 1
            else:
                print("✗ Unexpected metrics for single point equity curve")
        except Exception as e:
            print(f"✗ Failed to handle single point equity curve: {e}")

        # Test 3: Negative equity values
        edge_case_total += 1
        print("\nTesting negative equity values...")
        negative_equity = pd.Series([-1000, -2000, -1500], index=pd.date_range('2020-01-01', periods=3, freq='D'))
        try:
            metrics = calculate_performance(negative_equity)
            print("✓ Handled negative equity values")
            edge_case_passed += 1
        except Exception as e:
            print(f"✗ Failed to handle negative equity values: {e}")

        # Test 4: Invalid frequency
        edge_case_total += 1
        print("\nTesting invalid frequency handling...")
        try:
            # Create index with irregular frequency
            irregular_dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-15'), pd.Timestamp('2020-03-01')]
            irregular_equity = pd.Series([100, 105, 110], index=irregular_dates)
            freq = infer_trading_frequency(irregular_equity)
            print(f"✓ Inferred frequency for irregular data: {freq}")
            edge_case_passed += 1
        except Exception as e:
            print(f"✗ Failed to handle irregular frequency: {e}")

        # Test 5: Zero volatility returns
        edge_case_total += 1
        print("\nTesting zero volatility scenario...")
        flat_returns = pd.Series([100000] * 10, index=pd.date_range('2020-01-01', periods=10, freq='D'))
        try:
            metrics = calculate_performance(flat_returns)
            if metrics.annualized_volatility == 0.0:
                print("✓ Handled zero volatility correctly")
                edge_case_passed += 1
            else:
                print(f"✗ Unexpected volatility for flat returns: {metrics.annualized_volatility}")
        except Exception as e:
            print(f"✗ Failed to handle zero volatility: {e}")

        print(f"\nEdge case tests passed: {edge_case_passed}/{edge_case_total}")
        return edge_case_passed == edge_case_total

    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Task 7 tests."""
    print("Starting Task 7 Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("Core Performance Metrics", test_core_performance_metrics),
        ("Advanced Performance Metrics", test_advanced_performance_metrics),
        ("Bias Prevention", test_bias_prevention),
        ("Backtest Realism Validation", test_backtest_realism_validation),
        ("Edge Cases", test_edge_cases)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Task 7 Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Task 7 tests passed!")
        return True
    else:
        print("❌ Some Task 7 tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)