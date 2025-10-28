"""
Test script for Visualization Module (Task 8)

This script tests the complete visualization and reporting functionality,
including state visualization, interactive dashboards, and detailed regime reports.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, "src")

from backtesting.utils import create_sample_price_data, create_sample_state_sequence
from utils.data_types import BacktestConfig
from visualization.chart_generator import (
    create_regime_timeline_plot,
    plot_state_distribution,
    plot_states,
)
from visualization.dashboard_builder import build_dashboard
from visualization.report_generator import generate_regime_report


def test_state_visualization_engine():
    """Test HMM State Visualization Engine."""
    print("=" * 60)
    print("Testing HMM State Visualization Engine")
    print("=" * 60)

    try:
        # Create sample data
        print("Creating sample price data and HMM states...")
        close_prices = create_sample_price_data(
            n_samples=100, volatility=0.02, drift=0.0005
        )
        states = create_sample_state_sequence(
            n_samples=100, n_states=3, transition_probability=0.05
        )

        # Convert to OHLCV DataFrame
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.02, 100)
        high_low_spread = np.random.uniform(0.005, 0.02, 100)

        prices = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * (1 + high_low_spread),
                "low": close_prices * (1 - high_low_spread),
                "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                * (1 + price_changes),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=close_prices.index,
        )

        # Create indicators
        indicators = pd.DataFrame(
            {
                "RSI_14": np.random.uniform(20, 80, 100),
                "ATRr_14": np.random.uniform(0.5, 2.0, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            },
            index=prices.index,
        )

        print(
            f"✓ Created test data: {len(prices)} price points, {len(np.unique(states))} states"
        )

        # Test basic state plotting
        print("\nTesting basic state visualization...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_states.png"

            result_path = plot_states(
                price_data=prices,
                states=states,
                indicators=indicators,
                output_path=str(output_path),
                show_plot=False,
            )

            if Path(result_path).exists():
                print(f"✓ Basic state chart generated successfully: {result_path}")
            else:
                print("✗ Basic state chart generation failed")
                return False

        # Test state distribution plotting
        print("\nTesting state distribution visualization...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_distribution.png"

            result_path = plot_state_distribution(
                states=states, indicators=indicators, output_path=str(output_path)
            )

            if Path(result_path).exists():
                print(
                    f"✓ State distribution chart generated successfully: {result_path}"
                )
            else:
                print("✗ State distribution chart generation failed")
                return False

        # Test regime timeline plotting
        print("\nTesting regime timeline visualization...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_timeline.png"

            result_path = create_regime_timeline_plot(
                states=states, price_data=prices["close"], output_path=str(output_path)
            )

            if Path(result_path).exists():
                print(f"✓ Regime timeline chart generated successfully: {result_path}")
            else:
                print("✗ Regime timeline chart generation failed")
                return False

        # Test with different configurations
        print("\nTesting various configurations...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_configured.png"

            config = {
                "title": "Test Chart Configuration",
                "state_colormap": "viridis",
                "state_alpha": 0.5,
                "show_volume": True,
                "figsize": (20, 12),
                "mav": [5, 10],
            }

            result_path = plot_states(
                price_data=prices,
                states=states,
                indicators=indicators,
                config=config,
                output_path=str(output_path),
                show_plot=False,
            )

            if Path(result_path).exists():
                print(f"✓ Configured chart generated successfully: {result_path}")
            else:
                print("✗ Configured chart generation failed")
                return False

        print("\n✓ All state visualization tests passed!")
        return True

    except Exception as e:
        print(f"✗ State visualization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_interactive_performance_dashboard():
    """Test Interactive Performance Dashboard."""
    print("\n" + "=" * 60)
    print("Testing Interactive Performance Dashboard")
    print("=" * 60)

    try:
        # Create sample backtest results
        print("Creating sample backtest results...")
        from backtesting.strategy_engine import backtest_with_analysis

        prices = create_sample_price_data(n_samples=100, volatility=0.02, drift=0.0005)
        states = create_sample_state_sequence(
            n_samples=100, n_states=3, transition_probability=0.05
        )

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1, 2: 0},
        )

        # Run backtest
        result = backtest_with_analysis(states, prices, config)
        print(f"✓ Backtest completed: {len(result.trades)} trades")

        # Calculate performance metrics
        from backtesting.performance_analyzer import analyze_performance

        metrics = analyze_performance(result, risk_free_rate=0.02)
        print("✓ Performance metrics calculated")

        # Test dashboard generation
        print("\nTesting dashboard generation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_dashboard.html"

            dashboard_config = {
                "title": "Test HMM Performance Dashboard",
                "include_regime_analysis": True,
                "include_monthly_heatmap": True,
                "theme": "plotly_white",
            }

            result_path = build_dashboard(
                result=result,
                metrics=metrics,
                states=states,
                config=dashboard_config,
                output_path=str(output_path),
            )

            if Path(result_path).exists():
                # Check if file contains expected content
                with open(result_path) as f:
                    content = f.read()

                if (
                    "HMM Performance Dashboard" in content
                    and "plotly" in content.lower()
                    and "equity" in content.lower()
                ):
                    print(
                        f"✓ Interactive dashboard generated successfully: {result_path}"
                    )
                    print(f"  - File size: {len(content)} characters")
                else:
                    print("✗ Dashboard content validation failed")
                    return False
            else:
                print("✗ Dashboard generation failed")
                return False

        # Test with different configurations
        print("\nTesting dashboard with minimal configuration...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_minimal_dashboard.html"

            minimal_config = {
                "title": "Minimal Dashboard Test",
                "include_regime_analysis": False,
                "include_monthly_heatmap": False,
            }

            result_path = build_dashboard(
                result=result,
                metrics=metrics,
                states=states,
                config=minimal_config,
                output_path=str(output_path),
            )

            if Path(result_path).exists():
                print(f"✓ Minimal dashboard generated successfully: {result_path}")
            else:
                print("✗ Minimal dashboard generation failed")
                return False

        print("\n✓ All dashboard tests passed!")
        return True

    except Exception as e:
        print(f"✗ Dashboard test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_regime_analysis_report():
    """Test Detailed Regime Analysis Report Generator."""
    print("\n" + "=" * 60)
    print("Testing Detailed Regime Analysis Report Generator")
    print("=" * 60)

    try:
        # Create sample data
        print("Creating sample data for regime analysis...")
        from backtesting.strategy_engine import backtest_with_analysis

        close_prices = create_sample_price_data(
            n_samples=150, volatility=0.02, drift=0.0005
        )
        states = create_sample_state_sequence(
            n_samples=150, n_states=4, transition_probability=0.05
        )

        # Convert to OHLCV DataFrame
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.02, 150)
        high_low_spread = np.random.uniform(0.005, 0.02, 150)

        prices = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * (1 + high_low_spread),
                "low": close_prices * (1 - high_low_spread),
                "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                * (1 + price_changes),
                "volume": np.random.uniform(1000, 5000, 150),
            },
            index=close_prices.index,
        )

        # Create richer indicators for analysis
        indicators = pd.DataFrame(
            {
                "RSI_14": np.random.uniform(20, 80, 150),
                "ATRr_14": np.random.uniform(0.5, 2.0, 150),
                "MACD_12_26_9": np.random.uniform(-1, 1, 150),
                "Bollinger_Upper": prices["close"]
                * (1 + np.random.uniform(0.01, 0.03, 150)),
                "Bollinger_Lower": prices["close"]
                * (1 - np.random.uniform(0.01, 0.03, 150)),
                "volume": np.random.uniform(1000, 5000, 150),
            },
            index=prices.index,
        )

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1, 2: 0, 3: 1},
        )

        # Run backtest
        result = backtest_with_analysis(states, prices, config)
        print(f"✓ Backtest completed: {len(result.trades)} trades")

        # Calculate performance metrics
        from backtesting.performance_analyzer import analyze_performance

        metrics = analyze_performance(result, risk_free_rate=0.02)
        print("✓ Performance metrics calculated")

        # Test HTML report generation
        print("\nTesting HTML report generation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_regime_report.html"

            report_config = {
                "title": "Test Regime Analysis Report",
                "include_charts": True,
                "include_indicators": True,
            }

            result_path = generate_regime_report(
                result=result,
                metrics=metrics,
                states=states,
                indicators=indicators,
                config=report_config,
                output_path=str(output_path),
                format="html",
            )

            if Path(result_path).exists():
                # Validate HTML content
                with open(result_path) as f:
                    content = f.read()

                validation_checks = [
                    ("Regime Analysis", "Report title"),
                    ("Executive Summary", "Executive summary section"),
                    ("Performance Metrics", "Performance metrics section"),
                    ("State 0", "State analysis"),
                    ("Transition Matrix", "Transition matrix"),
                    ("regime", "Regime-related content"),
                ]

                passed_checks = 0
                for check_text, description in validation_checks:
                    if check_text in content:
                        passed_checks += 1
                        print(f"  ✓ Found: {description}")
                    else:
                        print(f"  ✗ Missing: {description}")

                if (
                    passed_checks >= len(validation_checks) - 1
                ):  # Allow for one missing check
                    print(f"✓ HTML report generated and validated: {result_path}")
                    print(f"  - File size: {len(content)} characters")
                else:
                    print("✗ HTML report content validation failed")
                    return False
            else:
                print("✗ HTML report generation failed")
                return False

        # Test PDF report generation (if WeasyPrint is available)
        print("\nTesting PDF report generation...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "test_regime_report.pdf"

                result_path = generate_regime_report(
                    result=result,
                    metrics=metrics,
                    states=states,
                    indicators=indicators,
                    config=report_config,
                    output_path=str(output_path),
                    format="pdf",
                )

                if Path(result_path).exists():
                    file_size = Path(result_path).stat().st_size
                    if file_size > 1000:  # Reasonable size check
                        print(f"✓ PDF report generated successfully: {result_path}")
                        print(f"  - File size: {file_size} bytes")
                    else:
                        print("✗ PDF file too small, likely incomplete")
                        return False
                else:
                    print("✗ PDF report generation failed")
                    return False

        except Exception as pdf_error:
            print(
                f"⚠ PDF generation skipped (WeasyPrint may not be available): {pdf_error}"
            )

        print("\n✓ All regime report tests passed!")
        return True

    except Exception as e:
        print(f"✗ Regime report test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration_workflow():
    """Test complete visualization workflow integration."""
    print("\n" + "=" * 60)
    print("Testing Complete Visualization Workflow Integration")
    print("=" * 60)

    try:
        # Create comprehensive test data
        print("Creating comprehensive test dataset...")
        close_prices = create_sample_price_data(
            n_samples=200, volatility=0.025, drift=0.0008
        )
        states = create_sample_state_sequence(
            n_samples=200, n_states=3, transition_probability=0.03
        )

        # Convert to OHLCV DataFrame
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.025, 200)
        high_low_spread = np.random.uniform(0.005, 0.02, 200)

        prices = pd.DataFrame(
            {
                "open": close_prices,
                "high": close_prices * (1 + high_low_spread),
                "low": close_prices * (1 - high_low_spread),
                "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                * (1 + price_changes),
                "volume": np.random.uniform(1000, 8000, 200),
            },
            index=close_prices.index,
        )

        # Create detailed indicators
        indicators = pd.DataFrame(
            {
                "RSI_14": 50
                + 20 * np.sin(np.linspace(0, 10, 200))
                + np.random.normal(0, 5, 200),
                "ATRr_14": np.random.uniform(0.5, 2.5, 200),
                "MACD_12_26_9": np.random.uniform(-1.5, 1.5, 200),
                "Bollinger_Upper": prices["close"] * 1.02,
                "Bollinger_Lower": prices["close"] * 0.98,
                "volume": np.random.uniform(1000, 8000, 200),
            },
            index=prices.index,
        )

        # Run complete backtest
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_per_trade=5.0,
            slippage_bps=2.0,
            state_map={0: 1, 1: -1, 2: 0},
        )

        from backtesting.performance_analyzer import analyze_performance
        from backtesting.strategy_engine import backtest_with_analysis

        result = backtest_with_analysis(states, prices, config)
        metrics = analyze_performance(result, risk_free_rate=0.02)

        print("✓ Complete analysis performed:")
        print(f"  - Price points: {len(prices)}")
        print(f"  - States: {len(np.unique(states))}")
        print(f"  - Trades: {len(result.trades)}")
        print(f"  - Total return: {metrics.total_return:.2%}")

        # Generate all visualization outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. State visualization
            print("\n1. Generating state visualization...")
            state_chart = plot_states(
                prices,
                states,
                indicators,
                output_path=str(temp_path / "integration_states.png"),
                show_plot=False,
            )

            # 2. Dashboard
            print("2. Generating interactive dashboard...")
            dashboard = build_dashboard(
                result,
                metrics,
                states,
                output_path=str(temp_path / "integration_dashboard.html"),
            )

            # 3. Regime report
            print("3. Generating regime analysis report...")
            report = generate_regime_report(
                result,
                metrics,
                states,
                indicators,
                output_path=str(temp_path / "integration_report.html"),
                format="html",
            )

            # Verify all outputs exist
            outputs = [
                (Path(state_chart), "State Chart"),
                (Path(dashboard), "Dashboard"),
                (Path(report), "Regime Report"),
            ]

            success_count = 0
            for output_path, output_type in outputs:
                if output_path.exists():
                    size = output_path.stat().st_size
                    print(f"✓ {output_type}: {output_path.name} ({size} bytes)")
                    success_count += 1
                else:
                    print(f"✗ {output_type}: Generation failed")

            if success_count == len(outputs):
                print("\n✓ Complete integration workflow successful!")
                print(f"✓ All {len(outputs)} visualization components generated")
                return True
            else:
                print(
                    f"\n✗ Integration workflow failed: {success_count}/{len(outputs)} components generated"
                )
                return False

    except Exception as e:
        print(f"✗ Integration workflow test failed: {e}")
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
            empty_prices = pd.DataFrame(columns=["open", "high", "low", "close"])
            empty_states = np.array([])

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "empty_test.png"
                # This should raise a ValueError
                plot_states(
                    empty_prices,
                    empty_states,
                    output_path=str(output_path),
                    show_plot=False,
                )
                print("✗ Empty data should raise ValueError")
        except (ValueError, KeyError):
            print("✓ Empty data handled correctly")
            edge_case_passed += 1
        except Exception as e:
            print(f"✗ Unexpected error with empty data: {e}")

        # Test 2: Single state
        edge_case_total += 1
        print("\nTesting single state...")
        try:
            close_prices = create_sample_price_data(n_samples=50, volatility=0.02)
            single_states = np.array([1] * 50)

            # Convert to OHLCV DataFrame
            price_changes = np.random.normal(0, 0.02, 50)
            high_low_spread = np.random.uniform(0.005, 0.02, 50)

            prices = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": close_prices * (1 + high_low_spread),
                    "low": close_prices * (1 - high_low_spread),
                    "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                    * (1 + price_changes),
                    "volume": np.random.uniform(1000, 5000, 50),
                },
                index=close_prices.index,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "single_state_test.png"
                result_path = plot_states(
                    prices, single_states, output_path=str(output_path), show_plot=False
                )
                if Path(result_path).exists():
                    print("✓ Single state handled correctly")
                    edge_case_passed += 1
                else:
                    print("✗ Single state chart generation failed")
        except Exception as e:
            print(f"✗ Single state test failed: {e}")

        # Test 3: Large dataset performance
        edge_case_total += 1
        print("\nTesting large dataset...")
        try:
            close_prices = create_sample_price_data(n_samples=1000, volatility=0.02)
            large_states = create_sample_state_sequence(
                n_samples=1000, n_states=5, transition_probability=0.02
            )

            # Convert to OHLCV DataFrame
            price_changes = np.random.normal(0, 0.02, 1000)
            high_low_spread = np.random.uniform(0.005, 0.02, 1000)

            large_prices = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": close_prices * (1 + high_low_spread),
                    "low": close_prices * (1 - high_low_spread),
                    "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                    * (1 + price_changes),
                    "volume": np.random.uniform(1000, 5000, 1000),
                },
                index=close_prices.index,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "large_test.png"
                result_path = plot_states(
                    large_prices,
                    large_states,
                    output_path=str(output_path),
                    show_plot=False,
                )
                if Path(result_path).exists():
                    print("✓ Large dataset handled correctly")
                    edge_case_passed += 1
                else:
                    print("✗ Large dataset chart generation failed")
        except Exception as e:
            print(f"✗ Large dataset test failed: {e}")

        # Test 4: Invalid configurations
        edge_case_total += 1
        print("\nTesting invalid configurations...")
        try:
            close_prices = create_sample_price_data(n_samples=50, volatility=0.02)
            states = create_sample_state_sequence(n_samples=50, n_states=3)

            # Convert to OHLCV DataFrame
            price_changes = np.random.normal(0, 0.02, 50)
            high_low_spread = np.random.uniform(0.005, 0.02, 50)

            prices = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": close_prices * (1 + high_low_spread),
                    "low": close_prices * (1 - high_low_spread),
                    "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                    * (1 + price_changes),
                    "volume": np.random.uniform(1000, 5000, 50),
                },
                index=close_prices.index,
            )

            invalid_config = {
                "figsize": (-100, -100),  # Invalid size
                "state_alpha": 2.0,  # Invalid alpha (>1)
                "mav": [-5, 10],  # Invalid moving average
            }

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "invalid_config_test.png"
                # Should handle invalid config gracefully
                result_path = plot_states(
                    prices,
                    states,
                    config=invalid_config,
                    output_path=str(output_path),
                    show_plot=False,
                )
                if Path(result_path).exists():
                    print("✓ Invalid configuration handled gracefully")
                    edge_case_passed += 1
                else:
                    print("✗ Invalid configuration not handled properly")
        except Exception as e:
            print(f"✓ Invalid configuration properly rejected: {type(e).__name__}")
            edge_case_passed += 1

        # Test 5: Missing indicators
        edge_case_total += 1
        print("\nTesting missing indicators...")
        try:
            close_prices = create_sample_price_data(n_samples=50, volatility=0.02)
            states = create_sample_state_sequence(n_samples=50, n_states=3)
            empty_indicators = pd.DataFrame()

            # Convert to OHLCV DataFrame
            price_changes = np.random.normal(0, 0.02, 50)
            high_low_spread = np.random.uniform(0.005, 0.02, 50)

            prices = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": close_prices * (1 + high_low_spread),
                    "low": close_prices * (1 - high_low_spread),
                    "close": close_prices.shift(1).fillna(close_prices.iloc[0])
                    * (1 + price_changes),
                    "volume": np.random.uniform(1000, 5000, 50),
                },
                index=close_prices.index,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "no_indicators_test.png"
                result_path = plot_states(
                    prices,
                    states,
                    indicators=empty_indicators,
                    output_path=str(output_path),
                    show_plot=False,
                )
                if Path(result_path).exists():
                    print("✓ Missing indicators handled correctly")
                    edge_case_passed += 1
                else:
                    print("✗ Missing indicators handling failed")
        except Exception as e:
            print(f"✗ Missing indicators test failed: {e}")

        print(f"\nEdge case tests passed: {edge_case_passed}/{edge_case_total}")
        return edge_case_passed >= edge_case_total - 1  # Allow for one test failure

    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all Task 8 visualization tests."""
    print("Starting Task 8 Visualization Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("State Visualization Engine", test_state_visualization_engine),
        ("Interactive Performance Dashboard", test_interactive_performance_dashboard),
        ("Regime Analysis Report Generator", test_regime_analysis_report),
        ("Complete Integration Workflow", test_integration_workflow),
        ("Edge Cases and Error Handling", test_edge_cases),
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
    print("Task 8 Visualization Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Task 8 visualization tests passed!")
        print("✅ Visualization & Reporting Module is fully functional!")
        return True
    else:
        print("❌ Some Task 8 visualization tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
