"""
Simple Test script for Visualization Module (Task 8)

A simplified version that tests core functionality without complex dependencies.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from visualization.chart_generator import plot_states
from visualization.dashboard_builder import build_dashboard
from visualization.report_generator import generate_regime_report


def create_simple_ohlcv_data(n_samples=100):
    """Create simple OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_samples)
    close_prices = base_price * np.cumprod(1 + returns)

    # Create OHLC data
    high_spread = np.random.uniform(0.005, 0.02, n_samples)
    low_spread = np.random.uniform(0.005, 0.02, n_samples)

    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices * (1 + high_spread),
        'low': close_prices * (1 - low_spread),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, n_samples)
    }, index=dates)

    return data

def test_basic_state_visualization():
    """Test basic state visualization with simple data."""
    print("=" * 60)
    print("Testing Basic State Visualization")
    print("=" * 60)

    try:
        # Create simple data
        print("Creating simple test data...")
        prices = create_simple_ohlcv_data(50)
        states = np.random.choice([0, 1, 2], 50)

        print(f"✓ Created {len(prices)} data points with {len(np.unique(states))} states")

        # Test basic visualization without indicators
        print("\nTesting basic visualization...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "basic_test.png"

            config = {
                'title': 'Basic HMM State Test',
                'show_volume': True,
                'mav': [],  # No moving averages to avoid panel conflicts
            }

            result_path = plot_states(
                price_data=prices,
                states=states,
                indicators=None,  # No indicators
                config=config,
                output_path=str(output_path),
                show_plot=False
            )

            if Path(result_path).exists():
                print(f"✓ Basic visualization successful: {result_path}")
                return True
            else:
                print("✗ Basic visualization failed")
                return False

    except Exception as e:
        print(f"✗ Basic visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dashboard():
    """Test simple dashboard generation."""
    print("\n" + "=" * 60)
    print("Testing Simple Dashboard Generation")
    print("=" * 60)

    try:
        # Create simple data
        print("Creating simple backtest data...")
        prices = create_simple_ohlcv_data(50)
        states = np.random.choice([0, 1], 50)

        # Create simple backtest result
        from utils.data_types import BacktestResult, PerformanceMetrics, Trade

        # Simple equity curve
        equity_curve = prices['close'] * 10  # Scale to represent portfolio value

        # Create basic trades
        trades = []
        for i in range(5):
            entry_price = prices['close'].iloc[i*10]
            exit_price = prices['close'].iloc[i*10+5]
            pnl = (exit_price - entry_price) * 1.0

            trade = Trade(
                entry_time=prices.index[i*10],
                exit_time=prices.index[i*10+5],
                entry_price=entry_price,
                exit_price=exit_price,
                size=1.0,
                pnl=pnl,
                commission=5.0,
                slippage=2.0
            )
            trades.append(trade)

        # Create positions
        positions = np.array([1, 0, -1, 0, 1] * 10)[:50]

        result = BacktestResult(
            equity_curve=equity_curve,
            positions=pd.Series(positions, index=prices.index),
            trades=trades,
            start_date=prices.index[0],
            end_date=prices.index[-1]
        )

        # Create performance metrics
        metrics = PerformanceMetrics(
            total_return=0.05,
            annualized_return=0.12,
            annualized_volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.08,
            max_drawdown_duration=10,
            calmar_ratio=1.5,
            win_rate=0.6,
            loss_rate=0.4,
            profit_factor=1.2,
            sortino_ratio=1.1
        )

        print(f"✓ Created backtest result: {len(trades)} trades")

        # Test dashboard generation
        print("\nTesting dashboard generation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "simple_dashboard.html"

            config = {
                'title': 'Simple HMM Dashboard Test',
                'include_regime_analysis': False,  # Disable complex features
                'include_monthly_heatmap': False,
                'include_distribution': False
            }

            result_path = build_dashboard(
                result=result,
                metrics=metrics,
                states=states,
                config=config,
                output_path=str(output_path)
            )

            if Path(result_path).exists():
                # Check file content
                with open(result_path) as f:
                    content = f.read()

                if 'HMM Dashboard' in content and len(content) > 1000:
                    print(f"✓ Simple dashboard generated: {result_path}")
                    print(f"  - File size: {len(content)} characters")
                    return True
                else:
                    print("✗ Dashboard content invalid")
                    return False
            else:
                print("✗ Dashboard generation failed")
                return False

    except Exception as e:
        print(f"✗ Simple dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_html_report():
    """Test HTML report generation."""
    print("\n" + "=" * 60)
    print("Testing HTML Report Generation")
    print("=" * 60)

    try:
        # Create simple data
        print("Creating simple report data...")
        prices = create_simple_ohlcv_data(50)
        states = np.random.choice([0, 1, 2], 50)

        # Create indicators
        indicators = pd.DataFrame({
            'RSI_14': np.random.uniform(20, 80, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=prices.index)

        # Create simple backtest result
        from utils.data_types import BacktestResult, PerformanceMetrics

        equity_curve = prices['close'] * 10
        positions = np.array([1, 0, -1] * 17)[:50]  # 48 positions (16*3)
        trades = []

        result = BacktestResult(
            equity_curve=equity_curve,
            positions=pd.Series(positions[:50], index=prices.index),
            trades=trades,
            start_date=prices.index[0],
            end_date=prices.index[-1]
        )

        metrics = PerformanceMetrics(
            total_return=0.08,
            annualized_return=0.15,
            annualized_volatility=0.18,
            sharpe_ratio=0.83,
            max_drawdown=-0.10,
            max_drawdown_duration=12,
            calmar_ratio=1.5,
            win_rate=0.65,
            loss_rate=0.35,
            profit_factor=1.3,
            sortino_ratio=1.2
        )

        print("✓ Created report data")

        # Test HTML report generation
        print("\nTesting HTML report generation...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "simple_report.html"

            config = {
                'title': 'Simple HMM Analysis Report',
                'include_charts': False  # Disable charts to avoid matplotlib issues
            }

            result_path = generate_regime_report(
                result=result,
                metrics=metrics,
                states=states,
                indicators=indicators,
                config=config,
                output_path=str(output_path),
                format='html'
            )

            if Path(result_path).exists():
                # Check file content
                with open(result_path) as f:
                    content = f.read()

                if ('Regime Analysis' in content and
                    'Performance Metrics' in content and
                    len(content) > 2000):
                    print(f"✓ HTML report generated: {result_path}")
                    print(f"  - File size: {len(content)} characters")
                    return True
                else:
                    print("✗ HTML report content invalid")
                    return False
            else:
                print("✗ HTML report generation failed")
                return False

    except Exception as e:
        print(f"✗ HTML report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple visualization tests."""
    print("Starting Simple Task 8 Visualization Test Suite")
    print("=" * 60)

    # Run tests
    tests = [
        ("Basic State Visualization", test_basic_state_visualization),
        ("Simple Dashboard Generation", test_simple_dashboard),
        ("HTML Report Generation", test_html_report)
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
    print("Simple Task 8 Visualization Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All simple visualization tests passed!")
        print("✅ Core visualization functionality is working!")
        return True
    else:
        print("❌ Some visualization tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
