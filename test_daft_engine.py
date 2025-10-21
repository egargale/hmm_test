"""
Test script for Daft Processing Engine

This script tests the Daft engine implementation and validates its functionality
for processing financial OHLCV data with Arrow-native operations.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from utils import ProcessingConfig
from processing_engines import process_daft
from processing_engines.daft_engine import compute_daft_with_progress
from processing_engines.factory import get_processing_engine_factory

def create_test_data():
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate synthetic price data
    n_points = 1000
    dates = pd.date_range('2020-01-01', periods=n_points, freq='1h')

    # Generate realistic price movements
    price = 100.0
    prices = [price]

    for i in range(1, n_points):
        change = np.random.normal(0, 0.02)  # 2% volatility
        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    # Create OHLCV data
    data = {
        'datetime': dates,
        'open': prices,
        'high': prices * np.random.uniform(1.0, 1.05, n_points),
        'low': prices * np.random.uniform(0.95, 1.0, n_points),
        'close': prices * np.random.uniform(0.98, 1.02, n_points),
        'volume': np.random.uniform(1000, 10000, n_points)
    }

    df = pd.DataFrame(data)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))

    return df

def test_daft_basic_functionality():
    """Test basic Daft engine functionality."""
    print("=" * 60)
    print("Testing Daft Engine Basic Functionality")
    print("=" * 60)

    try:
        # Check Daft availability
        try:
            import daft
            print(f"✓ Daft version {daft.__version__} is available")
        except ImportError:
            print("✗ Daft is not available")
            return False

        # Create test data
        print("Creating test data...")
        test_df = create_test_data()
        csv_path = "test_daft_data.csv"
        test_df.to_csv(csv_path, index=False)
        print(f"✓ Created test CSV with {len(test_df)} rows")

        # Create processing configuration
        config = ProcessingConfig(
            engine_type="daft",
            chunk_size=1000
        )
        print(f"✓ Created processing configuration")

        # Test Daft processing
        print("\nTesting Daft processing...")
        processed_df = process_daft(
            csv_path=csv_path,
            config=config,
            npartitions=2,
            show_progress=True
        )
        print(f"✓ Daft processing completed")

        # Test computation
        print("\nTesting Daft computation...")
        result_df = compute_daft_with_progress(processed_df, show_progress=True)
        print(f"✓ Daft computation completed: {len(result_df)} rows, {len(result_df.columns)} columns")

        # Validate results
        print("\nValidating results...")
        print(f"  - Original columns: {len(test_df.columns)}")
        print(f"  - Processed columns: {len(result_df.columns)}")
        print(f"  - Row count maintained: {len(test_df) == len(result_df)}")

        # Check for technical indicators
        feature_cols = [col for col in result_df.columns if any(indicator in col.lower()
                       for indicator in ['sma', 'ema', 'rsi', 'macd', 'bollinger'])]
        print(f"  - Technical indicators added: {len(feature_cols)}")

        if feature_cols:
            print(f"  - Sample indicators: {list(feature_cols[:3])}")

        # Test memory usage
        memory_mb = result_df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  - Memory usage: {memory_mb:.2f} MB")

        # Cleanup
        os.remove(csv_path)
        print(f"✓ Test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_daft_factory_integration():
    """Test Daft engine integration with ProcessingEngineFactory."""
    print("\n" + "=" * 60)
    print("Testing Daft Engine Factory Integration")
    print("=" * 60)

    try:
        # Create test data
        print("Creating test data...")
        test_df = create_test_data()
        csv_path = "test_daft_factory.csv"
        test_df.to_csv(csv_path, index=False)

        # Get factory instance
        factory = get_processing_engine_factory()
        print("✓ Retrieved processing engine factory")

        # Check available engines
        available_engines = factory.get_available_engines()
        print(f"✓ Available engines: {available_engines}")

        if 'daft' not in available_engines:
            print("✗ Daft engine not available in factory")
            return False

        # Test engine recommendation
        recommended = factory.recommend_engine(csv_path)
        print(f"✓ Recommended engine: {recommended}")

        # Test Daft processing through factory
        config = ProcessingConfig(
            engine_type="daft",
            chunk_size=1000
        )

        print("\nTesting Daft processing through factory...")
        result = factory.process_with_engine(
            csv_path=csv_path,
            config=config,
            engine="daft",
            compute_result=True,
            show_progress=True
        )

        print(f"✓ Factory Daft processing completed: {len(result)} rows, {len(result.columns)} columns")

        # Test engine info
        engine_info = factory.get_engine_info()
        daft_info = engine_info['engine_details'].get('daft', {})
        print(f"✓ Daft engine info: {daft_info.get('type', 'unknown')}")

        # Cleanup
        os.remove(csv_path)
        print(f"✓ Factory integration test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Factory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_daft_benchmark():
    """Test Daft engine benchmarking functionality."""
    print("\n" + "=" * 60)
    print("Testing Daft Engine Benchmarking")
    print("=" * 60)

    try:
        # Create smaller test data for benchmarking
        print("Creating benchmark data...")
        test_df = create_test_data().iloc[:500]  # Use smaller dataset
        csv_path = "test_daft_benchmark.csv"
        test_df.to_csv(csv_path, index=False)

        # Import benchmark function
        from processing_engines.daft_engine import benchmark_daft_engine

        print("Running Daft benchmark...")
        results = benchmark_daft_engine(
            csv_path=csv_path,
            partition_counts=[1, 2],
            use_accelerators=False
        )

        print("✓ Benchmark completed")

        # Analyze results
        for key, result in results.items():
            if result.get('success'):
                print(f"  - {key}: {result['time']:.2f}s, {result['rows']} rows, {result['memory_mb']:.2f}MB")
            else:
                print(f"  - {key}: FAILED - {result.get('error', 'unknown error')}")

        # Cleanup
        os.remove(csv_path)
        print(f"✓ Benchmark test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Daft engine tests."""
    print("Starting Daft Engine Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("Basic Functionality", test_daft_basic_functionality),
        ("Factory Integration", test_daft_factory_integration),
        ("Benchmark", test_daft_benchmark)
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
    print("Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All Daft engine tests passed!")
        return True
    else:
        print("❌ Some Daft engine tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)