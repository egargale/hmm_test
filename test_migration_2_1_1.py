#!/usr/bin/env python3
"""
Test script for Phase 2.1.1 migration validation.

This script tests that the migrated main.py functionality
produces equivalent results to the original implementation.
"""

import asyncio
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, "src")

from src.compatibility.main_adapter import HMMPipeline
from src.pipelines.pipeline_types import (
    FeatureConfig,
    PipelineConfig,
    StreamingConfig,
    TrainingConfig,
)


def create_test_data(n_samples=1000):
    """Create synthetic test data for validation"""
    np.random.seed(42)

    # Generate synthetic OHLCV data
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="H")

    # Base price with trend and noise
    base_price = 100 + np.cumsum(np.random.normal(0, 1, n_samples))

    # Generate OHLCV data
    data = {
        "DateTime": dates,
        "Open": base_price + np.random.normal(0, 0.5, n_samples),
        "High": base_price + np.abs(np.random.normal(1, 0.5, n_samples)),
        "Low": base_price - np.abs(np.random.normal(1, 0.5, n_samples)),
        "Close": base_price + np.random.normal(0, 0.5, n_samples),
        "Volume": np.random.exponential(1000000, n_samples),
    }

    df = pd.DataFrame(data)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = np.maximum(df["High"], np.maximum(df["Open"], df["Close"]))
    df["Low"] = np.minimum(df["Low"], np.minimum(df["Open"], df["Close"]))

    return df


async def test_backward_compatibility():
    """Test that the legacy interface produces expected results"""
    print("Testing backward compatibility...")

    # Create test data
    test_df = create_test_data(500)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_df.to_csv(f.name, index=False)
        test_csv_path = f.name

    try:
        # Test legacy argument parsing
        from src.compatibility.main_adapter import create_legacy_parser

        parser = create_legacy_parser()
        test_args = [
            test_csv_path,
            "--n_states",
            "3",
            "--max_iter",
            "50",  # Reduced for test speed
            "--backtest",
            "--plot",
        ]

        args = parser.parse_args(test_args)

        # Validate argument conversion
        assert args.n_states == 3
        assert args.max_iter == 50
        assert args.backtest
        assert args.plot

        print("✓ Legacy argument parsing works correctly")

        # Test pipeline creation from args
        pipeline = HMMPipeline.from_args(args)

        assert pipeline.config.training.n_states == 3
        assert pipeline.config.training.n_iter == 50
        assert pipeline.config.backtesting is not None

        print("✓ Pipeline creation from legacy args works correctly")

        # Test pipeline execution
        result = await pipeline.run(Path(test_csv_path))

        assert result.is_successful()
        assert result.processed_data is not None
        assert result.states is not None
        assert result.model is not None
        assert len(result.processed_data) > 0

        print("✓ Pipeline execution completed successfully")

        # Validate results
        unique_states = np.unique(result.states)
        assert len(unique_states) <= 3
        assert all(0 <= s < 3 for s in unique_states)

        print("✓ Results validation passed")

        return True

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

    finally:
        # Cleanup
        Path(test_csv_path).unlink(missing_ok=True)


def test_legacy_function_wrappers():
    """Test that legacy function wrappers work correctly"""
    print("Testing legacy function wrappers...")

    try:
        # Create test data
        test_df = create_test_data(100)

        # Test legacy add_features function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.compatibility.main_adapter import add_features

            featured_df = add_features(test_df)

            # Check that features were added
            expected_features = [
                "log_ret",
                "atr",
                "roc",
                "rsi",
                "bb_width",
                "bb_position",
                "adx",
                "stoch",
                "sma_5_ratio",
                "hl_ratio",
                "volume_ratio",
            ]

            for feature in expected_features:
                assert feature in featured_df.columns, f"Missing feature: {feature}"

            # Check that NaN rows were dropped
            assert len(featured_df) < len(test_df)
            assert not featured_df.isnull().any().any()

        print("✓ Legacy add_features wrapper works correctly")

        # Test legacy simple_backtest function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.compatibility.main_adapter import simple_backtest

            # Create dummy states
            states = np.random.choice([0, 1, 2], size=len(featured_df))

            backtest_result = simple_backtest(featured_df, states)

            assert isinstance(backtest_result, pd.Series)
            assert len(backtest_result) > 0

        print("✓ Legacy simple_backtest wrapper works correctly")

        # Test legacy perf_metrics function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.compatibility.main_adapter import perf_metrics

            sharpe, max_dd = perf_metrics(backtest_result)

            assert isinstance(sharpe, (int, float))
            assert isinstance(max_dd, (int, float))

        print("✓ Legacy perf_metrics wrapper works correctly")

        return True

    except Exception as e:
        print(f"✗ Legacy function wrapper test failed: {e}")
        return False


def test_configuration_mapping():
    """Test that configuration mapping works correctly"""
    print("Testing configuration mapping...")

    try:
        # Test PipelineConfig creation
        config = PipelineConfig(
            features=FeatureConfig(
                enable_atr=True, enable_rsi=True, atr_window=5, rsi_window=10
            ),
            training=TrainingConfig(n_states=4, n_iter=200, random_state=123),
            streaming=StreamingConfig(chunk_size=50000, show_progress=False),
        )

        # Test to_dict conversion
        config_dict = config.to_dict()
        assert "features" in config_dict
        assert "training" in config_dict
        assert "streaming" in config_dict

        # Test from_dict conversion
        recreated_config = PipelineConfig.from_dict(config_dict)
        assert recreated_config.features.enable_atr == config.features.enable_atr
        assert recreated_config.training.n_states == config.training.n_states
        assert recreated_config.streaming.chunk_size == config.streaming.chunk_size

        print("✓ Configuration mapping works correctly")

        return True

    except Exception as e:
        print(f"✗ Configuration mapping test failed: {e}")
        return False


async def main():
    """Run all migration tests"""
    print("=" * 60)
    print("PHASE 2.1.1 MIGRATION VALIDATION")
    print("=" * 60)

    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("Legacy Function Wrappers", test_legacy_function_wrappers),
        ("Configuration Mapping", test_configuration_mapping),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")

        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()

        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Migration Phase 2.1.1 is successful.")
        return 0
    else:
        print(
            f"\n❌ {total - passed} test(s) failed. Please fix issues before proceeding."
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
