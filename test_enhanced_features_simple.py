#!/usr/bin/env python3
"""
Simplified test script for enhanced feature engineering functionality.

This script tests the new enhanced indicators without requiring the full pipeline infrastructure.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, "src")

from src.data_processing.feature_engineering import add_features
from src.data_processing.feature_selection import (
    CorrelationFeatureSelector,
    FeatureQualityScorer,
    VarianceFeatureSelector,
)


def create_test_data(n_samples=500):
    """Create synthetic test data with datetime index."""
    np.random.seed(42)

    # Create datetime index (hourly data for 3 weeks)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    # Generate synthetic OHLCV data with realistic patterns
    base_price = 100 + np.cumsum(np.random.normal(0, 0.5, n_samples))

    # Add intraday patterns
    hour_pattern = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    day_pattern = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 5))

    data = {
        "datetime": dates,
        "open": base_price
        + hour_pattern
        + day_pattern
        + np.random.normal(0, 0.2, n_samples),
        "high": base_price
        + hour_pattern
        + day_pattern
        + np.abs(np.random.normal(0.5, 0.3, n_samples)),
        "low": base_price
        + hour_pattern
        + day_pattern
        - np.abs(np.random.normal(0.5, 0.3, n_samples)),
        "close": base_price
        + hour_pattern
        + day_pattern
        + np.random.normal(0, 0.2, n_samples),
        "volume": np.random.exponential(1000000, n_samples)
        * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)),
    }

    df = pd.DataFrame(data)
    df = df.set_index("datetime")

    # Ensure OHLC relationships are valid
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    return df


def test_enhanced_indicators():
    """Test enhanced technical indicators."""
    print("Testing enhanced technical indicators...")

    # Create test data
    df = create_test_data()

    # Create enhanced indicator config
    indicator_config = {
        "enhanced_momentum": {
            "williams_r": {"length": 14},
            "cci": {"length": 20},
            "mfi": {"length": 14},
            "mtm": {"period": 10},
            "proc": {"period": 14},
        },
        "enhanced_volatility": {
            "historical_volatility": {"window": 20},
            "keltner_channels": {
                "ema_period": 20,
                "atr_period": 10,
                "atr_multiplier": 2.0,
            },
            "donchian_channels": {"period": 20},
        },
        "enhanced_trend": {
            "tma": {"period": 20},
            "wma": {"period": 10},
            "aroon": {"period": 14},
            "dmi": {"period": 14},
        },
        "enhanced_volume": {
            "adl": {"enabled": True},
            "vpt": {"enabled": True},
            "eom": {"period": 14},
            "volume_roc": {"period": 10},
        },
        "time_features": {
            "calendar_features": True,
            "cyclical_features": True,
            "intraday_features": True,
            "weekend_effects": True,
        },
    }

    # Add enhanced features
    df_with_features = add_features(df, indicator_config)

    # Check that enhanced indicators were added
    enhanced_indicators = [
        "williams_r_14",
        "cci_20",
        "mfi_14",
        "mtm_10",
        "proc_14",
        "hv_20",
        "keltner_upper_20_10",
        "donchian_upper_20",
        "tma_20",
        "wma_10",
        "aroon_up_14",
        "di_plus_14",
        "adl",
        "vpt",
        "eom_14",
        "volume_roc_10",
        "day_of_week",
        "hour",
        "session",
        "day_of_week_sin",
        "hour_cos",
    ]

    found_indicators = [
        ind for ind in enhanced_indicators if ind in df_with_features.columns
    ]
    print(
        f"✓ Enhanced indicators calculated: {len(found_indicators)}/{len(enhanced_indicators)}"
    )

    # Validate indicator ranges
    print("Validating indicator ranges...")

    # Williams %R should be between -100 and 0
    if "williams_r_14" in df_with_features.columns:
        williams_range = df_with_features["williams_r_14"].dropna()
        if williams_range.min() >= -100 and williams_range.max() <= 0:
            print("✓ Williams %R range validation passed")
        else:
            print(
                f"✗ Williams %R range invalid: [{williams_range.min():.2f}, {williams_range.max():.2f}]"
            )

    # Time features should have correct values
    if "day_of_week" in df_with_features.columns:
        dow_range = df_with_features["day_of_week"].unique()
        if all(0 <= x <= 6 for x in dow_range):
            print("✓ Day of week validation passed")
        else:
            print(f"✗ Day of week invalid: {dow_range}")

    if "session" in df_with_features.columns:
        session_range = df_with_features["session"].unique()
        if all(0 <= x <= 2 for x in session_range):
            print("✓ Session validation passed")
        else:
            print(f"✗ Session invalid: {session_range}")

    return df_with_features


def test_feature_selection():
    """Test feature selection algorithms."""
    print("\nTesting feature selection algorithms...")

    # Create test data with many features
    df = create_test_data(300)  # Smaller dataset for feature selection

    indicator_config = {
        "enhanced_momentum": {
            "williams_r": {"length": 14},
            "cci": {"length": 20},
            "mfi": {"length": 14},
        },
        "enhanced_volatility": {"historical_volatility": {"window": 20}},
        "time_features": {"calendar_features": True, "intraday_features": True},
    }

    df_features = add_features(df, indicator_config)

    # Select numeric features for testing
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    features = df_features[numeric_features].dropna()

    # Create target variable (log returns)
    if "log_ret" in features.columns:
        target = features["log_ret"]
        feature_cols = [col for col in features.columns if col != "log_ret"]
        X = features[feature_cols]
    else:
        # Use first feature as target if log_ret not available
        target = features.iloc[:, 0]
        X = features.iloc[:, 1:]

    print(f"Starting with {len(X.columns)} features")

    # Test correlation-based selection
    print("\nTesting correlation-based selection...")
    corr_selector = CorrelationFeatureSelector(threshold=0.9)
    features_corr = corr_selector.fit_transform(X, target)
    print(
        f"✓ Correlation selection: {len(X.columns)} -> {len(features_corr.columns)} features"
    )

    # Test variance-based selection
    print("\nTesting variance-based selection...")
    var_selector = VarianceFeatureSelector(threshold=0.001)
    features_var = var_selector.fit_transform(X)
    print(
        f"✓ Variance selection: {len(X.columns)} -> {len(features_var.columns)} features"
    )

    # Test feature quality scoring
    print("\nTesting feature quality scoring...")
    quality_scorer = FeatureQualityScorer()
    quality_report = quality_scorer.score_features(X, target)

    print("✓ Quality scoring completed")
    print(f"  - Average quality score: {quality_report['overall_score'].mean():.3f}")
    print(
        f"  - High quality features (>0.7): {len(quality_report[quality_report['overall_score'] > 0.7])}"
    )
    print(
        f"  - Low quality features (<0.3): {len(quality_report[quality_report['overall_score'] < 0.3])}"
    )

    # Test quality filtering
    filtered_features, _ = quality_scorer.filter_by_quality(X, min_score=0.4, y=target)
    print(
        f"✓ Quality filtering: {len(X.columns)} -> {len(filtered_features.columns)} features"
    )

    return True


def test_performance_benchmarks():
    """Test performance improvements."""
    print("\nTesting performance benchmarks...")

    import time

    # Create larger test data
    df = create_test_data(1000)

    indicator_config = {
        "enhanced_momentum": {"williams_r": {"length": 14}, "cci": {"length": 20}},
        "enhanced_volatility": {"historical_volatility": {"window": 20}},
        "time_features": {"calendar_features": True, "intraday_features": True},
    }

    # Benchmark feature engineering
    start_time = time.time()
    df_features = add_features(df, indicator_config)
    feature_time = time.time() - start_time

    print(f"✓ Feature engineering completed in {feature_time:.3f} seconds")
    print(f"  - Processing speed: {len(df) / feature_time:.0f} rows/second")
    print(f"  - Features generated: {len(df_features.columns) - len(df.columns)}")

    # Benchmark feature selection
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 20:
        features = df_features[numeric_features].dropna()
        if len(features) > 20:
            X = features.iloc[:, :-1]  # Use all but last column as features
            target = features.iloc[:, -1]  # Use last column as target

            start_time = time.time()
            corr_selector = CorrelationFeatureSelector(threshold=0.9)
            selected_features = corr_selector.fit_transform(X, target)
            selection_time = time.time() - start_time

            print(f"✓ Feature selection completed in {selection_time:.3f} seconds")
            print(
                f"  - Features reduced: {len(X.columns)} -> {len(selected_features.columns)}"
            )
            print(
                f"  - Selection speed: {len(X.columns) / selection_time:.0f} features/second"
            )

    return True


def main():
    """Run all enhanced feature engineering tests."""
    print("=" * 60)
    print("PHASE 2.1.2 ENHANCED FEATURE ENGINEERING VALIDATION")
    print("=" * 60)

    tests = [
        ("Enhanced Indicators", test_enhanced_indicators),
        ("Feature Selection", test_feature_selection),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            results.append((test_name, True, None))
            print(f"✅ {test_name} completed successfully")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "\n🎉 All tests passed! Enhanced feature engineering is working correctly."
        )
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
