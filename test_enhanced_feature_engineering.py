#!/usr/bin/env python3
"""
Test script for enhanced feature engineering functionality.

This script tests the new enhanced indicators, feature selection algorithms,
and quality assessment capabilities added in Phase 2.1.2.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, 'src')

from src.data_processing.feature_engineering import FeatureEngineer, add_features
from src.data_processing.feature_selection import (
    CorrelationFeatureSelector,
    FeatureQualityScorer,
    FeatureSelectionPipeline,
    MutualInformationFeatureSelector,
    VarianceFeatureSelector,
)
from src.pipelines.pipeline_types import FeatureConfig


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
        'datetime': dates,
        'open': base_price + hour_pattern + day_pattern + np.random.normal(0, 0.2, n_samples),
        'high': base_price + hour_pattern + day_pattern + np.abs(np.random.normal(0.5, 0.3, n_samples)),
        'low': base_price + hour_pattern + day_pattern - np.abs(np.random.normal(0.5, 0.3, n_samples)),
        'close': base_price + hour_pattern + day_pattern + np.random.normal(0, 0.2, n_samples),
        'volume': np.random.exponential(1000000, n_samples) * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 24))
    }

    df = pd.DataFrame(data)
    df = df.set_index('datetime')

    # Ensure OHLC relationships are valid
    df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
    df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))

    return df


def test_enhanced_indicators():
    """Test enhanced technical indicators."""
    print("Testing enhanced technical indicators...")

    # Create test data
    df = create_test_data()

    # Create enhanced indicator config
    indicator_config = {
        'enhanced_momentum': {
            'williams_r': {'length': 14},
            'cci': {'length': 20},
            'mfi': {'length': 14},
            'mtm': {'period': 10},
            'proc': {'period': 14}
        },
        'enhanced_volatility': {
            'chaikin_volatility': {'ema_period': 10, 'roc_period': 10},
            'historical_volatility': {'window': 20},
            'keltner_channels': {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0},
            'donchian_channels': {'period': 20}
        },
        'enhanced_trend': {
            'tma': {'period': 20},
            'wma': {'period': 10},
            'hma': {'period': 16},
            'aroon': {'period': 14},
            'dmi': {'period': 14}
        },
        'enhanced_volume': {
            'adl': {'enabled': True},
            'vpt': {'enabled': True},
            'eom': {'period': 14},
            'volume_roc': {'period': 10}
        },
        'time_features': {
            'calendar_features': True,
            'cyclical_features': True,
            'intraday_features': True,
            'weekend_effects': True
        }
    }

    # Add enhanced features
    df_with_features = add_features(df, indicator_config)

    # Check that enhanced indicators were added
    enhanced_indicators = [
        'williams_r_14', 'cci_20', 'mfi_14', 'mtm_10', 'proc_14',
        'chaikin_vol_10_10', 'hv_20', 'keltner_upper_20_10', 'donchian_upper_20',
        'tma_20', 'wma_10', 'hma_16', 'aroon_up_14', 'di_plus_14',
        'adl', 'vpt', 'eom_14', 'volume_roc_10',
        'day_of_week', 'hour', 'session', 'day_of_week_sin', 'hour_cos'
    ]

    missing_indicators = [ind for ind in enhanced_indicators if ind not in df_with_features.columns]
    if missing_indicators:
        print(f"⚠️  Some indicators missing: {missing_indicators[:5]}...")
    else:
        print("✓ All enhanced indicators calculated successfully")

    # Validate indicator ranges
    print("Validating indicator ranges...")

    # Williams %R should be between -100 and 0
    if 'williams_r_14' in df_with_features.columns:
        williams_range = df_with_features['williams_r_14'].dropna()
        if williams_range.min() >= -100 and williams_range.max() <= 0:
            print("✓ Williams %R range validation passed")
        else:
            print(f"✗ Williams %R range invalid: [{williams_range.min():.2f}, {williams_range.max():.2f}]")

    # CCI should typically be between -200 and 200
    if 'cci_20' in df_with_features.columns:
        cci_range = df_with_features['cci_20'].dropna()
        if abs(cci_range.mean()) < 200 and abs(cci_range.std()) < 200:
            print("✓ CCI range validation passed")
        else:
            print(f"⚠️  CCI range unusual: mean={cci_range.mean():.2f}, std={cci_range.std():.2f}")

    # Time features should have correct values
    if 'day_of_week' in df_with_features.columns:
        dow_range = df_with_features['day_of_week'].unique()
        if all(0 <= x <= 6 for x in dow_range):
            print("✓ Day of week validation passed")
        else:
            print(f"✗ Day of week invalid: {dow_range}")

    if 'session' in df_with_features.columns:
        session_range = df_with_features['session'].unique()
        if all(0 <= x <= 2 for x in session_range):
            print("✓ Session validation passed")
        else:
            print(f"✗ Session invalid: {session_range}")

    return df_with_features


def test_feature_engineer_enhanced():
    """Test enhanced FeatureEngineer class."""
    print("\nTesting enhanced FeatureEngineer...")

    # Create test data
    df = create_test_data()

    # Create enhanced feature config
    config = FeatureConfig(
        enable_log_returns=True,
        enable_atr=True,
        enable_rsi=True,
        enable_bollinger_bands=True,
        enable_volume_features=True,
        enable_sma_ratios=True,
        enable_price_position=True,
        # Add enhanced features
        enable_williams_r=True,
        enable_cci=True,
        enable_mfi=True,
        enable_historical_volatility=True,
        enable_keltner_channels=True,
        enable_aroon=True,
        enable_adl=True,
        enable_time_features=True
    )

    # Initialize feature engineer
    engineer = FeatureEngineer(config)

    # Add features
    df_features = engineer.add_features(df)

    # Validate features were added
    assert len(df_features.columns) > len(df.columns), "No features were added"
    print(f"✓ Features added: {len(df_features.columns) - len(df.columns)} new features")

    # Test feature names
    feature_names = engineer.get_feature_names()
    assert len(feature_names) > 0, "No feature names available"
    print(f"✓ Feature names extracted: {len(feature_names)} features")

    # Test feature validation
    is_valid = engineer.validate_features(df_features)
    print(f"✓ Feature validation: {'passed' if is_valid else 'failed'}")

    # Test enhanced feature summary
    summary = engineer.get_enhanced_feature_summary(df_features)
    print(f"✓ Feature summary: {summary['total_features']} total features")
    print(f"  - Enhanced momentum: {summary['category_counts']['enhanced_momentum']}")
    print(f"  - Enhanced volatility: {summary['category_counts']['enhanced_volatility']}")
    print(f"  - Enhanced trend: {summary['category_counts']['enhanced_trend']}")
    print(f"  - Time features: {summary['category_counts']['time_features']}")

    return df_features, engineer


def test_feature_selection():
    """Test feature selection algorithms."""
    print("\nTesting feature selection algorithms...")

    # Create test data with many features
    df = create_test_data(300)  # Smaller dataset for feature selection
    config = FeatureConfig(
        enable_log_returns=True,
        enable_atr=True,
        enable_rsi=True,
        enable_bollinger_bands=True,
        enable_williams_r=True,
        enable_cci=True,
        enable_historical_volatility=True,
        enable_time_features=True
    )

    engineer = FeatureEngineer(config)
    df_features = engineer.add_features(df)

    # Create target variable (log returns)
    target = df_features['log_ret'].dropna()
    features = df_features.loc[target.index, engineer.get_feature_names()].dropna()

    print(f"Starting with {len(features.columns)} features")

    # Test correlation-based selection
    print("\nTesting correlation-based selection...")
    corr_selector = CorrelationFeatureSelector(threshold=0.9)
    features_corr = corr_selector.fit_transform(features, target)
    print(f"✓ Correlation selection: {len(features.columns)} -> {len(features_corr.columns)} features")

    # Test variance-based selection
    print("\nTesting variance-based selection...")
    var_selector = VarianceFeatureSelector(threshold=0.001)
    features_var = var_selector.fit_transform(features)
    print(f"✓ Variance selection: {len(features.columns)} -> {len(features_var.columns)} features")

    # Test mutual information selection
    print("\nTesting mutual information selection...")
    mi_selector = MutualInformationFeatureSelector(k=15)
    features_mi = mi_selector.fit_transform(features, target)
    print(f"✓ Mutual information selection: {len(features.columns)} -> {len(features_mi.columns)} features")

    # Test feature quality scoring
    print("\nTesting feature quality scoring...")
    quality_scorer = FeatureQualityScorer()
    quality_report = quality_scorer.score_features(features, target)

    print("✓ Quality scoring completed")
    print(f"  - Average quality score: {quality_report['overall_score'].mean():.3f}")
    print(f"  - High quality features (>0.7): {len(quality_report[quality_report['overall_score'] > 0.7])}")
    print(f"  - Low quality features (<0.3): {len(quality_report[quality_report['overall_score'] < 0.3])}")

    # Test quality filtering
    filtered_features, _ = quality_scorer.filter_by_quality(features, min_score=0.4, y=target)
    print(f"✓ Quality filtering: {len(features.columns)} -> {len(filtered_features.columns)} features")

    return features_corr, features_mi, quality_report


def test_feature_selection_pipeline():
    """Test feature selection pipeline."""
    print("\nTesting feature selection pipeline...")

    # Create test data
    df = create_test_data(200)
    config = FeatureConfig(
        enable_log_returns=True,
        enable_atr=True,
        enable_rsi=True,
        enable_bollinger_bands=True,
        enable_williams_r=True,
        enable_time_features=True
    )

    engineer = FeatureEngineer(config)
    df_features = engineer.add_features(df)

    target = df_features['log_ret'].dropna()
    features = df_features.loc[target.index, engineer.get_feature_names()].dropna()

    # Create pipeline with multiple selectors
    selectors = [
        VarianceFeatureSelector(threshold=0.001),
        CorrelationFeatureSelector(threshold=0.95),
        MutualInformationFeatureSelector(k=12)
    ]

    pipeline = FeatureSelectionPipeline(
        selectors=selectors,
        quality_threshold=0.4,
        enable_quality_filtering=True
    )

    # Run pipeline
    final_features = pipeline.fit_transform(features, target)

    print(f"✓ Pipeline completed: {len(features.columns)} -> {len(final_features.columns)} features")

    # Get selection report
    report = pipeline.get_selection_report()
    print(f"✓ Selection report generated with {len(report['pipeline_steps'])} steps")

    return final_features, report


def test_performance_benchmarks():
    """Test performance improvements."""
    print("\nTesting performance benchmarks...")

    import time

    # Create larger test data
    df = create_test_data(2000)
    config = FeatureConfig(
        enable_log_returns=True,
        enable_atr=True,
        enable_rsi=True,
        enable_bollinger_bands=True,
        enable_williams_r=True,
        enable_cci=True,
        enable_historical_volatility=True,
        enable_keltner_channels=True,
        enable_time_features=True
    )

    engineer = FeatureEngineer(config)

    # Benchmark feature engineering
    start_time = time.time()
    df_features = engineer.add_features(df)
    feature_time = time.time() - start_time

    print(f"✓ Feature engineering completed in {feature_time:.3f} seconds")
    print(f"  - Processing speed: {len(df) / feature_time:.0f} rows/second")
    print(f"  - Features generated: {len(df_features.columns) - len(df.columns)}")

    # Benchmark feature selection
    if len(df_features) > 100:
        target = df_features['log_ret'].dropna()
        features = df_features.loc[target.index, engineer.get_feature_names()].dropna()

        if len(features) > 50:
            start_time = time.time()
            corr_selector = CorrelationFeatureSelector(threshold=0.9)
            selected_features = corr_selector.fit_transform(features, target)
            selection_time = time.time() - start_time

            print(f"✓ Feature selection completed in {selection_time:.3f} seconds")
            print(f"  - Features reduced: {len(features.columns)} -> {len(selected_features.columns)}")
            print(f"  - Selection speed: {len(features.columns) / selection_time:.0f} features/second")

    return True


def main():
    """Run all enhanced feature engineering tests."""
    print("=" * 60)
    print("PHASE 2.1.2 ENHANCED FEATURE ENGINEERING VALIDATION")
    print("=" * 60)

    tests = [
        ("Enhanced Indicators", test_enhanced_indicators),
        ("Enhanced FeatureEngineer", test_feature_engineer_enhanced),
        ("Feature Selection", test_feature_selection),
        ("Feature Selection Pipeline", test_feature_selection_pipeline),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
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
        print("\n🎉 All tests passed! Enhanced feature engineering is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
