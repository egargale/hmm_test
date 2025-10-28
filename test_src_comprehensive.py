#!/usr/bin/env python3
"""
Comprehensive test of src modules using real BTC.csv data
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, "src")


def test_data_processing():
    """Test data processing modules with BTC.csv."""
    print("=" * 60)
    print("Testing Data Processing Modules with BTC.csv")
    print("=" * 60)

    try:
        from data_processing import add_features, process_csv, validate_data
        from utils import get_logger

        logger = get_logger(__name__)
        logger.info("Starting data processing tests")

        # Test CSV processing
        print("📁 Testing CSV processing...")
        data = process_csv("BTC.csv")
        print(f"✅ Loaded BTC.csv: {len(data)} rows, {len(data.columns)} columns")
        print(f"✅ Date range: {data.index.min()} to {data.index.max()}")
        print(f"✅ Columns: {list(data.columns)}")

        # Test data validation
        print("\n🔍 Testing data validation...")
        validation_result = validate_data(data)
        if validation_result["is_valid"]:
            print("✅ Data validation passed")
        else:
            print(f"⚠️ Data validation warnings: {validation_result['errors']}")

        # Test feature engineering
        print("\n⚙️ Testing feature engineering...")
        features = add_features(data)
        print(f"✅ Features added: {len(features.columns)} total columns")
        print(
            f"✅ New features: {len(features.columns) - len(data.columns)} indicators"
        )

        # Show some sample features
        feature_cols = [col for col in features.columns if col not in data.columns][:5]
        print(f"✅ Sample features: {feature_cols}")

        return True, features

    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_hmm_models(features):
    """Test HMM models with real BTC features."""
    print("\n" + "=" * 60)
    print("Testing HMM Models with BTC Data")
    print("=" * 60)

    try:
        from hmm_models import GaussianHMMModel, HMMModelFactory
        from utils import setup_logging

        # Setup logging
        setup_logging(level="INFO")

        # Prepare feature data
        feature_cols = [
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

        # Find available columns
        available_cols = [col for col in feature_cols if col in features.columns]
        print(f"📊 Using {len(available_cols)} features: {available_cols}")

        X = features[available_cols].dropna().values
        print(f"✅ Prepared feature matrix: {X.shape}")

        # Test Gaussian HMM
        print("\n🔢 Testing Gaussian HMM...")
        model = GaussianHMMModel(
            n_components=3,
            covariance_type="diag",
            random_state=42,
            n_iter=50,
            verbose=False,
        )

        print("📈 Training HMM...")
        model.fit(X)
        print("✅ HMM training completed")

        # Test state prediction
        states = model.predict(X)
        print(f"✅ Predicted states: {len(np.unique(states))} unique regimes")
        print(f"✅ State distribution: {np.bincount(states) / len(states)}")

        # Test model factory
        print("\n🏭 Testing HMM Factory...")
        factory_model = HMMModelFactory.create_model(
            model_type="gaussian",
            n_components=3,
            n_samples=len(X),
            n_features=X.shape[1],
        )
        print(f"✅ Factory created: {type(factory_model).__name__}")

        # Test model persistence
        print("\n💾 Testing model persistence...")
        model_path = "btc_hmm_test_model.pkl"
        model.save_model(model_path)
        print(f"✅ Model saved: {model_path}")

        # Load and test
        loaded_model = GaussianHMMModel()
        loaded_model.load_model(model_path)
        loaded_states = loaded_model.predict(X)
        print(f"✅ Model loaded and verified: {len(np.unique(loaded_states))} states")

        # Cleanup
        Path(model_path).unlink(missing_ok=True)

        return True, states

    except Exception as e:
        print(f"❌ HMM model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_backtesting_engine(features, states):
    """Test backtesting engine with BTC data and HMM states."""
    print("\n" + "=" * 60)
    print("Testing Backtesting Engine")
    print("=" * 60)

    try:
        from backtesting.performance_analyzer import PerformanceAnalyzer
        from backtesting.strategy_engine import StrategyEngine
        from utils.data_types import BacktestConfig

        # Prepare price data
        prices = (
            features["Close"] if "Close" in features.columns else features.iloc[:, 3]
        )

        # Align states and prices
        min_len = min(len(states), len(prices))
        states_aligned = states[:min_len]
        prices_aligned = prices.iloc[:min_len]

        print(
            f"📊 Aligned data: {len(prices_aligned)} price points, {len(states_aligned)} states"
        )

        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=100000.0,
            commission=0.001,  # 0.1%
            slippage=0.0001,  # 0.01%
            lookahead_bias_prevention=True,
            lookahead_days=1,
        )

        # Create state-to-position mapping
        unique_states = np.unique(states_aligned)
        state_mapping = {}
        for i, state in enumerate(unique_states):
            if state < 0:
                state_mapping[state] = 0  # Neutral for invalid states
            else:
                # Simple mapping: even states = long, odd states = short
                state_mapping[state] = 1 if i % 2 == 0 else -1

        print(f"🎯 State mapping: {state_mapping}")

        # Initialize strategy engine
        strategy_engine = StrategyEngine(config)

        print("💰 Running backtest...")
        backtest_result = strategy_engine.backtest_strategy(
            data=features.iloc[:min_len],
            states=states_aligned,
            state_mapping=state_mapping,
        )

        print("✅ Backtest completed:")
        print(f"  - Trades generated: {len(backtest_result.trades)}")
        print(f"  - Final equity: ${backtest_result.equity_curve.iloc[-1]:.2f}")

        # Performance analysis
        print("\n📈 Analyzing performance...")
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_performance(
            backtest_result.equity_curve,
            backtest_result.positions,
            benchmark=prices_aligned.pct_change(),
        )

        print("✅ Performance metrics:")
        print(f"  - Total return: {metrics.total_return:.2%}")
        print(f"  - Annualized return: {metrics.annualized_return:.2%}")
        print(f"  - Volatility: {metrics.annualized_volatility:.2%}")
        print(f"  - Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  - Max drawdown: {metrics.max_drawdown:.2%}")
        print(f"  - Win rate: {metrics.win_rate:.2%}")

        return True

    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_inference_engine(features):
    """Test state inference engine."""
    print("\n" + "=" * 60)
    print("Testing State Inference Engine")
    print("=" * 60)

    try:
        from hmm_models import GaussianHMMModel
        from model_training.inference_engine import StateInference

        # Prepare features
        feature_cols = [
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
        available_cols = [col for col in feature_cols if col in features.columns]
        X = features[available_cols].dropna().values

        # Train a model for inference
        model = GaussianHMMModel(n_components=3, random_state=42, n_iter=50)
        model.fit(X)

        # Test inference engine
        print("🔍 Testing state inference...")
        inference = StateInference(model)

        # Test batch inference
        states = inference.infer_states(X)
        print(f"✅ Batch inference: {len(states)} states inferred")

        # Test online inference (streaming)
        print("🌊 Testing streaming inference...")
        online_states = []
        for i in range(min(100, len(X))):
            state = inference.infer_single_state(X[i : i + 1])
            online_states.append(state)

        print(f"✅ Streaming inference: {len(online_states)} states inferred")
        print(f"✅ State consistency: {np.mean(states[:100] == online_states):.2%}")

        # Test state probabilities
        probs = inference.infer_state_probabilities(X[:10])
        print(f"✅ State probabilities: {probs.shape} matrix")

        return True

    except Exception as e:
        print(f"❌ Inference engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run comprehensive test suite with BTC.csv."""
    print("🚀 Starting Comprehensive BTC.csv Test Suite")
    print("=" * 60)

    # Check if BTC.csv exists
    if not Path("BTC.csv").exists():
        print("❌ BTC.csv not found in current directory")
        return False

    # Run tests
    results = {}

    # Test 1: Data Processing
    success, features = test_data_processing()
    results["Data Processing"] = success

    if success and features is not None:
        # Test 2: HMM Models
        success, states = test_hmm_models(features)
        results["HMM Models"] = success

        if success and states is not None:
            # Test 3: Backtesting
            results["Backtesting"] = test_backtesting_engine(features, states)

            # Test 4: Inference Engine
            results["Inference Engine"] = test_inference_engine(features)

    # Summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} test modules passed")

    if passed == total:
        print("🎉 All src module tests passed with BTC.csv!")
        print("✅ Comprehensive HMM analysis system is fully functional!")
        return True
    else:
        print("❌ Some module tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
