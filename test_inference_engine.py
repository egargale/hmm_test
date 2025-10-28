#!/usr/bin/env python3
"""
Test suite for HMM State Inference Engine (Task 5.1)

Tests the comprehensive inference functionality including:
- Basic state prediction with Viterbi algorithm
- Lagged state retrieval for lookahead bias prevention
- State stability analysis
- Comprehensive inference with confidence metrics
- Input validation and error handling
"""

import sys

import numpy as np

# Add src to path
sys.path.insert(0, "/home/1966enrico/src/hmm_test/src")


def generate_test_features(n_samples=100, n_features=3):
    """Generate synthetic test features for HMM testing."""
    np.random.seed(42)

    # Create realistic-looking features with some patterns
    features = np.random.randn(n_samples, n_features)

    # Add some autocorrelation to make it more realistic
    for i in range(1, n_samples):
        features[i] += 0.3 * features[i - 1] + 0.1 * np.random.randn(n_features)

    return features


def create_test_model():
    """Create a trained HMM model for testing."""
    from model_training import train_model
    from utils.config import HMMConfig

    config = HMMConfig(
        n_states=3,
        covariance_type="diag",
        max_iter=100,
        random_state=42,
        num_restarts=2,
    )

    features = generate_test_features(n_samples=100, n_features=3)
    model, scaler, score = train_model(features, config)

    return model, scaler, features, config


def test_basic_state_prediction():
    """Test basic state prediction functionality."""
    print("🔍 Testing Basic State Prediction")

    try:
        from model_training.inference_engine import predict_states
        print("✅ Inference engine module loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import inference engine: {e}")
        return False

    # Run all tests
    tests = [
        test_basic_state_prediction,
        test_lagged_state_prediction,
        test_comprehensive_inference,
        test_state_stability_analysis,
        test_input_validation,
        test_error_handling,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 70)
    print(f"Inference Engine Tests: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 Task 5.1 FULLY COMPLIANT!")
        print("\n✅ Key Achievements:")
        print("   ✓ Basic state prediction with Viterbi algorithm")
        print("   ✓ Posterior probability computation")
        print("   ✓ Lagged state retrieval for bias prevention")
        print("   ✓ Comprehensive inference with confidence metrics")
        print("   ✓ State stability analysis and transition tracking")
        print("   ✓ Input validation and error handling")
        print("   ✓ Edge case handling (single samples, small datasets)")
        print("\n🚀 Inference engine ready for production!")
        return True
    else:
        print("❌ Some test suites failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
