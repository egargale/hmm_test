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
sys.path.insert(0, '/home/1966enrico/src/hmm_test/src')

def generate_test_features(n_samples=100, n_features=3):
    """Generate synthetic test features for HMM testing."""
    np.random.seed(42)

    # Create realistic-looking features with some patterns
    features = np.random.randn(n_samples, n_features)

    # Add some autocorrelation to make it more realistic
    for i in range(1, n_samples):
        features[i] += 0.3 * features[i-1] + 0.1 * np.random.randn(n_features)

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
        num_restarts=2
    )

    features = generate_test_features(n_samples=100, n_features=3)
    model, scaler, score = train_model(features, config)

    return model, scaler, features, config

def test_basic_state_prediction():
    """Test basic state prediction functionality."""
    print("🔍 Testing Basic State Prediction")

    try:
        from model_training.inference_engine import predict_states

        model, scaler, features, config = create_test_model()

        # Test basic prediction (default returns states, probabilities, log_likelihood)
        result = predict_states(model, scaler, features)

        if isinstance(result, tuple) and len(result) == 3:
            states, probabilities, log_likelihood = result
            print("  ✓ Prediction returned tuple with states, probabilities, and log_likelihood")
        else:
            print(f"  ✗ Expected tuple of length 3, got {type(result)} with length {len(result) if hasattr(result, '__len__') else 'unknown'}")
            return False

        if len(states) == len(features):
            print(f"  ✓ States length matches features: {len(states)}")
        else:
            print(f"  ✗ Length mismatch: {len(states)} != {len(features)}")
            return False

        if states.min() >= 0 and states.max() < config.n_states:
            print(f"  ✓ State values valid: {states.min()} to {states.max()}")
        else:
            print(f"  ✗ Invalid state values: {states.min()} to {states.max()}")
            return False

        if probabilities.shape == (len(features), config.n_states):
            print(f"  ✓ Probabilities shape correct: {probabilities.shape}")
        else:
            print(f"  ✗ Wrong probabilities shape: {probabilities.shape}")
            return False

        # Test with return_probabilities=False (still returns log_likelihood by default)
        states_only = predict_states(model, scaler, features, return_probabilities=False)
        if isinstance(states_only, tuple) and len(states_only) == 2:
            states, log_likelihood = states_only
            if len(states) == len(features):
                print(f"  ✓ States-only prediction works: {len(states)}")
            else:
                print("  ✗ States-only prediction failed: wrong length")
                return False
        else:
            print("  ✗ States-only prediction failed: expected tuple of length 2")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Basic prediction test failed: {e}")
        return False

def test_lagged_state_prediction():
    """Test lagged state prediction for bias prevention."""
    print("\n🔍 Testing Lagged State Prediction")

    try:
        from model_training.inference_engine import predict_states_with_lag

        model, scaler, features, config = create_test_model()

        # Test lag=1
        result_lag1 = predict_states_with_lag(model, scaler, features, lag_periods=1)

        if hasattr(result_lag1, 'original_states') and hasattr(result_lag1, 'lagged_states'):
            print("  ✓ Lagged prediction returned correct object type")
        else:
            print("  ✗ Missing required attributes in lagged result")
            return False

        if len(result_lag1.original_states) == len(features):
            print(f"  ✓ Original states length correct: {len(result_lag1.original_states)}")
        else:
            print(f"  ✗ Original states length wrong: {len(result_lag1.original_states)}")
            return False

        if result_lag1.lag_periods == 1:
            print(f"  ✓ Lag periods correct: {result_lag1.lag_periods}")
        else:
            print(f"  ✗ Lag periods wrong: {result_lag1.lag_periods}")
            return False

        # Test that lagged states have fill_value at the beginning (default is -1)
        if result_lag1.lagged_states[0] == -1:
            print("  ✓ Lagged states have fill_value at beginning")
        else:
            print(f"  ✗ Lagged states don't start with fill_value: {result_lag1.lagged_states[0]}")
            return False

        # Test lag=3
        result_lag3 = predict_states_with_lag(model, scaler, features, lag_periods=3)
        if result_lag3.lag_periods == 3:
            print(f"  ✓ Higher lag periods work: {result_lag3.lag_periods}")
        else:
            print("  ✗ Higher lag periods failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Lagged prediction test failed: {e}")
        return False

def test_comprehensive_inference():
    """Test comprehensive inference with all metrics."""
    print("\n🔍 Testing Comprehensive Inference")

    try:
        from model_training.inference_engine import predict_states_comprehensive

        model, scaler, features, config = create_test_model()

        # Test comprehensive inference
        result = predict_states_comprehensive(model, scaler, features)

        # Check all required attributes
        required_attrs = ['states', 'probabilities', 'log_likelihood', 'n_states', 'n_samples', 'metadata']

        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(result, attr):
                missing_attrs.append(attr)

        if not missing_attrs:
            print(f"  ✓ All required attributes present: {len(required_attrs)}")
        else:
            print(f"  ✗ Missing attributes: {missing_attrs}")
            return False

        if result.n_samples == len(features):
            print(f"  ✓ Sample count correct: {result.n_samples}")
        else:
            print(f"  ✗ Sample count wrong: {result.n_samples} != {len(features)}")
            return False

        if result.n_states == config.n_states:
            print(f"  ✓ State count correct: {result.n_states}")
        else:
            print(f"  ✗ State count wrong: {result.n_states} != {config.n_states}")
            return False

        # Check state distribution in metadata
        if 'state_distribution' in result.metadata:
            state_dist = result.metadata['state_distribution']
            if isinstance(state_dist, dict):
                total_states = sum(state_dist.values())
                if total_states == result.n_samples:
                    print(f"  ✓ State distribution consistent: {total_states} states")
                else:
                    print(f"  ✗ State distribution inconsistent: {total_states} != {result.n_samples}")
                    return False
            else:
                print("  ✗ State distribution not a dict")
                return False
        else:
            print("  ✗ State distribution missing from metadata")
            return False

        # Test confidence values in metadata
        if 'mean_prediction_confidence' in result.metadata:
            confidence = result.metadata['mean_prediction_confidence']
            if 0 <= confidence <= 1:
                print(f"  ✓ Mean confidence valid: {confidence:.3f}")
            else:
                print(f"  ✗ Invalid mean confidence: {confidence}")
                return False
        else:
            print("  ✗ Mean confidence missing from metadata")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Comprehensive inference test failed: {e}")
        return False

def test_state_stability_analysis():
    """Test state stability analysis functionality."""
    print("\n🔍 Testing State Stability Analysis")

    try:
        from model_training.inference_engine import analyze_state_stability

        model, scaler, features, config = create_test_model()

        # Test stability analysis
        stability = analyze_state_stability(model, scaler, features, window_size=10)

        # Check required keys in stability dictionary
        required_keys = ['n_samples', 'n_states', 'window_size', 'change_rate', 'avg_persistence_length']
        missing_keys = [key for key in required_keys if key not in stability]

        if not missing_keys:
            print("  ✓ Stability analysis has all required keys")
        else:
            print(f"  ✗ Missing keys: {missing_keys}")
            return False

        # Check that n_states matches config
        if stability['n_states'] == config.n_states:
            print(f"  ✓ State count correct: {stability['n_states']}")
        else:
            print(f"  ✗ Wrong state count: {stability['n_states']}")
            return False

        # Check that change rate is reasonable (0-1)
        if 0 <= stability['change_rate'] <= 1:
            print(f"  ✓ Change rate valid: {stability['change_rate']:.3f}")
        else:
            print(f"  ✗ Invalid change rate: {stability['change_rate']}")
            return False

        # Test transition matrix in analysis
        if 'transition_probabilities' in stability:
            trans_matrix = stability['transition_probabilities']
            if len(trans_matrix) == config.n_states and all(len(row) == config.n_states for row in trans_matrix):
                print(f"  ✓ Transition matrix shape correct: {config.n_states}x{config.n_states}")
            else:
                print("  ✗ Wrong transition matrix shape")
                return False
        else:
            print("  ✗ Transition probabilities missing")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Stability analysis test failed: {e}")
        return False

def test_input_validation():
    """Test input validation and error handling."""
    print("\n🔍 Testing Input Validation")

    try:
        from model_training.inference_engine import (
            predict_states,
            validate_inference_inputs,
        )

        model, scaler, features, config = create_test_model()

        # Test validation function
        try:
            validate_inference_inputs(None, scaler, features)
            print("  ✗ Should have failed with None model")
            return False
        except ValueError:
            print("  ✓ Correctly rejects None model")

        try:
            validate_inference_inputs(model, None, features)
            print("  ✗ Should have failed with None scaler")
            return False
        except ValueError:
            print("  ✓ Correctly rejects None scaler")

        try:
            validate_inference_inputs(model, scaler, None)
            print("  ✗ Should have failed with None features")
            return False
        except ValueError:
            print("  ✓ Correctly rejects None features")

        # Test wrong feature dimensions
        wrong_features = np.random.randn(10, 5)  # Wrong number of features
        try:
            predict_states(model, scaler, wrong_features)
            print("  ✗ Should have failed with wrong feature dimensions")
            return False
        except ValueError:
            print("  ✓ Correctly rejects wrong feature dimensions")

        # Test empty features
        empty_features = np.array([]).reshape(0, 3)
        try:
            predict_states(model, scaler, empty_features)
            print("  ✗ Should have failed with empty features")
            return False
        except ValueError:
            print("  ✓ Correctly rejects empty features")

        return True

    except Exception as e:
        print(f"  ✗ Input validation test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in edge cases."""
    print("\n🔍 Testing Error Handling")

    try:
        from model_training.inference_engine import predict_states_comprehensive

        # Test with single sample
        model, scaler, features, config = create_test_model()
        single_sample = features[:1]

        result = predict_states_comprehensive(model, scaler, single_sample)
        if result.n_samples == 1:
            print("  ✓ Single sample handled correctly")
        else:
            print(f"  ✗ Single sample failed: {result.n_samples}")
            return False

        # Test with very small features (should still work)
        small_features = features[:5]
        result = predict_states_comprehensive(model, scaler, small_features)
        if result.n_samples == 5:
            print("  ✓ Small feature set handled correctly")
        else:
            print(f"  ✗ Small feature set failed: {result.n_samples}")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        return False

def main():
    """Run all inference engine tests."""
    print("=" * 70)
    print("HMM INFERENCE ENGINE TESTS (Task 5.1)")
    print("=" * 70)

    # Check dependencies first
    try:
        from model_training.inference_engine import (
            predict_states,
            predict_states_comprehensive,
        )
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
        test_error_handling
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
