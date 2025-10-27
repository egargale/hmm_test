"""
Test script for HMM Model Training Pipeline

This script tests the HMM training implementation and validates its functionality
for training Hidden Markov Models on financial time series data.
"""

import sys

import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from model_training.hmm_trainer import (
    evaluate_model,
    get_hmm_model_info,
    predict_states,
    train_model,
    validate_features_for_hmm,
    validate_hmm_config,
)
from utils import HMMConfig


def create_synthetic_features(n_samples=1000, n_features=8, n_states=3, random_state=42):
    """
    Create synthetic feature data for HMM testing.

    Args:
        n_samples: Number of time steps
        n_features: Number of features
        n_states: Number of hidden states to simulate
        random_state: Random seed

    Returns:
        numpy array of synthetic features
    """
    np.random.seed(random_state)

    # Simulate regime-switching data
    features = []
    current_state = 0

    # Define different regimes with different characteristics
    regimes = [
        {'mean': 0.0, 'std': 1.0, 'trend': 0.01},      # Bull market
        {'mean': 0.0, 'std': 2.0, 'trend': -0.005},   # Bear market
        {'mean': 0.0, 'std': 0.5, 'trend': 0.002}     # Sideways market
    ]

    for i in range(n_samples):
        # Switch states occasionally (Poisson process)
        if np.random.random() < 0.02:  # 2% chance of regime change
            current_state = np.random.randint(0, n_states)

        regime = regimes[current_state]

        # Generate features for this time step
        feature_vector = []

        for j in range(n_features):
            if j == 0:  # Returns
                value = np.random.normal(regime['mean'], regime['std'])
            elif j == 1:  # Volatility
                value = abs(np.random.normal(0, regime['std']))
            elif j == 2:  # Trend
                value = regime['trend'] + np.random.normal(0, 0.1)
            elif j == 3:  # Momentum
                value = np.random.normal(0, 1) * regime['std']
            elif j == 4:  # Volume indicator
                value = abs(np.random.normal(1, 0.5))
            elif j == 5:  # Price position
                value = np.random.normal(0.5, 0.2)
            elif j == 6:  # High-low ratio
                value = np.random.normal(1.02, 0.01)
            elif j == 7:  # Moving average deviation
                value = np.random.normal(0, regime['std'] * 0.5)

            feature_vector.append(value)

        features.append(feature_vector)

    return np.array(features)

def test_hmm_basic_training():
    """Test basic HMM training functionality."""
    print("=" * 60)
    print("Testing HMM Basic Training Functionality")
    print("=" * 60)

    try:
        # Check HMM availability
        try:
            from hmmlearn import hmm
            print("✓ HMMlearn is available")
        except ImportError:
            print("✗ HMMlearn is not available")
            return False

        # Create synthetic data
        print("Creating synthetic feature data...")
        features = create_synthetic_features(n_samples=500, n_features=6, n_states=3)
        print(f"✓ Created synthetic data: {features.shape}")

        # Create HMM configuration
        config = HMMConfig(
            n_states=3,
            covariance_type="diag",
            max_iter=50,
            random_state=42,
            tol=1e-3,
            num_restarts=3
        )
        print(f"✓ Created HMM config: {config.n_states} states, {config.num_restarts} restarts")

        # Test basic training
        print("\nTesting basic HMM training...")
        model, scaler, score = train_model(features, config)

        print("✓ HMM training completed successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of states: {model.n_components}")
        print(f"  - Number of features: {model.n_features}")
        print(f"  - Log likelihood score: {score:.4f}")
        print(f"  - Converged: {model.monitor_.converged}")
        print(f"  - Iterations: {len(model.monitor_.history)}")

        # Test prediction
        print("\nTesting state prediction...")
        states, posterior = predict_states(model, features, scaler)
        print(f"✓ State prediction completed: {states.shape} states, {posterior.shape} posterior probabilities")

        # Evaluate model
        print("\nTesting model evaluation...")
        eval_results = evaluate_model(model, features, scaler)
        print("✓ Model evaluation completed:")
        for key, value in eval_results.items():
            print(f"  - {key}: {value}")

        # Get model info
        print("\nTesting model info extraction...")
        model_info = get_hmm_model_info(model)
        print(f"✓ Model info extracted: {len(model_info)} fields")

        return True

    except Exception as e:
        print(f"✗ Basic HMM training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hmm_multiple_restarts():
    """Test HMM training with multiple restarts."""
    print("\n" + "=" * 60)
    print("Testing HMM Multiple Restarts")
    print("=" * 60)

    try:
        # Create synthetic data
        print("Creating synthetic feature data...")
        features = create_synthetic_features(n_samples=300, n_features=5, n_states=2)

        # Create config with multiple restarts
        config = HMMConfig(
            n_states=2,
            covariance_type="diag",
            max_iter=30,
            random_state=42,
            tol=1e-3,
            num_restarts=5
        )

        print(f"Training HMM with {config.num_restarts} restarts...")

        # Train with detailed results
        result = train_model(features, config, return_all_results=True)

        print("✓ Multiple restarts training completed:")
        print(f"  - Total restarts: {result.n_restarts_completed}")
        print(f"  - Successful restarts: {result.n_successful_restarts}")
        print(f"  - Success rate: {result.convergence_info['success_rate']:.2%}")
        print(f"  - Best score: {result.score:.4f}")
        print(f"  - Best restart: {result.convergence_info['best_restart']}")
        print(f"  - Converged: {result.convergence_info['best_converged']}")

        # Validate that we got good results
        if result.n_successful_restarts < config.num_restarts * 0.6:
            print(f"⚠ Warning: Low success rate ({result.n_successful_restarts}/{config.num_restarts})")

        return True

    except Exception as e:
        print(f"✗ Multiple restarts test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hmm_numerical_stability():
    """Test HMM numerical stability handling."""
    print("\n" + "=" * 60)
    print("Testing HMM Numerical Stability")
    print("=" * 60)

    try:
        # Test with challenging data (zero variance features)
        print("Creating challenging data with zero-variance features...")

        # Create data with some zero-variance features
        n_samples = 200
        base_features = np.random.randn(n_samples, 3)

        # Add zero-variance features
        zero_var_features = np.ones((n_samples, 2)) * 0.5  # Constant columns
        challenging_features = np.hstack([base_features, zero_var_features])

        print(f"✓ Created challenging data: {challenging_features.shape}")
        print(f"  - Features variance: {np.var(challenging_features, axis=0)}")

        # Test with and without numerical stability
        config = HMMConfig(
            n_states=2,
            covariance_type="diag",
            max_iter=20,
            random_state=42,
            num_restarts=2
        )

        # Test with numerical stability enabled
        print("\nTesting with numerical stability enabled...")
        try:
            model_stable, scaler_stable, score_stable = train_model(
                challenging_features, config, enable_numerical_stability=True
            )
            print(f"✓ Training with stability enabled: score={score_stable:.4f}")
        except Exception as e:
            print(f"⚠ Training with stability failed: {e}")
            model_stable = None

        # Test without numerical stability
        print("\nTesting without numerical stability...")
        try:
            model_unstable, scaler_unstable, score_unstable = train_model(
                challenging_features, config, enable_numerical_stability=False
            )
            print(f"✓ Training without stability: score={score_unstable:.4f}")
        except Exception as e:
            print(f"⚠ Training without stability failed: {e}")
            model_unstable = None

        # At least one should succeed
        if model_stable is not None or model_unstable is not None:
            print("✓ Numerical stability test passed (at least one configuration succeeded)")
            return True
        else:
            print("✗ Both stability configurations failed")
            return False

    except Exception as e:
        print(f"✗ Numerical stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hmm_input_validation():
    """Test HMM input validation."""
    print("\n" + "=" * 60)
    print("Testing HMM Input Validation")
    print("=" * 60)

    validation_passed = True

    try:
        # Test feature validation
        print("Testing feature validation...")

        # Test invalid features
        invalid_cases = [
            (np.array([]), "empty array"),
            (np.random.randn(100), "1D array"),
            (np.random.randn(0, 5), "zero rows"),
            (np.random.randn(100, 0), "zero columns"),
            (np.array([[np.nan, 1], [2, 3]]), "NaN values"),
            (np.array([[np.inf, 1], [2, 3]]), "infinite values")
        ]

        for invalid_features, case_name in invalid_cases:
            try:
                validate_features_for_hmm(invalid_features)
                print(f"✗ Expected validation to fail for {case_name}")
                validation_passed = False
            except (ValueError, RuntimeError):
                print(f"✓ Correctly rejected {case_name}")

        # Test config validation
        print("\nTesting config validation...")

        invalid_configs = [
            HMMConfig(n_states=0),  # Invalid number of states
            HMMConfig(max_iter=0),  # Invalid max iterations
            HMMConfig(tol=-0.1),    # Invalid tolerance
            HMMConfig(num_restarts=0),  # Invalid number of restarts
        ]

        config_names = ["n_states=0", "max_iter=0", "tol=-0.1", "num_restarts=0"]

        for invalid_config, name in zip(invalid_configs, config_names):
            try:
                validate_hmm_config(invalid_config)
                print(f"✗ Expected validation to fail for {name}")
                validation_passed = False
            except ValueError:
                print(f"✓ Correctly rejected {name}")

        return validation_passed

    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hmm_different_configurations():
    """Test HMM with different configurations."""
    print("\n" + "=" * 60)
    print("Testing HMM Different Configurations")
    print("=" * 60)

    try:
        # Create synthetic data
        features = create_synthetic_features(n_samples=400, n_features=6, n_states=3)

        # Test different configurations
        configs = [
            {"n_states": 2, "covariance_type": "diag", "name": "2-state diagonal"},
            {"n_states": 3, "covariance_type": "diag", "name": "3-state diagonal"},
            {"n_states": 2, "covariance_type": "full", "name": "2-state full"},
            {"n_states": 3, "covariance_type": "full", "name": "3-state full"},
        ]

        results = []

        for config_params in configs:
            name = config_params.pop("name")
            print(f"\nTesting {name} configuration...")

            config = HMMConfig(
                max_iter=30,
                random_state=42,
                num_restarts=2,
                **config_params
            )

            try:
                model, scaler, score = train_model(features, config)
                eval_results = evaluate_model(model, features, scaler)

                result = {
                    "name": name,
                    "success": True,
                    "score": score,
                    "converged": eval_results["converged"],
                    "iterations": eval_results["n_iterations"],
                    "transition_entropy": eval_results["transition_entropy"]
                }
                results.append(result)

                print(f"✓ {name}: score={score:.4f}, converged={eval_results['converged']}")

            except Exception as e:
                print(f"✗ {name}: FAILED - {e}")
                results.append({"name": name, "success": False, "error": str(e)})

        # Summary
        successful = [r for r in results if r["success"]]
        print("\nConfiguration test summary:")
        print(f"  - Successful: {len(successful)}/{len(results)}")

        if successful:
            best_result = max(successful, key=lambda x: x["score"])
            print(f"  - Best configuration: {best_result['name']} (score: {best_result['score']:.4f})")

        return len(successful) > 0

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all HMM training tests."""
    print("Starting HMM Training Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("Basic Training", test_hmm_basic_training),
        ("Multiple Restarts", test_hmm_multiple_restarts),
        ("Numerical Stability", test_hmm_numerical_stability),
        ("Input Validation", test_hmm_input_validation),
        ("Different Configurations", test_hmm_different_configurations)
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
        print("🎉 All HMM training tests passed!")
        return True
    else:
        print("❌ Some HMM training tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
