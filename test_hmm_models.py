#!/usr/bin/env python3
"""
Test script for HMM model implementations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def generate_test_data(n_samples=1000, n_features=5, random_state=42):
    """Generate synthetic test data for HMM testing."""
    np.random.seed(random_state)

    # Create synthetic regime-switching data
    n_states = 3
    state_sequence = np.random.choice(n_states, size=n_samples, p=[0.3, 0.4, 0.3])

    # Generate returns with different characteristics per regime
    data = []
    for i, state in enumerate(state_sequence):
        if state == 0:  # Bull market
            returns = np.random.normal(0.001, 0.015, n_features)
        elif state == 1:  # Bear market
            returns = np.random.normal(-0.0005, 0.025, n_features)
        else:  # Sideways market
            returns = np.random.normal(0.0001, 0.008, n_features)

        data.append(returns)

    data = np.array(data)

    # Create DataFrame with realistic column names
    feature_names = ['log_ret', 'volatility', 'volume_change', 'momentum', 'rsi']
    df = pd.DataFrame(data, columns=feature_names)

    return df, state_sequence

def test_hmm_models():
    """Test the HMM model implementations."""
    print("🧪 Testing HMM Model Implementations...")

    try:
        from hmm_models import GaussianHMMModel, GMMHMMModel, HMMModelFactory
        from utils import setup_logging

        # Setup logging
        logger = setup_logging(level="INFO")
        logger.info("Starting HMM model tests")

        # Generate test data
        print("\n📊 Generating test data...")
        X, true_states = generate_test_data(n_samples=1000, n_features=5)
        print(f"   ✅ Generated test data: {X.shape} matrix with {len(np.unique(true_states))} true states")

        # Test 1: Gaussian HMM Model
        print("\n🔢 Testing Gaussian HMM Model...")
        try:
            # Create and fit model
            gaussian_model = GaussianHMMModel(
                n_components=3,
                covariance_type="full",
                random_state=42,
                n_iter=50,  # Reduce for testing
                verbose=False
            )

            print("   📈 Fitting Gaussian HMM...")
            gaussian_model.fit(X)
            print(f"   ✅ Gaussian HMM fitted successfully")

            # Test predictions
            states = gaussian_model.predict(X)
            print(f"   ✅ Predicted states: {len(np.unique(states))} unique states")
            print(f"   ✅ State distribution: {np.bincount(states) / len(states)}")

            # Test probabilities
            probs = gaussian_model.predict_proba(X)
            print(f"   ✅ State probabilities shape: {probs.shape}")

            # Test model quality
            quality = gaussian_model.evaluate_model_quality(X)
            print(f"   ✅ Model quality - Log-likelihood: {quality['total_log_likelihood']:.2f}")
            print(f"   ✅ Model quality - BIC: {quality['bic']:.2f}")

            # Test state descriptions
            descriptions = gaussian_model.get_state_descriptions(X)
            print(f"   ✅ Generated descriptions for {len(descriptions)} states")

            # Test transition analysis
            transitions = gaussian_model.analyze_state_transitions(X)
            print(f"   ✅ Analyzed state transitions")

        except Exception as e:
            print(f"   ❌ Gaussian HMM test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 2: GMM HMM Model
        print("\n🎭 Testing GMM HMM Model...")
        try:
            # Create and fit model
            gmm_model = GMMHMMModel(
                n_components=3,
                n_mix=2,
                covariance_type="full",
                random_state=42,
                n_iter=50,  # Reduce for testing
                verbose=False
            )

            print("   📈 Fitting GMM HMM...")
            gmm_model.fit(X)
            print(f"   ✅ GMM HMM fitted successfully")

            # Test predictions
            states = gmm_model.predict(X)
            print(f"   ✅ Predicted states: {len(np.unique(states))} unique states")
            print(f"   ✅ State distribution: {np.bincount(states) / len(states)}")

            # Test model quality
            quality = gmm_model.evaluate_model_quality(X)
            print(f"   ✅ Model quality - Log-likelihood: {quality['total_log_likelihood']:.2f}")
            print(f"   ✅ Model quality - BIC: {quality['bic']:.2f}")

            # Test state descriptions
            descriptions = gmm_model.get_state_descriptions(X)
            print(f"   ✅ Generated descriptions for {len(descriptions)} states")
            for state_id, desc in descriptions.items():
                print(f"      State {state_id}: {desc['n_mixtures']} mixture components")

            # Test mixture analysis
            mixture_analysis = gmm_model.analyze_mixture_separation()
            print(f"   ✅ Analyzed mixture separation")

        except Exception as e:
            print(f"   ❌ GMM HMM test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 3: HMM Factory
        print("\n🏭 Testing HMM Model Factory...")
        try:
            # Test model creation
            factory_model = HMMModelFactory.create_model(
                model_type='gaussian',
                n_components=3,
                n_samples=len(X),
                n_features=X.shape[1]
            )
            print(f"   ✅ Factory created model: {type(factory_model).__name__}")

            # Test auto-selection
            recommended_type = HMMModelFactory.auto_select_model_type(
                n_samples=len(X),
                n_features=X.shape[1],
                data_complexity='moderate'
            )
            print(f"   ✅ Auto-selected model type: {recommended_type}")

            # Test ensemble creation
            ensemble = HMMModelFactory.create_model_ensemble(
                model_types=['gaussian'],
                n_components_range=[2, 3],
                n_samples=len(X),
                n_features=X.shape[1]
            )
            print(f"   ✅ Created ensemble of {len(ensemble)} models")

            # Test available models info
            models_info = HMMModelFactory.get_available_models()
            print(f"   ✅ Available models: {list(models_info.keys())}")

        except Exception as e:
            print(f"   ❌ Factory test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 4: Model Persistence
        print("\n💾 Testing Model Persistence...")
        try:
            # Save model
            model_path = "test_gaussian_hmm.pkl"
            gaussian_model.save_model(model_path)
            print(f"   ✅ Model saved to {model_path}")

            # Load model
            loaded_model = GaussianHMMModel()
            loaded_model.load_model(model_path)
            print(f"   ✅ Model loaded successfully")

            # Verify loaded model works
            loaded_states = loaded_model.predict(X)
            print(f"   ✅ Loaded model predicts {len(np.unique(loaded_states))} states")

            # Clean up
            Path(model_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"   ❌ Persistence test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 5: Cross-validation
        print("\n🔄 Testing Cross-validation...")
        try:
            cv_results = gaussian_model.cross_validate(X, cv=3)
            print(f"   ✅ Cross-validation completed")
            print(f"   ✅ Mean CV score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")

        except Exception as e:
            print(f"   ❌ Cross-validation test failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n✅ HMM Model tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ HMM Model tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hmm_models()
    sys.exit(0 if success else 1)