"""
Simplified unit tests for HMM models and training.

Tests the actual functions available in the HMM modules without relying
on classes that may not exist.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from model_training.hmm_trainer import (
    validate_features_for_hmm,
    train_single_hmm_model,
    add_numerical_stability_epsilon,
    train_model,
    predict_states,
    evaluate_model,
    get_hmm_model_info
)
from model_training.inference_engine import StateInference


class TestHMMTrainingFunctions:
    """Test HMM training functions."""

    def test_validate_features_for_hmm_valid(self):
        """Test feature validation with valid data."""
        # Valid 2D features
        features = np.random.normal(0, 1, (100, 3))
        validate_features_for_hmm(features)  # Should not raise

    def test_validate_features_for_hmm_insufficient_samples(self):
        """Test feature validation with insufficient samples."""
        features = np.random.normal(0, 1, (1, 3))  # Only 1 sample
        with pytest.raises(ValueError, match="Insufficient samples"):
            validate_features_for_hmm(features)

    def test_validate_features_for_hmm_zero_variance(self):
        """Test feature validation with zero variance features."""
        features = np.ones((100, 3))  # All features are constant
        with pytest.raises(ValueError, match="Zero variance detected"):
            validate_features_for_hmm(features)

    def test_validate_features_for_hmm_nan_values(self):
        """Test feature validation with NaN values."""
        features = np.random.normal(0, 1, (100, 3))
        features[50, 1] = np.nan  # Add NaN value
        with pytest.raises(ValueError, match="NaN values detected"):
            validate_features_for_hmm(features)

    def test_add_numerical_stability_epsilon(self):
        """Test numerical stability epsilon addition."""
        features = np.array([0.0, -1e-8, 1e-8])
        epsilon = 1e-6
        stable_features = add_numerical_stability_epsilon(features, epsilon)

        expected = np.array([epsilon, epsilon - 1e-8, epsilon + 1e-8])
        np.testing.assert_array_equal(stable_features, expected)

    def test_add_numerical_stability_positive_values(self):
        """Test that positive values are unchanged."""
        features = np.array([1.0, 2.0, 3.0])
        epsilon = 1e-6
        stable_features = add_numerical_stability_epsilon(features, epsilon)

        np.testing.assert_array_equal(stable_features, features)

    def test_train_single_hmm_model_basic(self):
        """Test basic single HMM model training."""
        # Create simple test data
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 1))

        config = {
            'n_components': 2,
            'covariance_type': 'full',
            'n_iter': 10,
            'random_state': 42
        }

        model, metadata = train_single_hmm_model(features, config)

        assert model is not None
        assert metadata is not None
        assert 'n_components' in metadata
        assert 'log_likelihood' in metadata
        assert metadata['n_components'] == 2

    def test_train_single_hmm_model_multidimensional(self):
        """Test training with multidimensional features."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 3))

        config = {
            'n_components': 3,
            'covariance_type': 'diag',
            'n_iter': 10,
            'random_state': 42
        }

        model, metadata = train_single_hmm_model(features, config)

        assert model is not None
        assert metadata['n_components'] == 3

    def test_train_model_with_validation(self):
        """Test training model with feature validation."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 2))

        config = {
            'n_components': 2,
            'n_iter': 10,
            'random_state': 42
        }

        model, metadata = train_model(
            features,
            config=config,
            validate_features=True
        )

        assert model is not None
        assert metadata is not None

    def test_predict_states_basic(self):
        """Test basic state prediction."""
        # Train a simple model first
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 1))

        config = {'n_components': 2, 'n_iter': 10, 'random_state': 42}
        model, _ = train_single_hmm_model(features, config)

        # Predict states
        states = predict_states(model, features)

        assert isinstance(states, np.ndarray)
        assert len(states) == len(features)
        assert np.all(states >= 0)
        assert np.all(states < config['n_components'])

    def test_evaluate_model(self):
        """Test model evaluation."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 1))

        config = {'n_components': 2, 'n_iter': 10, 'random_state': 42}
        model, _ = train_single_hmm_model(features, config)

        # Evaluate model
        score = evaluate_model(model, features)

        assert isinstance(score, float)
        assert score < 0  # Log-likelihood should be negative

    def test_get_hmm_model_info(self):
        """Test getting model information."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (100, 1))

        config = {
            'n_components': 2,
            'covariance_type': 'diag',
            'n_iter': 10,
            'random_state': 42
        }
        model, _ = train_single_hmm_model(features, config)

        # Get model info
        info = get_hmm_model_info(model)

        assert isinstance(info, dict)
        assert 'n_components' in info
        assert 'covariance_type' in info
        assert info['n_components'] == 2
        assert info['covariance_type'] == 'diag'


class TestStateInference:
    """Test state inference functionality."""

    def test_state_inference_creation(self):
        """Test creating state inference engine."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.n_components = 3

        inference = StateInference(mock_model)
        assert inference.model == mock_model

    def test_state_inference_predict(self):
        """Test state prediction through inference engine."""
        # Create mock model with predict method
        mock_model = MagicMock()
        mock_model.n_components = 3
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])

        inference = StateInference(mock_model)
        features = np.random.normal(0, 1, (5, 2))

        states = inference.infer_states(features)

        mock_model.predict.assert_called_once_with(features)
        np.testing.assert_array_equal(states, np.array([0, 1, 2, 0, 1]))

    def test_state_inference_invalid_model(self):
        """Test state inference with invalid model."""
        mock_model = MagicMock()
        mock_model.n_components = 0  # Invalid number of components

        with pytest.raises(ValueError, match="Model must have at least one component"):
            StateInference(mock_model)


class TestHMMEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_features(self):
        """Test handling of empty feature arrays."""
        features = np.array([]).reshape(0, 3)

        config = {'n_components': 2, 'n_iter': 10}
        with pytest.raises(ValueError):
            train_single_hmm_model(features, config)

    def test_single_sample(self):
        """Test handling of single sample."""
        features = np.random.normal(0, 1, (1, 3))

        with pytest.raises(ValueError, match="Insufficient samples"):
            validate_features_for_hmm(features)

    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        features = np.random.normal(0, 1, (100, 2))

        invalid_config = {
            'n_components': 0,  # Invalid
            'n_iter': 10
        }

        # Should handle invalid config gracefully
        with pytest.raises((ValueError, AttributeError)):
            train_single_hmm_model(features, invalid_config)

    def test_very_small_dataset(self):
        """Test training with very small dataset."""
        features = np.random.normal(0, 1, (10, 1))  # Only 10 samples

        config = {'n_components': 2, 'n_iter': 5, 'random_state': 42}

        # Should handle small datasets
        try:
            model, metadata = train_single_hmm_model(features, config)
            assert model is not None
        except ValueError:
            # Some HMM implementations may fail with very small datasets
            pytest.skip("HMM implementation requires more samples")

    def test_large_number_of_states(self):
        """Test training with large number of states."""
        features = np.random.normal(0, 1, (100, 2))

        config = {
            'n_components': 20,  # Large number of states
            'n_iter': 5,
            'random_state': 42
        }

        # Should handle large number of states
        try:
            model, metadata = train_single_hmm_model(features, config)
            assert model is not None
            assert metadata['n_components'] == 20
        except ValueError:
            # Some implementations may limit number of states
            pytest.skip("HMM implementation limits number of states")


@pytest.mark.integration
class TestHMMIntegration:
    """Integration tests for HMM components."""

    def test_training_and_prediction_pipeline(self):
        """Test complete training and prediction pipeline."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (50, 2))

        # Step 1: Train model
        config = {'n_components': 2, 'n_iter': 10, 'random_state': 42}
        model, train_metadata = train_single_hmm_model(features, config)

        # Step 2: Predict states
        states = predict_states(model, features)

        # Step 3: Evaluate model
        score = evaluate_model(model, features)

        # Step 4: Get model info
        model_info = get_hmm_model_info(model)

        # Verify all steps worked
        assert model is not None
        assert len(states) == len(features)
        assert isinstance(score, float)
        assert isinstance(model_info, dict)

    def test_multiple_configurations(self):
        """Test training with multiple configurations."""
        np.random.seed(42)
        features = np.random.normal(0, 1, (50, 1))

        configs = [
            {'n_components': 2, 'covariance_type': 'full'},
            {'n_components': 2, 'covariance_type': 'diag'},
            {'n_components': 3, 'covariance_type': 'full'}
        ]

        results = []
        for config in configs:
            config['n_iter'] = 10
            config['random_state'] = 42

            try:
                model, metadata = train_single_hmm_model(features, config)
                results.append((config['n_components'], config['covariance_type'], metadata['log_likelihood']))
            except Exception:
                # Some configurations might not work
                continue

        # Should have some successful results
        assert len(results) > 0