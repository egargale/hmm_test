"""
Unit tests for HMM models and training.

Tests the Hidden Markov Model implementations, training algorithms,
and state inference components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import Dict, Any
import pickle
from pathlib import Path

from hmm_models.base import BaseHMMModel
from hmm_models.gaussian_hmm import GaussianHMMModel
from hmm_models.factory import HMMModelFactory
from model_training.hmm_trainer import train_model, validate_features_for_hmm
from model_training.inference_engine import StateInference
from model_training.model_persistence import ModelPersistence


class TestBaseHMMModel:
    """Test base HMM model functionality."""

    def test_base_hmm_model_interface(self):
        """Test that base HMM model defines correct interface."""
        # The base class should define the interface that all HMM models must implement
        required_methods = ['fit', 'predict', 'score', 'get_model_info']

        # Test that these are abstract methods (can't instantiate base class)
        with pytest.raises(TypeError):
            BaseHMMModel()


class TestGaussianHMMModel:
    """Test Gaussian HMM model implementation."""

    def test_gaussian_hmm_creation(self):
        """Test creating GaussianHMMModel with default parameters."""
        model = GaussianHMMModel()
        assert model.n_components == 3
        assert model.covariance_type == 'full'
        assert model.n_iter == 100
        assert model.tol == 1e-4

    def test_gaussian_hmm_custom_parameters(self):
        """Test creating GaussianHMMModel with custom parameters."""
        model = GaussianHMMModel(
            n_components=4,
            covariance_type='diag',
            n_iter=200,
            tol=1e-6,
            random_state=42
        )
        assert model.n_components == 4
        assert model.covariance_type == 'diag'
        assert model.n_iter == 200
        assert model.tol == 1e-6
        assert model.random_state == 42

    def test_gaussian_hmm_fit_valid_data(self, sample_features):
        """Test fitting GaussianHMMModel with valid data."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Use close prices as features
        features = sample_features['close'].values.reshape(-1, 1)

        model.fit(features)
        assert model.is_fitted is True

    def test_gaussian_hmm_predict(self, sample_features):
        """Test state prediction with GaussianHMMModel."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Fit model
        features = sample_features['close'].values.reshape(-1, 1)
        model.fit(features)

        # Predict states
        states = model.predict(features)
        assert isinstance(states, np.ndarray)
        assert len(states) == len(features)
        assert states.dtype == np.int64
        assert np.all(states >= 0)
        assert np.all(states < model.n_components)

    def test_gaussian_hmm_score(self, sample_features):
        """Test scoring with GaussianHMMModel."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Fit model
        features = sample_features['close'].values.reshape(-1, 1)
        model.fit(features)

        # Score model
        score = model.score(features)
        assert isinstance(score, float)
        assert score < 0  # Log-likelihood should be negative

    def test_gaussian_hmm_get_model_info(self, sample_features):
        """Test getting model information."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Fit model
        features = sample_features['close'].values.reshape(-1, 1)
        model.fit(features)

        # Get model info
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert 'n_components' in info
        assert 'covariance_type' in info
        assert 'n_iter' in info
        assert 'converged' in info

    def test_gaussian_hmm_predict_before_fit(self):
        """Test prediction before model is fitted."""
        model = GaussianHMMModel()
        features = np.random.normal(0, 1, (100, 1))

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(features)

    def test_gaussian_hmm_invalid_features(self):
        """Test fitting with invalid features."""
        model = GaussianHMMModel()

        # Empty features
        with pytest.raises(ValueError):
            model.fit(np.array([]))

        # Features with NaN values
        features_with_nan = np.array([[1.0], [np.nan], [2.0]])
        model.fit(features_with_nan)  # Should handle NaN values gracefully

    def test_gaussian_hmm_multidimensional_features(self, sample_features):
        """Test fitting with multidimensional features."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Use multiple features
        feature_columns = ['close', 'returns', 'volatility_14']
        features = sample_features[feature_columns].dropna().values

        model.fit(features)
        assert model.is_fitted is True

        # Predict and verify
        states = model.predict(features)
        assert len(states) == len(features)


class TestHMMModelFactory:
    """Test HMM model factory functionality."""

    def test_factory_create_gaussian_hmm(self):
        """Test creating Gaussian HMM through factory."""
        model = HMMModelFactory.create_model('gaussian')
        assert isinstance(model, GaussianHMMModel)

    def test_factory_create_with_config(self):
        """Test creating model with configuration."""
        config = {
            'n_components': 4,
            'covariance_type': 'diag',
            'random_state': 123
        }
        model = HMMModelFactory.create_model('gaussian', **config)
        assert model.n_components == 4
        assert model.covariance_type == 'diag'
        assert model.random_state == 123

    def test_factory_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            HMMModelFactory.create_model('invalid_type')

    def test_factory_get_available_models(self):
        """Test getting list of available models."""
        available = HMMModelFactory.get_available_models()
        assert isinstance(available, list)
        assert 'gaussian' in available


class TestHMMTrainer:
    """Test HMM trainer functionality."""

    def test_trainer_creation(self):
        """Test creating HMM trainer."""
        trainer = HMMTrainer()
        assert trainer is not None

    def test_trainer_with_config(self, hmm_config):
        """Test creating trainer with configuration."""
        trainer = HMMTrainer(**hmm_config)
        assert trainer.n_components == hmm_config['n_components']
        assert trainer.covariance_type == hmm_config['covariance_type']

    def test_train_single_model(self, sample_features):
        """Test training a single HMM model."""
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)

        model, metadata = trainer.train_single_model(features)

        assert model is not None
        assert isinstance(metadata, dict)
        assert 'n_components' in metadata
        assert 'log_likelihood' in metadata

    def test_train_with_restarts(self, sample_features):
        """Test training with multiple restarts."""
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)

        best_model, best_metadata = trainer.train_with_restarts(
            features, n_restarts=5
        )

        assert best_model is not None
        assert best_metadata is not None
        assert 'n_restarts' in best_metadata
        assert best_metadata['n_restarts'] == 5

    def test_train_model_with_validation(self, sample_features):
        """Test training model with feature validation."""
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)

        # Should validate features without raising exceptions
        model, metadata = trainer.train_model(features, validate_features=True)

        assert model is not None
        assert metadata is not None

    def test_validate_features_for_hmm(self):
        """Test feature validation function."""
        # Valid features
        valid_features = np.random.normal(0, 1, (100, 3))
        validate_features_for_hmm(valid_features)  # Should not raise

        # Invalid features (too few samples)
        with pytest.raises(ValueError, match="Insufficient samples"):
            validate_features_for_hmm(np.array([[1, 2, 3]]))

        # Invalid features (zero variance)
        with pytest.raises(ValueError, match="Zero variance detected"):
            validate_features_for_hmm(np.array([[1, 2, 3]] * 10))

        # Invalid features (NaN values)
        features_with_nan = np.array([[1.0, 2.0, np.nan]] * 10)
        validate_features_for_hmm(features_with_nan)  # Should handle gracefully

    def test_trainer_numerical_stability(self, sample_features):
        """Test trainer numerical stability handling."""
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)

        # Add numerical stability epsilon
        stable_features = trainer.add_numerical_stability_epsilon(features)
        assert np.all(stable_features >= trainer.epsilon)

    def test_trainer_model_comparison(self, sample_features):
        """Test comparing multiple models."""
        trainer = HMMTrainer(random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)

        # Train models with different configurations
        configs = [
            {'n_components': 2},
            {'n_components': 3},
            {'n_components': 4}
        ]

        results = []
        for config in configs:
            model_trainer = HMMTrainer(**config, random_state=42)
            model, metadata = model_trainer.train_single_model(features)
            results.append((config['n_components'], metadata['log_likelihood']))

        # Should get different log-likelihoods for different configurations
        likelihoods = [result[1] for result in results]
        assert len(set(likelihoods)) > 1  # Should have different values


class TestStateInference:
    """Test state inference functionality."""

    def test_state_inference_creation(self):
        """Test creating state inference engine."""
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = np.random.normal(0, 1, (100, 1))
        model, _ = trainer.train_single_model(features)

        inference = StateInference(model)
        assert inference.model is model

    def test_state_prediction(self, sample_features):
        """Test state prediction."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, _ = trainer.train_single_model(features)

        # Create inference engine
        inference = StateInference(model)

        # Predict states
        states = inference.infer_states(features)
        assert isinstance(states, np.ndarray)
        assert len(states) == len(features)

    def test_state_probabilities(self, sample_features):
        """Test getting state probabilities."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, _ = trainer.train_single_model(features)

        # Create inference engine
        inference = StateInference(model)

        # Get state probabilities
        probabilities = inference.get_state_probabilities(features)
        assert probabilities.shape[0] == len(features)
        assert probabilities.shape[1] == model.n_components
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_state_transition_analysis(self, sample_features):
        """Test state transition analysis."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, _ = trainer.train_single_model(features)

        # Create inference engine
        inference = StateInference(model)

        # Get transition analysis
        states = inference.infer_states(features)
        transitions = inference.analyze_transitions(states)

        assert isinstance(transitions, dict)
        assert 'transition_matrix' in transitions
        assert 'state_counts' in transitions

    def test_state_stability_analysis(self, sample_features):
        """Test state stability analysis."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, _ = trainer.train_single_model(features)

        # Create inference engine
        inference = StateInference(model)

        # Get stability analysis
        states = inference.infer_states(features)
        stability = inference.analyze_stability(states)

        assert isinstance(stability, dict)
        assert 'mean_durations' in stability
        assert 'variance_durations' in stability


class TestModelPersistence:
    """Test model persistence functionality."""

    def test_save_load_model(self, sample_features, temp_dir):
        """Test saving and loading model."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, metadata = trainer.train_single_model(features)

        # Save model
        persistence = ModelPersistence()
        model_path = temp_dir / "test_model.pkl"
        persistence.save_model(model, metadata, str(model_path))

        # Load model
        loaded_model, loaded_metadata = persistence.load_model(str(model_path))

        # Verify loaded model
        assert loaded_model is not None
        assert loaded_metadata is not None
        assert loaded_metadata['n_components'] == metadata['n_components']

    def test_integrity_validation(self, sample_features, temp_dir):
        """Test model integrity validation."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, metadata = trainer.train_single_model(features)

        # Save model
        persistence = ModelPersistence()
        model_path = temp_dir / "test_model.pkl"
        persistence.save_model(model, metadata, str(model_path))

        # Validate integrity
        is_valid = persistence.validate_model_integrity(str(model_path))
        assert is_valid is True

    def test_corrupted_model_handling(self, temp_dir):
        """Test handling of corrupted model files."""
        persistence = ModelPersistence()

        # Create corrupted file
        corrupted_path = temp_dir / "corrupted_model.pkl"
        corrupted_path.write_text("This is not a valid pickle file")

        # Should handle corruption gracefully
        with pytest.raises((pickle.PickleError, EOFError, AttributeError)):
            persistence.load_model(str(corrupted_path))

    def test_metadata_functionality(self, sample_features, temp_dir):
        """Test model metadata functionality."""
        # Train model
        trainer = HMMTrainer(n_components=2, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, metadata = trainer.train_single_model(features)

        # Add additional metadata
        enhanced_metadata = {
            **metadata,
            'training_date': '2024-01-01',
            'model_version': '1.0',
            'notes': 'Test model for unit testing'
        }

        # Save with enhanced metadata
        persistence = ModelPersistence()
        model_path = temp_dir / "enhanced_model.pkl"
        persistence.save_model(model, enhanced_metadata, str(model_path))

        # Load and verify metadata
        loaded_model, loaded_metadata = persistence.load_model(str(model_path))
        assert loaded_metadata['training_date'] == '2024-01-01'
        assert loaded_metadata['model_version'] == '1.0'
        assert loaded_metadata['notes'] == 'Test model for unit testing'


@pytest.mark.integration
class TestHMMIntegration:
    """Integration tests for HMM components."""

    def test_end_to_end_hmm_workflow(self, sample_features):
        """Test complete HMM workflow."""
        # Step 1: Train model
        trainer = HMMTrainer(n_components=3, random_state=42)
        features = sample_features['close'].values.reshape(-1, 1)
        model, metadata = trainer.train_with_restarts(features, n_restarts=3)

        # Step 2: State inference
        inference = StateInference(model)
        states = inference.infer_states(features)
        probabilities = inference.get_state_probabilities(features)

        # Step 3: Analysis
        transitions = inference.analyze_transitions(states)
        stability = inference.analyze_stability(states)

        # Verify all components work together
        assert model is not None
        assert len(states) == len(features)
        assert probabilities.shape[0] == len(features)
        assert len(transitions) > 0
        assert len(stability) > 0

    def test_different_feature_sets(self, sample_features):
        """Test HMM with different feature sets."""
        trainer = HMMTrainer(n_components=2, random_state=42)

        # Test with single feature
        single_feature = sample_features['close'].values.reshape(-1, 1)
        model1, metadata1 = trainer.train_single_model(single_feature)

        # Test with multiple features
        multi_features = sample_features[['close', 'returns', 'volatility_14']].dropna().values
        model2, metadata2 = trainer.train_single_model(multi_features)

        # Both models should be trained successfully
        assert model1 is not None
        assert model2 is not None
        assert metadata1['n_components'] == metadata2['n_components']