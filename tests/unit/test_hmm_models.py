"""
Unit tests for HMM models and training.

Tests the Hidden Markov Model implementations, training algorithms,
and state inference components.
"""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hmm_models.base import BaseHMMModel
from hmm_models.factory import HMMModelFactory
from hmm_models.gaussian_hmm import GaussianHMMModel
from model_training import hmm_trainer, inference_engine, model_persistence


class TestBaseHMMModel:
    """Test base HMM model functionality."""

    def test_base_hmm_model_interface(self):
        """Test that base HMM model defines correct interface."""
        # The base class should define the interface that all HMM models must implement
        required_methods = ["fit", "predict", "score", "get_model_info"]

        # Test that these are abstract methods (can't instantiate base class)
        with pytest.raises(TypeError):
            BaseHMMModel()


class TestGaussianHMMModel:
    """Test Gaussian HMM model implementation."""

    def test_gaussian_hmm_creation(self):
        """Test creating GaussianHMMModel with default parameters."""
        model = GaussianHMMModel()
        assert model.n_components == 3
        assert model.covariance_type == "full"
        assert model.n_iter == 100
        assert model.tol == 1e-4

    def test_gaussian_hmm_custom_parameters(self):
        """Test creating GaussianHMMModel with custom parameters."""
        model = GaussianHMMModel(
            n_components=4,
            covariance_type="diag",
            n_iter=200,
            tol=1e-6,
            random_state=42,
        )
        assert model.n_components == 4
        assert model.covariance_type == "diag"
        assert model.n_iter == 200
        assert model.tol == 1e-6
        assert model.random_state == 42

    def test_gaussian_hmm_fit_valid_data(self, sample_features):
        """Test fitting GaussianHMMModel with valid data."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Use close prices as features
        features = sample_features["close"].values.reshape(-1, 1)

        model.fit(features)
        assert model.is_fitted is True

    def test_gaussian_hmm_predict(self, sample_features):
        """Test state prediction with GaussianHMMModel."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Fit model
        features = sample_features["close"].values.reshape(-1, 1)
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
        features = sample_features["close"].values.reshape(-1, 1)
        model.fit(features)

        # Score model
        score = model.score(features)
        assert isinstance(score, float)
        assert score < 0  # Log-likelihood should be negative

    def test_gaussian_hmm_get_model_info(self, sample_features):
        """Test getting model information."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Fit model
        features = sample_features["close"].values.reshape(-1, 1)
        model.fit(features)

        # Get model info
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert "n_components" in info
        assert "covariance_type" in info
        assert "n_iter" in info
        assert "converged" in info

    def test_gaussian_hmm_predict_before_fit(self):
        """Test prediction before model is fitted."""
        model = GaussianHMMModel()
        features = np.random.normal(0, 1, (100, 1))

        with pytest.warns(UserWarning):
            model.predict(features)

    def test_gaussian_hmm_invalid_features(self):
        """Test fitting with invalid features."""
        model = GaussianHMMModel()

        # Empty features
        with pytest.warns(UserWarning):
            model.fit(np.array([]))

        # Features with NaN values
        features_with_nan = np.array([[1.0], [np.nan], [2.0]])
        model.fit(features_with_nan)  # Should handle NaN values gracefully

    def test_gaussian_hmm_multidimensional_features(self, sample_features):
        """Test fitting with multidimensional features."""
        model = GaussianHMMModel(n_components=2, random_state=42)

        # Use multiple features
        feature_columns = ["close", "returns", "volatility_14"]
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
        model = HMMModelFactory.create_model("gaussian")
        assert isinstance(model, GaussianHMMModel)

    def test_factory_create_with_config(self):
        """Test creating model with configuration."""
        config = {"n_components": 4, "covariance_type": "diag", "random_state": 123}
        model = HMMModelFactory.create_model("gaussian", **config)
        assert model.n_components == 4
        assert model.covariance_type == "diag"
        assert model.random_state == 123

    def test_factory_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.warns(UserWarning):
            HMMModelFactory.create_model("invalid_type")

    def test_factory_get_available_models(self):
        """Test getting list of available models."""
        available = HMMModelFactory.get_available_models()
        assert isinstance(available, list)
        assert "gaussian" in available


class TestHMMTrainer:
    """Test HMM trainer functionality."""

    def test_train_single_model(self, sample_features):
        """Test training a single HMM model."""
        features = sample_features["close"].values.reshape(-1, 1)
        config = {
            "n_components": 2,
            "covariance_type": "full",
            "n_iter": 100,
            "random_state": 42,
        }

        model, metadata = hmm_trainer.train_single_hmm_model(features, config)

        assert model is not None
        assert metadata is not None
        assert "n_components" in metadata
        assert "log_likelihood" in metadata

    def test_train_model_with_validation(self, sample_features):
        """Test training model with feature validation."""
        features = sample_features["close"].values.reshape(-1, 1)
        config = {"n_components": 2, "n_iter": 100, "random_state": 42}

        # Should validate features without raising exceptions
        model, metadata = hmm_trainer.train_model(
            features, config=config, validate_features=True
        )

        assert model is not None
        assert metadata is not None

    def test_validate_features_for_hmm(self):
        """Test feature validation function."""
        # Valid features
        valid_features = np.random.normal(0, 1, (100, 3))
        hmm_trainer.validate_features_for_hmm(valid_features)  # Should not raise

        # Invalid features (too few samples)
        with pytest.warns(UserWarning):
            hmm_trainer.validate_features_for_hmm(np.array([[1, 2, 3]]))

        # Invalid features (zero variance)
        with pytest.warns(UserWarning):
            hmm_trainer.validate_features_for_hmm(np.array([[1, 2, 3]] * 10))

        # Invalid features (NaN values)
        features_with_nan = np.array([[1.0, 2.0, np.nan]] * 10)
        with pytest.warns(UserWarning):
            hmm_trainer.validate_features_for_hmm(features_with_nan)

    def test_numerical_stability(self, sample_features):
        """Test numerical stability handling."""
        features = sample_features["close"].values.reshape(-1, 1)

        # Add numerical stability epsilon
        stable_features = hmm_trainer.add_numerical_stability_epsilon(features)
        assert np.all(stable_features >= 1e-8)  # Default epsilon

    def test_model_comparison(self, sample_features):
        """Test comparing multiple models."""
        features = sample_features["close"].values.reshape(-1, 1)

        # Train models with different configurations
        configs = [
            {"n_components": 2, "n_iter": 50, "random_state": 42},
            {"n_components": 3, "n_iter": 50, "random_state": 42},
            {"n_components": 4, "n_iter": 50, "random_state": 42},
        ]

        results = []
        for config in configs:
            model, metadata = hmm_trainer.train_single_hmm_model(features, config)
            results.append((config["n_components"], metadata["log_likelihood"]))

        # Should get different log-likelihoods for different configurations
        likelihoods = [result[1] for result in results]
        assert len(set(likelihoods)) > 1  # Should have different values


class TestStateInference:
    """Test state inference functionality."""

    def test_state_inference_functions(self, sample_features):
        """Test state inference functions."""
        # Create mock model and scaler for testing
        mock_model = MagicMock()
        mock_model.n_components = 2
        mock_model.n_features = 1
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.normal(0, 1, (5, 1))

        features = np.random.normal(0, 1, (5, 1))

        # Test predict_states function
        states = inference_engine.predict_states(mock_model, mock_scaler, features)
        assert isinstance(states, np.ndarray)
        assert len(states) == 5

        # Test comprehensive prediction
        result = inference_engine.predict_states_comprehensive(
            mock_model, mock_scaler, features, ["close"]
        )
        assert hasattr(result, "states")
        assert hasattr(result, "probabilities")
        assert hasattr(result, "metadata")

    def test_lagged_inference(self):
        """Test lagged inference functionality."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.n_components = 2
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.normal(0, 1, (5, 1))

        features = np.random.normal(0, 1, (5, 1))

        # Test lagged inference
        lagged_result = inference_engine.predict_states_with_lag(
            mock_model, mock_scaler, features, lag_periods=1
        )
        assert hasattr(lagged_result, "original_states")
        assert hasattr(lagged_result, "lagged_states")
        assert hasattr(lagged_result, "valid_periods")

    def test_state_stability_analysis(self, sample_features):
        """Test state stability analysis."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.n_components = 2
        mock_model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.normal(0, 1, (50, 1))

        features = sample_features["close"].values.reshape(-1, 1)

        # Mock predict_states for stability analysis
        with patch("model_training.inference_engine.predict_states") as mock_predict:
            mock_predict.return_value = np.random.choice([0, 1], size=len(features))

            analysis = inference_engine.analyze_state_stability(
                mock_model, mock_scaler, features, window_size=10
            )

            assert isinstance(analysis, dict)
            assert "n_samples" in analysis
            assert "n_states" in analysis
            assert "change_rate" in analysis
            assert "avg_persistence_length" in analysis

    def test_inference_dataframe_creation(self):
        """Test creating inference DataFrame."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.n_components = 2
        mock_model.covariance_type = "full"
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.normal(0, 1, (5, 1))

        features = np.random.normal(0, 1, (5, 1))
        timestamps = pd.date_range("2024-01-01", periods=5, freq="D")

        # Test DataFrame creation
        df = inference_engine.create_inference_dataframe(
            mock_model, mock_scaler, features, ["close"], timestamps
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "state" in df.columns
        assert "prob_state_0" in df.columns
        assert "prob_state_1" in df.columns


class TestModelPersistence:
    """Test model persistence functionality."""

    def test_save_load_model(self, sample_features, temp_dir):
        """Test saving and loading model."""
        # Train model
        features = sample_features["close"].values.reshape(-1, 1)
        config = {"n_components": 2, "n_iter": 50, "random_state": 42}
        model, metadata = hmm_trainer.train_single_hmm_model(features, config)

        # Create mock scaler
        mock_scaler = MagicMock()
        mock_scaler.mean_ = np.array([0.0])
        mock_scaler.scale_ = np.array([1.0])

        # Save model
        model_path = temp_dir / "test_model.pkl"
        model_persistence.save_model(model, mock_scaler, config, str(model_path))

        # Load model
        loaded_model, loaded_scaler, loaded_config = model_persistence.load_model(
            str(model_path)
        )

        # Verify loaded model
        assert loaded_model is not None
        assert loaded_scaler is not None
        assert loaded_config is not None
        assert loaded_config["n_components"] == config["n_components"]

    def test_integrity_validation(self, sample_features, temp_dir):
        """Test model integrity validation."""
        # Train model
        features = sample_features["close"].values.reshape(-1, 1)
        config = {"n_components": 2, "n_iter": 50, "random_state": 42}
        model, metadata = hmm_trainer.train_single_hmm_model(features, config)

        # Create mock scaler
        mock_scaler = MagicMock()
        mock_scaler.mean_ = np.array([0.0])
        mock_scaler.scale_ = np.array([1.0])

        # Save model
        model_path = temp_dir / "test_model.pkl"
        model_persistence.save_model(model, mock_scaler, config, str(model_path))

        # Validate integrity
        is_valid = model_persistence.validate_model_integrity(str(model_path))
        assert is_valid is True

    def test_corrupted_model_handling(self, temp_dir):
        """Test handling of corrupted model files."""
        # Create corrupted file
        corrupted_path = temp_dir / "corrupted_model.pkl"
        corrupted_path.write_text("This is not a valid pickle file")

        # Should handle corruption gracefully
        with pytest.raises((pickle.PickleError, EOFError, AttributeError)):
            model_persistence.load_model(str(corrupted_path))

    def test_metadata_functionality(self, sample_features, temp_dir):
        """Test model metadata functionality."""
        # Train model
        features = sample_features["close"].values.reshape(-1, 1)
        config = {
            "n_components": 2,
            "n_iter": 50,
            "random_state": 42,
            "training_date": "2024-01-01",
            "model_version": "1.0",
            "notes": "Test model for unit testing",
        }
        model, metadata = hmm_trainer.train_single_hmm_model(features, config)

        # Create mock scaler
        mock_scaler = MagicMock()
        mock_scaler.mean_ = np.array([0.0])
        mock_scaler.scale_ = np.array([1.0])

        # Save with enhanced metadata
        model_path = temp_dir / "enhanced_model.pkl"
        model_persistence.save_model(model, mock_scaler, config, str(model_path))

        # Load and verify metadata
        loaded_model, loaded_scaler, loaded_config = model_persistence.load_model(
            str(model_path)
        )
        assert loaded_config["training_date"] == "2024-01-01"
        assert loaded_config["model_version"] == "1.0"
        assert loaded_config["notes"] == "Test model for unit testing"


@pytest.mark.integration
class TestHMMIntegration:
    """Integration tests for HMM components."""

    def test_end_to_end_hmm_workflow(self, sample_features):
        """Test complete HMM workflow."""
        # Step 1: Train model
        features = sample_features["close"].values.reshape(-1, 1)
        config = {"n_components": 3, "n_iter": 50, "random_state": 42}
        model, metadata = hmm_trainer.train_single_hmm_model(features, config)

        # Step 2: Create mock scaler for inference
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = features

        # Step 3: State inference
        states = inference_engine.predict_states(model, mock_scaler, features)

        # Verify all components work together
        assert model is not None
        assert len(states) == len(features)

    def test_different_feature_sets(self, sample_features):
        """Test HMM with different feature sets."""
        # Test with single feature
        single_feature = sample_features["close"].values.reshape(-1, 1)
        config1 = {"n_components": 2, "n_iter": 50, "random_state": 42}
        model1, metadata1 = hmm_trainer.train_single_hmm_model(single_feature, config1)

        # Test with multiple features
        multi_features = (
            sample_features[["close", "returns", "volatility_14"]].dropna().values
        )
        config2 = {"n_components": 2, "n_iter": 50, "random_state": 42}
        model2, metadata2 = hmm_trainer.train_single_hmm_model(multi_features, config2)

        # Both models should be trained successfully
        assert model1 is not None
        assert model2 is not None
        assert metadata1["n_components"] == metadata2["n_components"]
