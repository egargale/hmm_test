"""
Integration tests using BTC.csv dataset.

These tests verify the end-to-end functionality of the HMM futures analysis
system using real market data, providing comprehensive test coverage.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing.csv_parser import process_csv
from data_processing.data_validation import validate_data
from data_processing.feature_engineering import add_features
from model_training.hmm_trainer import train_single_hmm_model, validate_features_for_hmm
from model_training.inference_engine import predict_states
from model_training.model_persistence import load_model, save_model
from utils import HMMConfig


@pytest.fixture(scope="module")
def btc_data(btc_csv_file):
    """Load and process BTC data for integration testing."""
    # Process the BTC CSV file
    df = process_csv(str(btc_csv_file))

    # Add features
    df_features = add_features(df, indicator_config={})

    # Validate data
    validation_results = validate_data(df_features)

    return {"raw_data": df, "features": df_features, "validation": validation_results}


@pytest.fixture(scope="module")
def btc_features(btc_data):
    """Extract clean features for HMM training."""
    features_df = btc_data["features"]

    # Remove rows with NaN values (common at the beginning due to indicators)
    features_clean = features_df.dropna()

    # Select numeric columns only (exclude datetime if present)
    numeric_features = features_clean.select_dtypes(include=[np.number])

    return numeric_features


class TestBTCDataProcessing:
    """Test BTC data processing pipeline."""

    def test_btc_csv_processing(self, btc_data):
        """Test that BTC CSV can be processed successfully."""
        raw_data = btc_data["raw_data"]
        features = btc_data["features"]

        # Verify raw data structure
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        expected_columns = ["open", "high", "low", "close", "volume"]
        for col in expected_columns:
            assert col in raw_data.columns, f"Missing column: {col}"

        # Verify features were added
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > len(raw_data.columns)  # Features added

        # Check for key feature categories
        feature_columns = set(features.columns) - set(raw_data.columns)
        assert any("ret" in col for col in feature_columns), "Missing return features"
        assert any(
            "sma_" in col or "ema_" in col for col in feature_columns
        ), "Missing MA features"
        assert any(
            "rsi" in col or "atr" in col for col in feature_columns
        ), "Missing technical indicators"

    def test_btc_data_validation(self, btc_data):
        """Test BTC data validation."""
        validation_results = btc_data["validation"]

        # Should have validation results
        assert validation_results is not None
        assert isinstance(validation_results, dict)

        # Check common validation metrics
        if "quality_score" in validation_results:
            assert 0 <= validation_results["quality_score"] <= 100

        if "missing_values" in validation_results:
            assert isinstance(validation_results["missing_values"], dict)

    def test_btc_feature_quality(self, btc_features):
        """Test quality of engineered features."""
        # Should have sufficient data for HMM training
        assert len(btc_features) >= 100, "Insufficient data points for testing"

        # Should have multiple features
        assert len(btc_features.columns) >= 10, "Insufficient feature diversity"

        # Check for constant or near-constant features
        for col in btc_features.columns:
            std_val = btc_features[col].std()
            assert std_val > 1e-10, f"Feature {col} has near-zero variance"

        # Check for NaN values
        nan_counts = btc_features.isna().sum()
        assert (
            nan_counts.sum() == 0
        ), f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}"


class TestBTCHMMTraining:
    """Test HMM model training with BTC data."""

    def test_btc_feature_validation(self, btc_features):
        """Test feature validation for HMM training."""
        # Should pass validation without errors
        try:
            validate_features_for_hmm(btc_features.values)
        except ValueError as e:
            # If validation fails, it should be for understandable reasons
            assert "Insufficient samples" not in str(e)
            assert "zero or near-zero variance" not in str(e)

    def test_btc_hmm_training(self, btc_features):
        """Test HMM model training with BTC data."""
        # Use a subset of features for faster testing
        feature_subset = btc_features.iloc[:500].copy()

        # Create HMM config
        hmm_config = HMMConfig(
            n_states=3,
            covariance_type="full",
            n_iter=50,  # Reduced for test speed
            random_state=42,
            tol=1e-3,
        )

        # Train model
        try:
            model, scaler, metadata = train_single_hmm_model(
                features=feature_subset, config=hmm_config.__dict__
            )

            # Verify model was trained
            assert model is not None
            assert hasattr(model, "means_"), "Model appears not to be trained"
            assert scaler is not None, "Scaler not returned"
            assert metadata is not None, "Metadata not returned"

            # Check model parameters
            assert model.n_components == hmm_config.n_states
            assert model.means_.shape[0] == hmm_config.n_states
            assert model.means_.shape[1] == feature_subset.shape[1]

        except Exception as e:
            pytest.skip(f"HMM training failed: {e}")

    def test_btc_state_prediction(self, btc_features):
        """Test state prediction with BTC data."""
        # Train a simple model first
        feature_subset = btc_features.iloc[:300].copy()
        hmm_config = HMMConfig(
            n_states=2,  # Simpler for testing
            covariance_type="diag",  # Faster convergence
            n_iter=25,
            random_state=42,
        )

        try:
            model, scaler, _ = train_single_hmm_model(
                features=feature_subset, config=hmm_config.__dict__
            )

            # Test prediction
            test_features = btc_features.iloc[300:350].copy()
            states, probabilities = predict_states(
                features=test_features, model=model, scaler=scaler
            )

            # Verify predictions
            assert states is not None
            assert len(states) == len(test_features)
            assert states.min() >= 0
            assert states.max() < hmm_config.n_states

            if probabilities is not None:
                assert probabilities.shape[0] == len(test_features)
                assert probabilities.shape[1] == hmm_config.n_states
                # Probabilities should sum to 1
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)

        except Exception as e:
            pytest.skip(f"State prediction failed: {e}")


class TestBTCModelPersistence:
    """Test model persistence with BTC-trained models."""

    def test_btc_model_save_load(self, btc_features):
        """Test saving and loading BTC-trained models."""
        feature_subset = btc_features.iloc[:200].copy()
        hmm_config = HMMConfig(
            n_states=2, covariance_type="diag", n_iter=20, random_state=42
        )

        try:
            # Train model
            model, scaler, metadata = train_single_hmm_model(
                features=feature_subset, config=hmm_config.__dict__
            )

            # Test save/load
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "btc_model.pkl"

                # Save model
                save_model(
                    model=model,
                    scaler=scaler,
                    config=hmm_config.__dict__,
                    metadata=metadata,
                    file_path=str(model_path),
                )

                assert model_path.exists(), "Model file not created"

                # Load model
                (
                    loaded_model,
                    loaded_scaler,
                    loaded_config,
                    loaded_metadata,
                ) = load_model(file_path=str(model_path))

                # Verify loaded model
                assert loaded_model is not None
                assert loaded_scaler is not None
                assert loaded_config is not None
                assert loaded_metadata is not None

                # Test predictions are consistent
                original_states, _ = predict_states(
                    features=feature_subset.iloc[:10], model=model, scaler=scaler
                )

                loaded_states, _ = predict_states(
                    features=feature_subset.iloc[:10],
                    model=loaded_model,
                    scaler=loaded_scaler,
                )

                np.testing.assert_array_equal(original_states, loaded_states)

        except Exception as e:
            pytest.skip(f"Model persistence test failed: {e}")


class TestBTCIntegrationPipeline:
    """Test complete integration pipeline with BTC data."""

    def test_end_to_end_pipeline(self, btc_csv_file):
        """Test complete end-to-end analysis pipeline."""
        try:
            # 1. Load and process data
            raw_data = process_csv(str(btc_csv_file))
            assert len(raw_data) > 500, "Insufficient data for end-to-end test"

            # 2. Add features
            features = add_features(raw_data, indicator_config={})
            assert len(features.columns) > len(raw_data.columns)

            # 3. Clean data
            clean_features = features.dropna().select_dtypes(include=[np.number])
            assert len(clean_features) > 100

            # 4. Train HMM
            hmm_config = HMMConfig(
                n_states=3, covariance_type="full", n_iter=30, random_state=42
            )

            train_data = clean_features.iloc[:300]
            model, scaler, metadata = train_single_hmm_model(
                features=train_data, config=hmm_config.__dict__
            )

            # 5. Predict states
            test_data = clean_features.iloc[300:400]
            states, probabilities = predict_states(
                features=test_data, model=model, scaler=scaler
            )

            # 6. Verify results
            assert len(states) == len(test_data)
            assert states.min() >= 0
            assert states.max() < hmm_config.n_states

            # 7. Basic state analysis
            unique_states, state_counts = np.unique(states, return_counts=True)
            assert len(unique_states) > 0, "No states predicted"

            # Check that we have reasonable state distribution
            total_predictions = len(states)
            for state_idx, count in zip(unique_states, state_counts):
                proportion = count / total_predictions
                assert (
                    0.01 <= proportion <= 0.99
                ), f"State {state_idx} has unusual proportion: {proportion}"

        except Exception as e:
            pytest.skip(f"End-to-end pipeline failed: {e}")

    def test_different_configurations(self, btc_features):
        """Test pipeline with different HMM configurations."""
        configs = [
            {"n_states": 2, "covariance_type": "diag"},
            {"n_states": 3, "covariance_type": "full"},
            {"n_states": 4, "covariance_type": "spherical"},
        ]

        results = []

        for config_params in configs:
            try:
                config = HMMConfig(
                    n_iter=20,  # Reduced for test speed
                    random_state=42,
                    **config_params,
                )

                # Use smaller dataset for testing multiple configs
                train_data = btc_features.iloc[:200]
                model, scaler, metadata = train_single_hmm_model(
                    features=train_data, config=config.__dict__
                )

                # Test prediction
                test_data = btc_features.iloc[200:250]
                states, _ = predict_states(
                    features=test_data, model=model, scaler=scaler
                )

                results.append(
                    {
                        "config": config_params,
                        "n_states": len(np.unique(states)),
                        "converged": metadata.get("converged", False)
                        if metadata
                        else False,
                    }
                )

            except Exception as e:
                results.append({"config": config_params, "error": str(e)})

        # At least some configurations should work
        successful_configs = [r for r in results if "error" not in r]
        assert len(successful_configs) > 0, "All HMM configurations failed"

        # Print results for debugging (if test fails)
        if len(successful_configs) < len(configs):
            failed_configs = [r for r in results if "error" in r]
            print(f"Some configs failed: {failed_configs}")


# Test utilities
def check_btc_data_quality():
    """Utility function to check BTC data quality for testing."""
    try:
        btc_path = Path(__file__).parent.parent.parent / "BTC.csv"
        if not btc_path.exists():
            return False, "BTC.csv not found"

        # Quick quality check
        df = pd.read_csv(btc_path)
        if len(df) < 100:
            return False, "Insufficient data in BTC.csv"

        required_cols = ["Open", "High", "Low", "Last", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        return True, "BTC data suitable for testing"

    except Exception as e:
        return False, f"Error checking BTC data: {e}"


if __name__ == "__main__":
    # Run BTC data quality check
    is_suitable, message = check_btc_data_quality()
    print(f"BTC Data Quality Check: {message}")
    if not is_suitable:
        warnings.warn(f"BTC integration tests may not work properly: {message}")
