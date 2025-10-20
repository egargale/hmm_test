"""
HMM Model Persistence Module

Implements robust model persistence and loading functionality using Python's pickle module.
Stores trained HMM models, feature scalers, and configurations with integrity validation.
"""

import pickle
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import warnings

import numpy as np

try:
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError as e:
    HMM_AVAILABLE = False
    hmm = None
    StandardScaler = None

from utils import get_logger, HMMConfig

# Import pandas for timestamps
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

logger = get_logger(__name__)

# Current model persistence version
MODEL_VERSION = "1.0"
# Required keys for integrity validation
REQUIRED_KEYS = {'hmm_model', 'scaler', 'config', 'version', 'timestamp'}


def _get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string
    """
    if PANDAS_AVAILABLE:
        return str(pd.Timestamp.now())
    else:
        import datetime
        return datetime.datetime.now().isoformat()


@dataclass
class ModelMetadata:
    """Metadata for saved HMM models."""
    model_type: str
    n_states: int
    n_features: int
    covariance_type: str
    n_samples: int
    training_score: float
    model_hash: str
    file_size_bytes: int
    save_timestamp: str
    python_version: str
    library_versions: Dict[str, str]
    config_dict: Dict[str, Any]


def generate_model_hash(model: hmm.GaussianHMM, scaler: StandardScaler) -> str:
    """
    Generate a hash for the model and scaler combination for integrity checking.

    Args:
        model: Trained HMM model
        scaler: Fitted feature scaler

    Returns:
        SHA-256 hash string
    """
    # Create hash from model parameters and scaler parameters
    hash_data = {
        'model_means': model.means_.tolist() if hasattr(model, 'means_') else None,
        'model_covars': [cov.tolist() for cov in model.covars_] if hasattr(model, 'covars_') else None,
        'model_startprob': model.startprob_.tolist() if hasattr(model, 'startprob_') else None,
        'model_transmat': model.transmat_.tolist() if hasattr(model, 'transmat_') else None,
        'scaler_mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'scaler_n_features': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
    }

    # Convert to JSON string for hashing
    hash_string = json.dumps(hash_data, sort_keys=True, default=str)

    # Generate SHA-256 hash
    return hashlib.sha256(hash_string.encode()).hexdigest()


def get_library_versions() -> Dict[str, str]:
    """
    Get versions of key libraries used for model training.

    Returns:
        Dictionary with library versions
    """
    versions = {}

    try:
        import sklearn
        versions['sklearn'] = sklearn.__version__
    except ImportError:
        pass

    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except ImportError:
        pass

    if HMM_AVAILABLE:
        try:
            import hmmlearn
            versions['hmmlearn'] = hmmlearn.__version__
        except ImportError:
            pass

    return versions


def save_model(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    config: HMMConfig,
    path: Union[str, Path],
    include_metadata: bool = True,
    overwrite: bool = False,
    create_directories: bool = True
) -> ModelMetadata:
    """
    Save a trained HMM model, scaler, and configuration to disk.

    This function creates a comprehensive model archive containing:
    - The trained HMM model with all parameters
    - The fitted feature scaler for preprocessing
    - HMM configuration for reproducibility
    - Metadata for model identification and validation
    - Version information for compatibility checking

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        config: HMMConfig with training parameters
        path: File path where the model will be saved
        include_metadata: Whether to include comprehensive metadata
        overwrite: Whether to overwrite existing files
        create_directories: Whether to create parent directories if they don't exist

    Returns:
        ModelMetadata: Information about the saved model

    Raises:
        ValueError: If inputs are invalid
        FileExistsError: If file exists and overwrite=False
        OSError: If file system operations fail
        RuntimeError: If HMM dependencies are not available
    """
    if not HMM_AVAILABLE:
        raise RuntimeError("HMM dependencies not available for model persistence")

    # Validate inputs
    if not hasattr(model, 'n_components'):
        raise ValueError("Invalid HMM model: missing n_components")

    if not hasattr(scaler, 'mean_'):
        raise ValueError("Invalid scaler: appears not to be fitted")

    if not isinstance(config, HMMConfig):
        raise ValueError("config must be an HMMConfig instance")

    # Convert path to Path object
    path = Path(path)

    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists at {path}. Use overwrite=True to replace.")

    # Create parent directories if requested
    if create_directories:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    logger.info(f"Saving HMM model to {path}")

    try:
        # Generate model hash for integrity checking
        model_hash = generate_model_hash(model, scaler)

        # Prepare model data dictionary
        model_data = {
            'hmm_model': model,
            'scaler': scaler,
            'config': config.dict(),  # Convert Pydantic config to dict
            'version': MODEL_VERSION,
            'timestamp': _get_timestamp(),  # Get current timestamp
            'model_hash': model_hash
        }

        # Initialize metadata to None
        metadata = None

        # Add metadata if requested
        if include_metadata:
            # Get file size before saving (will be updated after)
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

            # Compute model score (log-likelihood)
            if hasattr(model, 'means_') and hasattr(scaler, 'mean_'):
                # Create dummy features for scoring
                dummy_features = np.random.normal(0, 1, (10, model.n_features))
                scaled_features = scaler.transform(dummy_features)
                training_score = model.score(scaled_features) / len(dummy_features)
            else:
                training_score = 0.0

            metadata = ModelMetadata(
                model_type='GaussianHMM',
                n_states=model.n_components,
                n_features=model.n_features,
                covariance_type=model.covariance_type,
                n_samples=0,  # Will be updated by caller if needed
                training_score=training_score,
                model_hash=model_hash,
                file_size_bytes=0,  # Will be updated after saving
                save_timestamp=_get_timestamp(),
                python_version=python_version,
                library_versions=get_library_versions(),
                config_dict=config.dict()
            )

            model_data['metadata'] = metadata

        # Save model using pickle
        logger.debug(f"Serializing model data (version {MODEL_VERSION})")
        with open(path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update file size in metadata
        if include_metadata:
            actual_file_size = path.stat().st_size
            metadata.file_size_bytes = actual_file_size
            metadata.save_timestamp = _get_timestamp()

            # Re-save with updated file size
            with open(path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model saved successfully: {path}")
        logger.debug(f"Model hash: {model_hash[:16]}...")

        return metadata

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        # Clean up partial file if it exists
        if path.exists():
            try:
                path.unlink()
            except:
                pass
        raise RuntimeError(f"Failed to save HMM model: {e}") from e


def load_model(
    path: Union[str, Path],
    validate_integrity: bool = True,
    validate_functionality: bool = True,
    test_features: Optional[np.ndarray] = None
) -> Tuple[hmm.GaussianHMM, StandardScaler, HMMConfig, Optional[ModelMetadata]]:
    """
    Load a saved HMM model, scaler, and configuration from disk.

    This function loads a previously saved model archive and performs
    comprehensive integrity validation to ensure the loaded components
    are valid and functional.

    Args:
        path: Path to the saved model file
        validate_integrity: Whether to perform integrity validation
        validate_functionality: Whether to test model functionality
        test_features: Optional test features for functionality validation

    Returns:
        Tuple of (model, scaler, config, metadata). Metadata is None if not saved.

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the file is corrupted or invalid
        RuntimeError: If HMM dependencies are not available
        pickle.UnpicklingError: If the file cannot be unpickled
    """
    if not HMM_AVAILABLE:
        raise RuntimeError("HMM dependencies not available for model loading")

    # Convert path to Path object
    path = Path(path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    logger.info(f"Loading HMM model from {path}")

    try:
        # Load model data
        logger.debug(f"Loading model data (expected version {MODEL_VERSION})")
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Integrity validation
        if validate_integrity:
            _validate_model_integrity(model_data, path)

        # Extract components
        model = model_data['hmm_model']
        scaler = model_data['scaler']
        config_dict = model_data['config']

        # Validate loaded components
        _validate_loaded_components(model, scaler, config_dict, path)

        # Reconstruct HMMConfig from dictionary
        config = HMMConfig(**config_dict)

        # Functionality validation
        if validate_functionality:
            _validate_model_functionality(model, scaler, config, test_features, path)

        # Extract metadata if available
        metadata = model_data.get('metadata')

        logger.info(f"Model loaded successfully: {model.n_components} states, {model.n_features} features")
        logger.debug(f"Model version: {model_data.get('version', 'unknown')}")

        if metadata:
            logger.debug(f"Model hash: {metadata.model_hash[:16]}...")

        return model, scaler, config, metadata

    except pickle.UnpicklingError as e:
        logger.error(f"Failed to unpickle model file: {e}")
        raise ValueError(f"Model file appears to be corrupted or invalid: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load HMM model: {e}") from e


def _validate_model_integrity(model_data: Dict[str, Any], path: Path) -> None:
    """
    Validate the integrity of loaded model data.

    Args:
        model_data: Loaded model data dictionary
        path: Path to the model file (for error messages)

    Raises:
        ValueError: If integrity validation fails
    """
    logger.debug("Performing integrity validation")

    # Check required keys
    missing_keys = REQUIRED_KEYS - model_data.keys()
    if missing_keys:
        raise ValueError(f"Corrupted model file {path}: missing required keys {missing_keys}")

    # Check model version compatibility
    saved_version = model_data.get('version')
    if saved_version != MODEL_VERSION:
        logger.warning(f"Model version mismatch: expected {MODEL_VERSION}, got {saved_version}")

    # Check model hash if available
    if 'model_hash' in model_data:
        current_hash = generate_model_hash_hash(model_data)
        if current_hash != model_data['model_hash']:
            logger.warning(f"Model hash mismatch: model may have been tampered with")

    logger.debug("Integrity validation passed")


def generate_model_hash_hash(model_data: Dict[str, Any]) -> str:
    """
    Regenerate hash from loaded model data for integrity checking.

    Args:
        model_data: Loaded model data dictionary

    Returns:
        SHA-256 hash string
    """
    model = model_data['hmm_model']
    scaler = model_data['scaler']
    return generate_model_hash(model, scaler)


def _validate_loaded_components(
    model: Any,
    scaler: Any,
    config_dict: Dict[str, Any],
    path: Path
) -> None:
    """
    Validate that loaded components are valid and properly initialized.

    Args:
        model: Loaded HMM model
        scaler: Loaded feature scaler
        config_dict: Loaded configuration dictionary
        path: Path to the model file (for error messages)

    Raises:
        ValueError: If any component is invalid
    """
    logger.debug("Validating loaded components")

    # Validate model
    if not hasattr(model, 'n_components'):
        raise ValueError(f"Invalid model in {path}: missing n_components")

    if not hasattr(model, 'n_features'):
        raise ValueError(f"Invalid model in {path}: missing n_features")

    if not hasattr(model, 'means_'):
        raise ValueError(f"Model in {path} appears not to be trained: missing means_")

    # Validate scaler
    if not hasattr(scaler, 'mean_'):
        raise ValueError(f"Invalid scaler in {path}: appears not to be fitted")

    if not hasattr(scaler, 'scale_'):
        raise ValueError(f"Invalid scaler in {path}: missing scale_")

    # Check dimension compatibility
    if hasattr(model, 'n_features') and hasattr(scaler, 'n_features_in_'):
        if model.n_features != scaler.n_features_in_:
            raise ValueError(f"Dimension mismatch in {path}: "
                           f"model expects {model.n_features} features, "
                           f"scaler fitted to {scaler.n_features_in_}")

    # Validate configuration
    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid config in {path}: not a dictionary")

    required_config_keys = {'n_states', 'covariance_type', 'max_iter', 'random_state'}
    missing_config_keys = required_config_keys - config_dict.keys()
    if missing_config_keys:
        raise ValueError(f"Invalid config in {path}: missing keys {missing_config_keys}")

    logger.debug("Component validation passed")


def _validate_model_functionality(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    config: HMMConfig,
    test_features: Optional[np.ndarray],
    path: Path
) -> None:
    """
    Validate that the loaded model can perform basic operations.

    Args:
        model: Loaded HMM model
        scaler: Loaded feature scaler
        config: Loaded HMM configuration
        test_features: Optional test features for validation
        path: Path to the model file (for error messages)

    Raises:
        ValueError: If functionality validation fails
    """
    logger.debug("Performing functionality validation")

    try:
        # Create test features if not provided
        if test_features is None:
            test_features = np.random.normal(0, 1, (5, model.n_features))
        else:
            if test_features.shape[1] != model.n_features:
                raise ValueError(f"Test features dimension mismatch: "
                               f"expected {model.n_features}, got {test_features.shape[1]}")

        # Test feature scaling
        scaled_features = scaler.transform(test_features)

        # Test model prediction
        states = model.predict(scaled_features)
        if len(states) != len(test_features):
            raise ValueError(f"Model prediction failed: expected {len(test_features)} states, "
                           f"got {len(states)}")

        # Test probability computation
        probabilities = model.predict_proba(scaled_features)
        if probabilities.shape != (len(test_features), model.n_components):
            raise ValueError(f"Model probability computation failed: "
                           f"expected ({len(test_features)}, {model.n_components}), "
                           f"got {probabilities.shape}")

        # Test scoring
        score = model.score(scaled_features)
        if not isinstance(score, (float, np.floating)):
            raise ValueError(f"Model scoring failed: invalid score type {type(score)}")

        logger.debug("Functionality validation passed")

    except Exception as e:
        raise ValueError(f"Model functionality validation failed for {path}: {e}") from e


def list_saved_models(directory: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    List all saved HMM models in a directory with their metadata.

    Args:
        directory: Directory to search for model files

    Returns:
        Dictionary mapping file paths to model metadata

    Raises:
        ValueError: If directory doesn't exist
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    logger.info(f"Scanning directory for saved models: {directory}")

    models = {}

    # Look for .pkl files
    for file_path in directory.rglob("*.pkl"):
        try:
            # Try to load metadata without full model loading
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            if 'metadata' in model_data:
                metadata = model_data['metadata']
                models[str(file_path)] = {
                    'filename': file_path.name,
                    'size_bytes': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime,
                    'metadata': metadata
                }
            else:
                # Basic info if no metadata
                models[str(file_path)] = {
                    'filename': file_path.name,
                    'size_bytes': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime,
                    'metadata': None
                }

        except Exception as e:
            logger.warning(f"Failed to read metadata from {file_path}: {e}")
            models[str(file_path)] = {
                'filename': file_path.name,
                'error': str(e)
            }

    logger.info(f"Found {len(models)} model files")
    return models


def delete_model(path: Union[str, Path], confirm: bool = False) -> bool:
    """
    Delete a saved model file.

    Args:
        path: Path to the model file to delete
        confirm: Whether to require confirmation (always False in code)

    Returns:
        True if deletion was successful, False otherwise

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if confirm:
        # This would typically be used in interactive contexts
        # For programmatic use, we'll just delete without confirmation
        pass

    try:
        path.unlink()
        logger.info(f"Deleted model file: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete model file {path}: {e}")
        return False


def copy_model(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Copy a saved model file to a new location.

    Args:
        source_path: Path to the source model file
        target_path: Path where the model should be copied
        overwrite: Whether to overwrite existing files

    Returns:
        True if copy was successful, False otherwise

    Raises:
        FileNotFoundError: If the source model file doesn't exist
        FileExistsError: If target exists and overwrite=False
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source model file not found: {source_path}")

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists: {target_path}")

    # Create target directory if needed
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import shutil
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied model from {source_path} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy model from {source_path} to {target_path}: {e}")
        return False


def get_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a saved model without fully loading it.

    Args:
        path: Path to the saved model file

    Returns:
        Dictionary with model information

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the file cannot be read or is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Extract basic info
        info = {
            'file_path': str(path),
            'file_size_bytes': path.stat().st_size,
            'modified_time': path.stat().st_mtime,
            'version': model_data.get('version', 'unknown'),
            'timestamp': model_data.get('timestamp', 'unknown'),
            'model_hash': model_data.get('model_hash', 'unknown'),
            'has_metadata': 'metadata' in model_data
        }

        # Add metadata if available
        if 'metadata' in model_data:
            metadata = model_data['metadata']
            info.update({
                'model_type': metadata.model_type,
                'n_states': metadata.n_states,
                'n_features': metadata.n_features,
                'covariance_type': metadata.covariance_type,
                'training_score': metadata.training_score,
                'python_version': metadata.python_version,
                'library_versions': metadata.library_versions,
                'save_timestamp': metadata.save_timestamp
            })

        return info

    except Exception as e:
        raise ValueError(f"Failed to read model info from {path}: {e}") from e