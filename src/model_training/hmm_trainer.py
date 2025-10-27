"""
HMM Model Training Module

Implements the core Hidden Markov Model training functionality using hmmlearn,
including feature scaling, convergence monitoring, multiple restarts, and numerical stability.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from hmmlearn import hmm
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None
    StandardScaler = None
    ConvergenceWarning = None

from utils import HMMConfig, get_logger

logger = get_logger(__name__)


@dataclass
class HMMTrainingResult:
    """Container for HMM training results."""
    model: Optional[hmm.GaussianHMM]
    scaler: Optional[StandardScaler]
    score: float
    n_restarts_completed: int
    n_successful_restarts: int
    convergence_info: Dict[str, Any]
    warnings: list


def validate_features_for_hmm(features: np.ndarray) -> None:
    """
    Validate input features for HMM training.

    Args:
        features: Input feature matrix

    Raises:
        ValueError: If features are invalid for HMM training
    """
    if not HMM_AVAILABLE:
        raise ImportError("HMM dependencies not available. Install with: uv add hmmlearn scikit-learn")

    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be a numpy array")

    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got {features.ndim}D")

    if features.shape[0] == 0:
        raise ValueError("Features cannot be empty")

    if features.shape[1] == 0:
        raise ValueError("Features must have at least one column")

    # Check for NaN or infinite values
    if np.any(~np.isfinite(features)):
        raise ValueError("Features contain NaN or infinite values")

    # Check for zero-variance features
    feature_variances = np.var(features, axis=0)
    zero_var_features = np.where(feature_variances < 1e-10)[0]

    if len(zero_var_features) > 0:
        logger.warning(f"Features {zero_var_features} have zero or near-zero variance, which may cause numerical issues")


def validate_hmm_config(config: HMMConfig) -> None:
    """
    Validate HMM configuration parameters.

    Args:
        config: HMM configuration

    Raises:
        ValueError: If configuration parameters are invalid
    """
    if config.n_states < 1:
        raise ValueError("Number of states must be at least 1")

    if config.n_states > 20:
        logger.warning(f"Large number of states ({config.n_states}) may cause numerical instability")

    if config.max_iter < 1:
        raise ValueError("Maximum iterations must be at least 1")

    if config.tol <= 0:
        raise ValueError("Tolerance must be positive")

    if config.num_restarts < 1:
        raise ValueError("Number of restarts must be at least 1")


def create_feature_scaler(scaler_type: str = "standard") -> StandardScaler:
    """
    Create a feature scaler for HMM training.

    Args:
        scaler_type: Type of scaler ('standard' or 'minmax')

    Returns:
        Configured feature scaler
    """
    if not HMM_AVAILABLE:
        raise ImportError("Scikit-learn not available")

    if scaler_type == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")


def train_single_hmm_model(
    features: np.ndarray,
    config: HMMConfig,
    random_state: int,
    scaler: Optional[StandardScaler] = None
) -> Tuple[Optional[hmm.GaussianHMM], float, Dict[str, Any]]:
    """
    Train a single HMM model with specified parameters.

    Args:
        features: Scaled feature matrix
        config: HMM configuration
        random_state: Random state for this training run
        scaler: Optional pre-fitted scaler

    Returns:
        Tuple of (trained_model, log_likelihood_score, training_info)
    """
    training_info = {
        'random_state': random_state,
        'converged': False,
        'n_iterations': 0,
        'final_log_likelihood': -np.inf,
        'warnings': []
    }

    try:
        # Create HMM model
        model = hmm.GaussianHMM(
            n_components=config.n_states,
            covariance_type=config.covariance_type,
            n_iter=config.max_iter,
            tol=config.tol,
            random_state=random_state,
            init_params='stmc',  # Initialize start prob, trans prob, emission prob
            params='stmc',       # Update start prob, trans prob, emission prob
            verbose=False
        )

        # Capture convergence warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)

            # Fit the model
            model.fit(features)

            # Check for convergence warnings
            convergence_warnings = [warning for warning in w if issubclass(warning.category, ConvergenceWarning)]
            if convergence_warnings:
                training_info['warnings'].extend([str(w.message) for w in convergence_warnings])
                logger.warning(f"HMM training convergence warnings for random_state {random_state}")

            # Get final log likelihood
            final_score = model.score(features)
            training_info['final_log_likelihood'] = final_score

            # Check convergence status
            training_info['converged'] = model.monitor_.converged
            training_info['n_iterations'] = len(model.monitor_.history)

            logger.debug(f"HMM training completed: random_state={random_state}, "
                        f"score={final_score:.4f}, converged={training_info['converged']}, "
                        f"iterations={training_info['n_iterations']}")

            return model, final_score, training_info

    except Exception as e:
        error_msg = f"HMM training failed for random_state {random_state}: {str(e)}"
        logger.warning(error_msg)
        training_info['warnings'].append(error_msg)
        return None, -np.inf, training_info


def add_numerical_stability_epsilon(features: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Add small epsilon to features for numerical stability.

    Args:
        features: Feature matrix
        epsilon: Small value to add

    Returns:
        Stabilized feature matrix
    """
    # Add epsilon to avoid zero variance issues
    stabilized_features = features.copy()

    # Add epsilon to features that have very small variance
    feature_std = np.std(features, axis=0)
    small_variance_mask = feature_std < epsilon

    if np.any(small_variance_mask):
        logger.debug(f"Adding epsilon to {np.sum(small_variance_mask)} low-variance features")
        stabilized_features[:, small_variance_mask] += epsilon

    return stabilized_features


def train_model(
    features: np.ndarray,
    config: HMMConfig,
    scaler_type: str = "standard",
    enable_numerical_stability: bool = True,
    return_all_results: bool = False
) -> Tuple[hmm.GaussianHMM, StandardScaler, float]:
    """
    Train a Hidden Markov Model with multiple restarts and best model selection.

    This function implements the core HMM training pipeline with:
    - Feature scaling using StandardScaler
    - Multiple restarts to avoid local optima
    - Convergence monitoring and selection of best model
    - Numerical stability checks and handling

    Args:
        features: Input feature matrix (n_samples, n_features)
        config: HMM configuration parameters
        scaler_type: Type of feature scaler ('standard' or 'minmax')
        enable_numerical_stability: Whether to apply numerical stability measures
        return_all_results: Whether to return detailed training results

    Returns:
        Tuple of (best_model, fitted_scaler, best_score)

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If HMM fails to converge after all restarts
    """
    logger.info(f"Starting HMM model training: n_states={config.n_states}, "
                f"covariance_type={config.covariance_type}, n_restarts={config.num_restarts}")

    # Validate inputs
    validate_features_for_hmm(features)
    validate_hmm_config(config)

    # Create and fit scaler
    logger.info("Creating and fitting feature scaler")
    scaler = create_feature_scaler(scaler_type)

    try:
        scaled_features = scaler.fit_transform(features)
        logger.debug(f"Features scaled: mean={np.mean(scaled_features, axis=0)[:3]}...")
    except Exception as e:
        raise RuntimeError(f"Feature scaling failed: {e}")

    # Apply numerical stability measures if enabled
    if enable_numerical_stability:
        scaled_features = add_numerical_stability_epsilon(scaled_features)
        logger.debug("Applied numerical stability epsilon")

    # Initialize tracking variables
    best_model = None
    best_score = -np.inf
    best_training_info = None
    all_results = []

    # Train multiple models with different random states
    logger.info(f"Training {config.num_restarts} HMM models with different random states")

    for restart_idx in range(config.num_restarts):
        random_state = config.random_state + restart_idx
        logger.debug(f"Training restart {restart_idx + 1}/{config.num_restarts} "
                    f"with random_state {random_state}")

        model, score, training_info = train_single_hmm_model(
            scaled_features, config, random_state, scaler
        )

        all_results.append({
            'restart_idx': restart_idx,
            'model': model,
            'score': score,
            'training_info': training_info
        })

        # Update best model if this one is better
        if model is not None and score > best_score:
            best_model = model
            best_score = score
            best_training_info = training_info
            logger.debug(f"New best model found: score={score:.4f}, restart={restart_idx}")

    # Check if we found a valid model
    if best_model is None:
        error_msg = f"HMM failed to converge after {config.num_restarts} restarts"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Prepare final results
    n_successful = sum(1 for result in all_results if result['model'] is not None)
    convergence_info = {
        'best_restart': best_training_info['random_state'],
        'best_converged': best_training_info['converged'],
        'best_iterations': best_training_info['n_iterations'],
        'successful_restarts': n_successful,
        'total_restarts': config.num_restarts,
        'success_rate': n_successful / config.num_restarts
    }

    logger.info("HMM training completed successfully:")
    logger.info(f"  - Best score: {best_score:.4f}")
    logger.info(f"  - Successful restarts: {n_successful}/{config.num_restarts}")
    logger.info(f"  - Best model converged: {best_training_info['converged']}")
    logger.info(f"  - Best iterations: {best_training_info['n_iterations']}")

    if return_all_results:
        result = HMMTrainingResult(
            model=best_model,
            scaler=scaler,
            score=best_score,
            n_restarts_completed=config.num_restarts,
            n_successful_restarts=n_successful,
            convergence_info=convergence_info,
            warnings=[w for result in all_results for w in result['training_info']['warnings']]
        )
        return result

    return best_model, scaler, best_score


def predict_states(
    model: hmm.GaussianHMM,
    features: np.ndarray,
    scaler: StandardScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict hidden states using trained HMM model.

    Args:
        model: Trained HMM model
        features: Input features (unscaled)
        scaler: Fitted feature scaler

    Returns:
        Tuple of (state_sequence, posterior_probabilities)
    """
    # Scale features
    scaled_features = scaler.transform(features)

    # Predict most likely states
    state_sequence = model.predict(scaled_features)

    # Get posterior probabilities
    posterior_probabilities = model.predict_proba(scaled_features)

    return state_sequence, posterior_probabilities


def evaluate_model(
    model: hmm.GaussianHMM,
    features: np.ndarray,
    scaler: StandardScaler
) -> Dict[str, float]:
    """
    Evaluate trained HMM model performance.

    Args:
        model: Trained HMM model
        features: Input features (unscaled)
        scaler: Fitted feature scaler

    Returns:
        Dictionary of evaluation metrics
    """
    scaled_features = scaler.transform(features)

    # Log likelihood
    log_likelihood = model.score(scaled_features)

    # Get model information
    n_states = model.n_components
    n_features = model.n_features

    # Transition matrix entropy (measure of model complexity)
    transition_entropy = -np.sum(model.transmat_ * np.log(model.transmat_ + 1e-10))

    return {
        'log_likelihood': log_likelihood,
        'n_states': n_states,
        'n_features': n_features,
        'transition_entropy': transition_entropy,
        'converged': model.monitor_.converged,
        'n_iterations': len(model.monitor_.history)
    }


def get_hmm_model_info(model: hmm.GaussianHMM) -> Dict[str, Any]:
    """
    Get detailed information about a trained HMM model.

    Args:
        model: Trained HMM model

    Returns:
        Dictionary with model information
    """
    if not hasattr(model, 'n_components'):
        return {'error': 'Model not properly trained'}

    return {
        'n_states': model.n_components,
        'n_features': model.n_features,
        'covariance_type': model.covariance_type,
        'converged': model.monitor_.converged,
        'n_iterations': len(model.monitor_.history) if hasattr(model, 'monitor_') else None,
        'start_probabilities': model.startprob_.tolist(),
        'transition_matrix': model.transmat_.tolist(),
        'means': model.means_.tolist(),
        'covariances': [cov.tolist() for cov in model.covars_] if hasattr(model.covars_, 'tolist') else str(model.covars_)
    }
