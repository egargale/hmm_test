"""
HMM State Inference Engine Module

Implements comprehensive state inference functionality for trained HMM models,
including Viterbi algorithm-based state prediction, probability extraction,
and lookahead bias prevention mechanisms for backtesting applications.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass

try:
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError as e:
    HMM_AVAILABLE = False
    hmm = None
    StandardScaler = None

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Container for HMM inference results."""
    states: np.ndarray
    probabilities: np.ndarray
    log_likelihood: float
    n_states: int
    n_samples: int
    metadata: Dict[str, Any]


@dataclass
class LaggedInferenceResult:
    """Container for lagged inference results (bias prevention)."""
    original_states: np.ndarray
    lagged_states: np.ndarray
    lag_periods: int
    lagged_probabilities: np.ndarray
    valid_periods: np.ndarray
    metadata: Dict[str, Any]


def validate_inference_inputs(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray
) -> None:
    """
    Validate inputs for HMM inference.

    Args:
        model: Trained HMM model
        scaler: Fitted feature scaler
        features: Input feature matrix

    Raises:
        ValueError: If inputs are invalid for inference
    """
    if not HMM_AVAILABLE:
        raise ImportError("HMM dependencies not available")

    # Validate model
    if not hasattr(model, 'n_components'):
        raise ValueError("Invalid HMM model - missing n_components")

    if not hasattr(model, 'n_features'):
        raise ValueError("Invalid HMM model - missing n_features")

    if not hasattr(model, 'means_'):
        raise ValueError("HMM model appears not to be trained - missing means_")

    # Validate scaler
    if not hasattr(scaler, 'mean_'):
        raise ValueError("Scaler appears not to be fitted - missing mean_")

    if not hasattr(scaler, 'scale_'):
        raise ValueError("Scaler appears not to be fitted - missing scale_")

    # Validate features
    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be a numpy array")

    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got {features.ndim}D")

    if features.shape[0] == 0:
        raise ValueError("Features cannot be empty")

    if features.shape[1] != model.n_features:
        raise ValueError(f"Feature dimension mismatch: expected {model.n_features}, got {features.shape[1]}")

    if scaler.n_features_in_ != features.shape[1]:
        raise ValueError(f"Scaler dimension mismatch: expected {scaler.n_features_in_}, got {features.shape[1]}")

    # Check for NaN or infinite values
    if np.any(~np.isfinite(features)):
        raise ValueError("Features contain NaN or infinite values")


def predict_states(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
    return_probabilities: bool = True,
    return_log_likelihood: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Predict hidden states using trained HMM model with Viterbi algorithm.

    This function implements the core HMM inference functionality by:
    1. Scaling input features using the trained scaler
    2. Applying the Viterbi algorithm for most likely state sequence
    3. Extracting posterior probabilities for state uncertainty quantification
    4. Computing model log-likelihood for confidence assessment

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        features: Input feature matrix (n_samples, n_features)
        return_probabilities: Whether to return posterior probabilities
        return_log_likelihood: Whether to return model log-likelihood

    Returns:
        If return_probabilities and return_log_likelihood:
            (states, probabilities, log_likelihood)
        If return_probabilities only:
            (states, probabilities)
        If return_log_likelihood only:
            (states, log_likelihood)
        If neither:
            states only

    Raises:
        ValueError: If inputs are invalid or inference fails
    """
    logger.info(f"Starting HMM state inference for {features.shape[0]} samples")

    # Validate inputs
    validate_inference_inputs(model, scaler, features)

    try:
        # Scale features using the fitted scaler
        logger.debug("Scaling features with fitted scaler")
        scaled_features = scaler.transform(features)

        # Predict most likely state sequence using Viterbi algorithm
        logger.debug("Running Viterbi algorithm for state prediction")
        states = model.predict(scaled_features)

        results = [states]

        # Extract posterior probabilities if requested
        if return_probabilities:
            logger.debug("Computing posterior probabilities")
            probabilities = model.predict_proba(scaled_features)

            # Validate probabilities
            if not np.allclose(np.sum(probabilities, axis=1), 1.0, rtol=1e-5):
                logger.warning("Posterior probabilities do not sum to 1 (possible numerical issues)")

            results.append(probabilities)

        # Compute log-likelihood if requested
        if return_log_likelihood:
            logger.debug("Computing model log-likelihood")
            log_likelihood = model.score(scaled_features)
            results.append(log_likelihood)

        logger.info(f"HMM inference completed: {len(states)} states predicted")

        if return_probabilities and return_log_likelihood:
            return states, probabilities, log_likelihood
        elif return_probabilities:
            return states, probabilities
        elif return_log_likelihood:
            return states, log_likelihood
        else:
            return states

    except Exception as e:
        logger.error(f"HMM inference failed: {e}")
        raise RuntimeError(f"HMM state inference failed: {e}") from e


def predict_states_comprehensive(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> InferenceResult:
    """
    Perform comprehensive HMM inference with detailed results and metadata.

    This function provides a complete inference analysis including:
    - State sequence prediction with Viterbi algorithm
    - Posterior probabilities for uncertainty quantification
    - Model log-likelihood for confidence assessment
    - Detailed statistics and metadata
    - State distribution analysis

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        features: Input feature matrix (n_samples, n_features)
        feature_names: Optional names for features

    Returns:
        InferenceResult: Comprehensive inference results with metadata

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If inference fails
    """
    logger.info("Starting comprehensive HMM inference analysis")

    # Perform basic inference
    states, probabilities, log_likelihood = predict_states(
        model, scaler, features, return_probabilities=True, return_log_likelihood=True
    )

    # Compute detailed statistics
    n_states = model.n_components
    n_samples = len(states)

    # State distribution analysis
    unique_states, state_counts = np.unique(states, return_counts=True)
    state_distribution = dict(zip(unique_states.tolist(), state_counts.tolist()))

    # State probabilities analysis
    mean_probabilities = np.mean(probabilities, axis=0)
    max_probabilities = np.max(probabilities, axis=1)
    mean_confidence = np.mean(max_probabilities)

    # Transition analysis (if we have enough samples)
    transition_matrix = model.transmat_
    expected_transitions = {}
    if n_samples > 1:
        for i in range(n_samples - 1):
            transition = (states[i], states[i + 1])
            expected_transitions[transition] = expected_transitions.get(transition, 0) + 1

    # Create metadata
    metadata = {
        'feature_names': feature_names or [f'feature_{i}' for i in range(features.shape[1])],
        'n_features': features.shape[1],
        'model_type': 'GaussianHMM',
        'covariance_type': model.covariance_type,
        'state_distribution': state_distribution,
        'state_percentages': {k: v/n_samples*100 for k, v in state_distribution.items()},
        'mean_state_probabilities': mean_probabilities.tolist(),
        'mean_prediction_confidence': float(mean_confidence),
        'transition_counts': expected_transitions,
        'expected_transitions_per_sample': sum(expected_transitions.values()) / max(n_samples - 1, 1),
        'inference_timestamp': pd.Timestamp.now().isoformat(),
        'log_likelihood_per_sample': log_likelihood / n_samples,
        'convergence_info': {
            'converged': model.monitor_.converged if hasattr(model, 'monitor_') else None,
            'n_iterations': len(model.monitor_.history) if hasattr(model, 'monitor_') else None
        }
    }

    logger.info(f"Comprehensive inference completed: {n_samples} samples, {n_states} states")
    logger.debug(f"State distribution: {state_distribution}")
    logger.debug(f"Mean prediction confidence: {mean_confidence:.3f}")

    return InferenceResult(
        states=states,
        probabilities=probabilities,
        log_likelihood=log_likelihood,
        n_states=n_states,
        n_samples=n_samples,
        metadata=metadata
    )


def get_lagged_states(
    states: np.ndarray,
    lag_periods: int = 1,
    fill_value: int = -1
) -> np.ndarray:
    """
    Apply lag to state sequence for lookahead bias prevention.

    This function is crucial for backtesting applications where decisions
    at time t should only be based on information available at time t-N.

    Args:
        states: Original state sequence
        lag_periods: Number of periods to lag (N)
        fill_value: Value to use for periods where lag cannot be applied

    Returns:
        Lagged state sequence where result[t] = states[t-lag_periods]

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(states, np.ndarray):
        raise ValueError("States must be a numpy array")

    if states.ndim != 1:
        raise ValueError("States must be 1D array")

    if lag_periods < 0:
        raise ValueError("Lag periods must be non-negative")

    n_samples = len(states)

    if lag_periods == 0:
        return states.copy()

    # Create lagged array
    lagged_states = np.full(n_samples, fill_value, dtype=states.dtype)

    # Apply lag where possible
    if lag_periods < n_samples:
        lagged_states[lag_periods:] = states[:-lag_periods]

    logger.debug(f"Applied {lag_periods}-period lag to {n_samples} states")
    logger.debug(f"Fill value used for first {lag_periods} periods: {fill_value}")

    return lagged_states


def predict_states_with_lag(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
    lag_periods: int = 1,
    fill_value: int = -1,
    return_probabilities: bool = True
) -> LaggedInferenceResult:
    """
    Predict states with automatic lag application for bias prevention.

    This function combines state prediction with lagging to ensure
    realistic backtesting without lookahead bias.

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        features: Input feature matrix (n_samples, n_features)
        lag_periods: Number of periods to lag for bias prevention
        fill_value: Value to use for periods where lag cannot be applied
        return_probabilities: Whether to return lagged probabilities

    Returns:
        LaggedInferenceResult: Results with lagged states and metadata

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If inference fails
    """
    logger.info(f"Starting lagged HMM inference with {lag_periods}-period lag")

    # Perform basic inference
    states, probabilities = predict_states(
        model, scaler, features, return_probabilities=True, return_log_likelihood=False
    )

    # Apply lag to states
    lagged_states = get_lagged_states(states, lag_periods, fill_value)

    # Create valid periods mask (True where lagged state is not fill_value)
    valid_periods = lagged_states != fill_value

    # Apply lag to probabilities if requested
    if return_probabilities:
        lagged_probabilities = np.full_like(probabilities, np.nan)
        if lag_periods < len(probabilities):
            lagged_probabilities[lag_periods:] = probabilities[:-lag_periods]
    else:
        lagged_probabilities = None

    # Create metadata
    metadata = {
        'lag_periods': lag_periods,
        'fill_value': fill_value,
        'valid_samples': int(np.sum(valid_periods)),
        'invalid_samples': int(np.sum(~valid_periods)),
        'valid_ratio': float(np.sum(valid_periods) / len(valid_periods)),
        'bias_prevention_applied': True,
        'lookahead_bias_prevented': lag_periods > 0,
        'lag_timestamp': pd.Timestamp.now().isoformat()
    }

    logger.info(f"Lagged inference completed: {metadata['valid_samples']}/{len(states)} valid predictions")

    return LaggedInferenceResult(
        original_states=states,
        lagged_states=lagged_states,
        lag_periods=lag_periods,
        lagged_probabilities=lagged_probabilities,
        valid_periods=valid_periods,
        metadata=metadata
    )


def analyze_state_stability(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Analyze state prediction stability over time windows.

    This function helps assess how stable the HMM state predictions are
    by analyzing state persistence and transition patterns.

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        features: Input feature matrix
        window_size: Size of rolling windows for stability analysis

    Returns:
        Dictionary with stability analysis results

    Raises:
        ValueError: If inputs are invalid
    """
    logger.info(f"Analyzing state stability with window size {window_size}")

    # Get basic predictions
    states = predict_states(model, scaler, features, return_probabilities=False, return_log_likelihood=False)

    n_samples = len(states)
    n_states = model.n_components

    # Calculate state persistence (how long states tend to stay the same)
    state_changes = np.diff(states) != 0
    persistence_lengths = []
    current_length = 1

    for i in range(1, len(states)):
        if states[i] == states[i-1]:
            current_length += 1
        else:
            persistence_lengths.append(current_length)
            current_length = 1
    persistence_lengths.append(current_length)  # Add final segment

    # Calculate rolling state consistency
    if n_samples >= window_size:
        rolling_consistency = []
        for i in range(window_size, n_samples):
            window_states = states[i-window_size:i]
            most_common_state = np.bincount(window_states).argmax()
            consistency = np.sum(window_states == most_common_state) / window_size
            rolling_consistency.append(consistency)

        avg_consistency = np.mean(rolling_consistency)
        min_consistency = np.min(rolling_consistency)
    else:
        avg_consistency = np.nan
        min_consistency = np.nan
        rolling_consistency = []

    # Transition frequency analysis
    transition_counts = np.zeros((n_states, n_states))
    for i in range(1, n_samples):
        transition_counts[states[i-1], states[i]] += 1

    transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
    transition_probs = np.nan_to_num(transition_probs)  # Handle zero rows

    analysis = {
        'n_samples': n_samples,
        'n_states': n_states,
        'window_size': window_size,
        'state_changes': int(np.sum(state_changes)),
        'change_rate': float(np.sum(state_changes) / (n_samples - 1)),
        'avg_persistence_length': float(np.mean(persistence_lengths)),
        'max_persistence_length': int(np.max(persistence_lengths)),
        'min_persistence_length': int(np.min(persistence_lengths)),
        'avg_rolling_consistency': avg_consistency,
        'min_rolling_consistency': min_consistency,
        'rolling_consistency_series': rolling_consistency,
        'transition_counts': transition_counts.tolist(),
        'transition_probabilities': transition_probs.tolist(),
        'self_transition_probabilities': np.diag(transition_probs).tolist(),
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    logger.info(f"State stability analysis completed: change_rate={analysis['change_rate']:.3f}, "
                f"avg_persistence={analysis['avg_persistence_length']:.1f}")

    return analysis


def create_inference_dataframe(
    model: hmm.GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    timestamps: Optional[pd.DatetimeIndex] = None,
    lag_periods: int = 0
) -> pd.DataFrame:
    """
    Create a comprehensive pandas DataFrame with inference results.

    This function is useful for analysis and visualization of HMM inference results.

    Args:
        model: Trained hmmlearn GaussianHMM model
        scaler: Fitted StandardScaler used during training
        features: Input feature matrix
        feature_names: Optional names for features
        timestamps: Optional datetime index for the results
        lag_periods: Number of periods to lag for bias prevention

    Returns:
        pandas.DataFrame with comprehensive inference results

    Raises:
        ValueError: If inputs are invalid
    """
    logger.info("Creating comprehensive inference DataFrame")

    # Perform inference
    if lag_periods > 0:
        lagged_result = predict_states_with_lag(
            model, scaler, features, lag_periods=lag_periods, return_probabilities=True
        )
        states = lagged_result.lagged_states
        probabilities = lagged_result.lagged_probabilities
        valid_mask = lagged_result.valid_periods
    else:
        comprehensive_result = predict_states_comprehensive(model, scaler, features, feature_names)
        states = comprehensive_result.states
        probabilities = comprehensive_result.probabilities
        valid_mask = np.ones(len(states), dtype=bool)

    n_samples = len(states)
    n_states = model.n_components

    # Create base DataFrame
    if timestamps is not None:
        if len(timestamps) != n_samples:
            raise ValueError(f"Timestamps length {len(timestamps)} doesn't match samples {n_samples}")
        index = timestamps
    else:
        index = pd.RangeIndex(0, n_samples)

    df = pd.DataFrame(index=index)

    # Add state information
    df['state'] = states
    df['valid_prediction'] = valid_mask

    # Add probability information
    if probabilities is not None:
        for i in range(n_states):
            df[f'prob_state_{i}'] = probabilities[:, i]
        df['max_probability'] = np.max(probabilities, axis=1)
        df['predicted_state_confidence'] = df['max_probability']

    # Add metadata columns
    df['n_states'] = n_states
    df['model_covariance_type'] = model.covariance_type
    df['lag_periods_applied'] = lag_periods

    # Add feature names if provided
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            if i < features.shape[1]:
                df[f'feature_{name}'] = features[:, i]

    logger.info(f"Created inference DataFrame with {len(df)} rows and {len(df.columns)} columns")

    return df