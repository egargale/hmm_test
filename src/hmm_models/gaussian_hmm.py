"""
Gaussian HMM Model Implementation

Implements Hidden Markov Model with Gaussian emission distributions
for financial regime detection and state modeling.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    hmm = None

from utils import get_logger
from .base import BaseHMMModel

logger = get_logger(__name__)


class GaussianHMMModel(BaseHMMModel):
    """
    Hidden Markov Model with Gaussian emission distributions.

    Suitable for financial time series where returns and other continuous
    features can be modeled as Gaussian distributions within each regime.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: Optional[int] = None,
        n_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        implementation: str = "log",
        params: str = "stmc",  # s=startprob, t=transmat, m=means, c=covars
        init_params: str = "stmc"
    ):
        """
        Initialize Gaussian HMM model.

        Args:
            n_components: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ("full", "tied", "diag", "spherical")
            random_state: Random seed for reproducibility
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            verbose: Whether to print training progress
            implementation: Implementation type ("log" or "scaling")
            params: Parameters to train
            init_params: Parameters to initialize
        """
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=n_iter,  # Keep max_iter for compatibility with base class
            tol=tol,
            verbose=verbose
        )

        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is not installed. Install with: pip install hmmlearn")

        self.implementation = implementation
        self.params = params
        self.init_params = init_params
        self.n_iter = n_iter  # Store n_iter for hmmlearn

    def _create_model(self) -> "hmm.GaussianHMM":
        """Create the Gaussian HMM model."""
        model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
            implementation=self.implementation,
            params=self.params,
            init_params=self.init_params
        )
        return model

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prepare features for Gaussian HMM training.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Prepared feature array
        """
        if isinstance(X, pd.DataFrame):
            # Handle missing values
            X_clean = X.dropna()
            if len(X_clean) < len(X):
                logger.warning(f"Removed {len(X) - len(X_clean)} rows with missing values")

            # Convert to numpy and ensure float64
            X_array = X_clean.values.astype(np.float64)

            # Optional: Standardize features for better convergence
            # This is a common practice for Gaussian HMM
            X_array = (X_array - np.mean(X_array, axis=0)) / (np.std(X_array, axis=0) + 1e-8)

        else:
            X_array = X.astype(np.float64)
            # Handle NaN values
            mask = ~np.isnan(X_array).any(axis=1)
            X_array = X_array[mask]

            # Standardize
            X_array = (X_array - np.mean(X_array, axis=0)) / (np.std(X_array, axis=0) + 1e-8)

        return X_array

    def _extract_parameters(self) -> Dict[str, Any]:
        """
        Extract model parameters for analysis.

        Returns:
            Dictionary with model parameters
        """
        if not self.is_fitted_:
            return {}

        params = {
            'startprob_': self.model_.startprob_.copy(),
            'transmat_': self.model_.transmat_.copy(),
            'means_': self.model_.means_.copy(),
            'covars_': self.model_.covars_.copy()
        }

        # Add parameter analysis
        params['parameter_analysis'] = {
            'startprob_entropy': -np.sum(self.model_.startprob_ * np.log(self.model_.startprob_ + 1e-10)),
            'transmat_entropy': np.mean([
                -np.sum(row * np.log(row + 1e-10)) for row in self.model_.transmat_
            ]),
            'mean_condition_numbers': [
                np.linalg.cond(cov) for cov in self.model_.covars_
            ],
            'covariance_determinants': [
                np.linalg.det(cov) for cov in self.model_.covars_
            ],
            'covariance_traces': [np.trace(cov) for cov in self.model_.covars_]
        }

        return params

    def _count_emission_parameters(self) -> int:
        """Count Gaussian emission distribution parameters."""
        if not self.is_fitted_:
            return 0

        n_features = self.model_.means_.shape[1]

        if self.covariance_type == "full":
            # Means: n_components * n_features
            # Covariances: n_components * n_features * (n_features + 1) / 2
            n_means = self.n_components * n_features
            n_covars = self.n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            # Means: n_components * n_features
            # Variances: n_components * n_features
            n_means = self.n_components * n_features
            n_covars = self.n_components * n_features
        elif self.covariance_type == "tied":
            # Means: n_components * n_features
            # Covariance: n_features * (n_features + 1) / 2
            n_means = self.n_components * n_features
            n_covars = n_features * (n_features + 1) // 2
        elif self.covariance_type == "spherical":
            # Means: n_components * n_features
            # Variances: n_components
            n_means = self.n_components * n_features
            n_covars = self.n_components
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")

        return n_means + n_covars

    def get_state_descriptions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Generate human-readable descriptions of each state.

        Args:
            X: Training data used for context

        Returns:
            Dictionary with state descriptions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting state descriptions")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get state assignments
        states = self.predict(X_array)

        # Extract parameters
        params = self.get_model_parameters()

        descriptions = {}
        for state in range(self.n_components):
            state_mask = states == state
            X_array[state_mask]
            state_prob = np.mean(state_mask)

            description = {
                'state_id': state,
                'frequency': state_prob,
                'mean_values': params['means_'][state],
                'covariance_matrix': params['covars_'][state],
                'sample_count': np.sum(state_mask),
                'volatility_level': np.sqrt(np.trace(params['covars_'][state])),
                'dominant_features': self._get_dominant_features(params['means_'][state])
            }

            # Add financial interpretation if feature names are available
            if self.feature_names_:
                description['financial_interpretation'] = self._interpret_state_financially(
                    params['means_'][state],
                    params['covars_'][state],
                    self.feature_names_
                )

            descriptions[state] = description

        return descriptions

    def _get_dominant_features(self, state_mean: np.ndarray) -> Dict[str, float]:
        """
        Identify dominant features in a state based on mean values.

        Args:
            state_mean: Mean values for the state

        Returns:
            Dictionary with dominant features
        """
        if self.feature_names_:
            feature_importance = dict(zip(self.feature_names_, state_mean))
            # Sort by absolute value
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            return dict(sorted_features[:5])  # Top 5 features
        else:
            return {f"feature_{i}": val for i, val in enumerate(state_mean)}

    def _interpret_state_financially(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        feature_names: list
    ) -> Dict[str, str]:
        """
        Provide financial interpretation of a state based on feature means.

        Args:
            mean: State mean values
            cov: State covariance matrix
            feature_names: Names of features

        Returns:
            Dictionary with financial interpretations
        """
        interpretation = {}

        # Look for common financial features
        feature_map = {name.lower(): i for i, name in enumerate(feature_names)}

        # Returns
        if 'log_ret' in feature_map:
            ret_idx = feature_map['log_ret']
            ret_mean = mean[ret_idx]
            if ret_mean > 0.001:
                interpretation['return_regime'] = "Bullish (Positive Returns)"
            elif ret_mean < -0.001:
                interpretation['return_regime'] = "Bearish (Negative Returns)"
            else:
                interpretation['return_regime'] = "Sideways (Neutral Returns)"

        # Volatility (use standard deviation of returns if available)
        if 'log_ret' in feature_map:
            ret_idx = feature_map['log_ret']
            ret_var = cov[ret_idx, ret_idx]
            ret_vol = np.sqrt(ret_var)
            if ret_vol > 0.02:
                interpretation['volatility_regime'] = "High Volatility"
            elif ret_vol > 0.01:
                interpretation['volatility_regime'] = "Medium Volatility"
            else:
                interpretation['volatility_regime'] = "Low Volatility"

        # Volume
        if 'volume' in feature_map or 'obv' in feature_map:
            vol_idx = feature_map.get('volume', feature_map.get('obv'))
            if vol_idx is not None:
                vol_mean = mean[vol_idx]
                if vol_mean > np.mean(mean):
                    interpretation['volume_regime'] = "High Volume"
                else:
                    interpretation['volume_regime'] = "Low Volume"

        # Trend indicators
        trend_features = ['sma', 'ema', 'macd']
        trend_found = False
        for feature in trend_features:
            if any(feature in name.lower() for name in feature_names):
                interpretation['trend_regime'] = "Trending State"
                trend_found = True
                break

        if not trend_found:
            interpretation['trend_regime'] = "Mean-Reverting State"

        return interpretation

    def analyze_state_transitions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze state transition patterns.

        Args:
            X: Input data

        Returns:
            Dictionary with transition analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before analyzing transitions")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get state sequence
        states = self.predict(X_array)

        # Get transition matrix
        transmat = self.get_transition_matrix()

        # Analyze transition patterns
        transition_analysis = {
            'transition_matrix': transmat,
            'state_frequencies': np.bincount(states, minlength=self.n_components) / len(states),
            'expected_residence_times': 1 / np.diag(transmat),
            'transition_entropy': -np.sum(transmat * np.log(transmat + 1e-10)),
            'most_likely_transitions': []
        }

        # Find most likely transitions
        for i in range(self.n_components):
            for j in range(self.n_components):
                if i != j:
                    transition_analysis['most_likely_transitions'].append(
                        (i, j, transmat[i, j])
                    )

        # Sort by probability
        transition_analysis['most_likely_transitions'].sort(
            key=lambda x: x[2], reverse=True
        )

        # Analyze actual transitions in the data
        actual_transitions = 0
        transition_counts = np.zeros((self.n_components, self.n_components))
        for i in range(len(states) - 1):
            transition_counts[states[i], states[i + 1]] += 1
            if states[i] != states[i + 1]:
                actual_transitions += 1

        transition_analysis.update({
            'actual_transition_counts': transition_counts,
            'actual_transition_probabilities': transition_counts / transition_counts.sum(axis=1, keepdims=True),
            'total_transitions': actual_transitions,
            'transition_rate': actual_transitions / (len(states) - 1)
        })

        return transition_analysis

    def detect_regime_changes(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Detect significant regime changes in the data.

        Args:
            X: Input data
            threshold: Probability threshold for regime change detection

        Returns:
            Boolean array indicating regime changes
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before detecting regime changes")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get state probabilities
        probabilities = self.predict_proba(X_array)

        # Detect regime changes (when the dominant state changes)
        dominant_states = np.argmax(probabilities, axis=1)
        regime_changes = np.zeros(len(dominant_states), dtype=bool)

        for i in range(1, len(dominant_states)):
            if dominant_states[i] != dominant_states[i-1]:
                # Check if the new state is significantly more probable
                if probabilities[i, dominant_states[i]] > threshold:
                    regime_changes[i] = True

        return regime_changes

    def compute_state_persistence(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """
        Compute persistence statistics for each state.

        Args:
            X: Input data

        Returns:
            Dictionary with persistence statistics
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing persistence")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get state sequence
        states = self.predict(X_array)

        persistence_stats = {}
        for state in range(self.n_components):
            state_mask = states == state
            state_indices = np.where(state_mask)[0]

            if len(state_indices) == 0:
                persistence_stats[state] = {
                    'mean_duration': 0,
                    'max_duration': 0,
                    'min_duration': 0,
                    'num_periods': 0
                }
                continue

            # Find contiguous periods
            if len(state_indices) > 0:
                # Find gaps in indices to identify periods
                periods = []
                current_period = [state_indices[0]]

                for i in range(1, len(state_indices)):
                    if state_indices[i] == state_indices[i-1] + 1:
                        current_period.append(state_indices[i])
                    else:
                        periods.append(current_period)
                        current_period = [state_indices[i]]
                periods.append(current_period)

                # Calculate statistics
                durations = [len(period) for period in periods]
                persistence_stats[state] = {
                    'mean_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'num_periods': len(periods),
                    'total_observations': len(state_indices)
                }

        return persistence_stats
