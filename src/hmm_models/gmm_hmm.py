"""
GMM HMM Model Implementation

Implements Hidden Markov Model with Gaussian Mixture Model emission distributions
for more complex financial regime modeling with multi-modal distributions.
"""

from typing import Any, Dict, List, Optional, Union

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


class GMMHMMModel(BaseHMMModel):
    """
    Hidden Markov Model with Gaussian Mixture Model emission distributions.

    Suitable for financial time series where return distributions within
    regimes may be multi-modal or have complex shapes that cannot be
    adequately captured by simple Gaussian distributions.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_mix: int = 2,
        covariance_type: str = "full",
        random_state: Optional[int] = None,
        n_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        implementation: str = "log",
        params: str = "stmcw",  # s=startprob, t=transmat, m=means, c=covars, w=weights
        init_params: str = "stmcw"
    ):
        """
        Initialize GMM HMM model.

        Args:
            n_components: Number of hidden states (regimes)
            n_mix: Number of mixture components per state
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

        self.n_mix = n_mix
        self.implementation = implementation
        self.params = params
        self.init_params = init_params
        self.n_iter = n_iter  # Store n_iter for hmmlearn

    def _create_model(self) -> "hmm.GMMHMM":
        """Create the GMM HMM model."""
        model = hmm.GMMHMM(
            n_components=self.n_components,
            n_mix=self.n_mix,
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
        Prepare features for GMM HMM training.

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
            # More important for GMM than simple Gaussian
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
            'weights_': self.model_.weights_.copy(),
            'means_': self.model_.means_.copy(),
            'covars_': self.model_.covars_.copy()
        }

        # Add parameter analysis
        params['parameter_analysis'] = {
            'startprob_entropy': -np.sum(self.model_.startprob_ * np.log(self.model_.startprob_ + 1e-10)),
            'transmat_entropy': np.mean([
                -np.sum(row * np.log(row + 1e-10)) for row in self.model_.transmat_
            ]),
            'mixture_weights_entropy': np.mean([
                -np.sum(weights * np.log(weights + 1e-10)) for weights in self.model_.weights_
            ]),
            'mean_condition_numbers': [],
            'covariance_determinants': [],
            'covariance_traces': []
        }

        # Analyze mixture components
        for state in range(self.n_components):
            for mix in range(self.n_mix):
                if self.covariance_type == "full":
                    cov = self.model_.covars_[state, mix]
                else:
                    cov = self.model_.covars_[state]

                params['parameter_analysis']['mean_condition_numbers'].append(np.linalg.cond(cov))
                params['parameter_analysis']['covariance_determinants'].append(np.linalg.det(cov))
                params['parameter_analysis']['covariance_traces'].append(np.trace(cov))

        return params

    def _count_emission_parameters(self) -> int:
        """Count GMM emission distribution parameters."""
        if not self.is_fitted_:
            return 0

        n_features = self.model_.means_.shape[2]  # (n_components, n_mix, n_features)

        # Mixture weights: n_components * (n_mix - 1)
        n_weights = self.n_components * (self.n_mix - 1)

        # Means: n_components * n_mix * n_features
        n_means = self.n_components * self.n_mix * n_features

        # Covariances: depends on covariance type
        if self.covariance_type == "full":
            # n_components * n_mix * n_features * (n_features + 1) / 2
            n_covars = self.n_components * self.n_mix * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            # n_components * n_mix * n_features
            n_covars = self.n_components * self.n_mix * n_features
        elif self.covariance_type == "tied":
            # n_components * n_features * (n_features + 1) / 2
            n_covars = self.n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == "spherical":
            # n_components * n_mix
            n_covars = self.n_components * self.n_mix
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")

        return n_weights + n_means + n_covars

    def get_state_descriptions(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Generate human-readable descriptions of each state including mixture components.

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
            state_prob = np.mean(state_mask)

            description = {
                'state_id': state,
                'frequency': state_prob,
                'sample_count': np.sum(state_mask),
                'n_mixtures': self.n_mix,
                'mixture_components': [],
                'overall_statistics': self._compute_overall_state_statistics(state, params)
            }

            # Analyze each mixture component
            for mix in range(self.n_mix):
                if self.covariance_type == "full":
                    mix_mean = params['means_'][state, mix]
                    mix_cov = params['covars_'][state, mix]
                    mix_weight = params['weights_'][state, mix]
                else:
                    mix_mean = params['means_'][state, mix]
                    mix_cov = np.diag(params['covars_'][state, mix])  # Convert to full matrix
                    mix_weight = params['weights_'][state, mix]

                component_desc = {
                    'mixture_id': mix,
                    'weight': mix_weight,
                    'mean_values': mix_mean,
                    'covariance_matrix': mix_cov,
                    'volatility_level': np.sqrt(np.trace(mix_cov)),
                    'dominant_features': self._get_dominant_features(mix_mean),
                    'financial_interpretation': self._interpret_state_financially(
                        mix_mean, mix_cov, self.feature_names_
                    )
                }

                description['mixture_components'].append(component_desc)

            # Add financial interpretation for the entire state
            if self.feature_names_:
                description['financial_interpretation'] = self._interpret_mixture_state_financially(
                    description['mixture_components']
                )

            descriptions[state] = description

        return descriptions

    def _compute_overall_state_statistics(self, state: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall statistics for a state by combining mixture components."""
        weights = params['weights_'][state]

        # Weighted mean of means
        weighted_mean = np.sum(weights[:, np.newaxis] * params['means_'][state], axis=0)

        # Overall volatility (approximate)
        volatilities = []
        for mix in range(self.n_mix):
            if self.covariance_type == "full":
                cov = params['covars_'][state, mix]
            else:
                cov = np.diag(params['covars_'][state, mix])
            volatilities.append(np.sqrt(np.trace(cov)))

        overall_volatility = np.sum(weights * np.array(volatilities))

        # Mixture complexity (entropy of weights)
        weight_entropy = -np.sum(weights * np.log(weights + 1e-10))

        return {
            'weighted_mean': weighted_mean,
            'overall_volatility': overall_volatility,
            'weight_entropy': weight_entropy,
            'dominant_mixture': np.argmax(weights),
            'mixture_weights': weights
        }

    def _get_dominant_features(self, state_mean: np.ndarray) -> Dict[str, float]:
        """
        Identify dominant features in a state based on mean values.

        Args:
            state_mean: Mean values for the state/mixture

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
        Provide financial interpretation of a mixture component.

        Args:
            mean: Component mean values
            cov: Component covariance matrix
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
                interpretation['return_regime'] = "Bullish Component"
            elif ret_mean < -0.001:
                interpretation['return_regime'] = "Bearish Component"
            else:
                interpretation['return_regime'] = "Neutral Component"

        # Volatility
        if 'log_ret' in feature_map:
            ret_idx = feature_map['log_ret']
            ret_var = cov[ret_idx, ret_idx]
            ret_vol = np.sqrt(ret_var)
            if ret_vol > 0.02:
                interpretation['volatility_regime'] = "High Volatility Component"
            elif ret_vol > 0.01:
                interpretation['volatility_regime'] = "Medium Volatility Component"
            else:
                interpretation['volatility_regime'] = "Low Volatility Component"

        return interpretation

    def _interpret_mixture_state_financially(self, mixture_components: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Provide financial interpretation for the entire mixture state.

        Args:
            mixture_components: List of mixture component descriptions

        Returns:
            Dictionary with financial interpretations
        """
        interpretation = {}

        # Analyze dominant characteristics across mixtures
        returns = []
        volatilities = []
        weights = []

        for component in mixture_components:
            weights.append(component['weight'])
            volatilities.append(component['volatility_level'])

            if 'return_regime' in component.get('financial_interpretation', {}):
                ret_interpretation = component['financial_interpretation']['return_regime']
                if 'Bullish' in ret_interpretation:
                    returns.append(1)
                elif 'Bearish' in ret_interpretation:
                    returns.append(-1)
                else:
                    returns.append(0)

        # Weighted average interpretation
        if returns:
            weighted_return = np.sum(np.array(returns) * np.array(weights))
            weighted_volatility = np.sum(np.array(volatilities) * np.array(weights))

            if weighted_return > 0.2:
                interpretation['overall_return_tendency'] = "Bullish-Dominated State"
            elif weighted_return < -0.2:
                interpretation['overall_return_tendency'] = "Bearish-Dominated State"
            else:
                interpretation['overall_return_tendency'] = "Mixed/Balanced State"

            if weighted_volatility > 0.02:
                interpretation['overall_volatility_level'] = "High Volatility State"
            elif weighted_volatility > 0.01:
                interpretation['overall_volatility_level'] = "Medium Volatility State"
            else:
                interpretation['overall_volatility_level'] = "Low Volatility State"

        # Check for multi-modality
        weight_entropy = -np.sum(np.array(weights) * np.log(np.array(weights) + 1e-10))
        if weight_entropy > 0.5:
            interpretation['mixture_complexity'] = "Highly Multi-Modal State"
        elif weight_entropy > 0.2:
            interpretation['mixture_complexity'] = "Moderately Multi-Modal State"
        else:
            interpretation['mixture_complexity'] = "Primarily Single-Modal State"

        return interpretation

    def analyze_mixture_separation(self) -> Dict[str, Any]:
        """
        Analyze how well-separated the mixture components are within each state.

        Returns:
            Dictionary with separation analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before analyzing mixture separation")

        separation_analysis = {
            'state_separations': {},
            'overall_separation': {}
        }

        for state in range(self.n_components):
            state_separations = []

            # Get mixture components for this state
            weights = self.model_.weights_[state]
            means = self.model_.means_[state]

            # Compute pairwise distances between mixture components
            for i in range(self.n_mix):
                for j in range(i + 1, self.n_mix):
                    # Mahalanobis distance between mixture components
                    mean_diff = means[i] - means[j]

                    if self.covariance_type == "full":
                        # Use average covariance for Mahalanobis distance
                        avg_cov = (self.model_.covars_[state, i] + self.model_.covars_[state, j]) / 2
                        try:
                            inv_cov = np.linalg.inv(avg_cov)
                            mahal_dist = np.sqrt(mean_diff @ inv_cov @ mean_diff.T)
                        except np.linalg.LinAlgError:
                            # Fall back to Euclidean distance
                            mahal_dist = np.linalg.norm(mean_diff)
                    else:
                        # Euclidean distance for other covariance types
                        mahal_dist = np.linalg.norm(mean_diff)

                    # Weight the distance by the product of mixture weights
                    weighted_distance = mahal_dist * np.sqrt(weights[i] * weights[j])

                    state_separations.append({
                        'component_1': i,
                        'component_2': j,
                        'mahalanobis_distance': mahal_dist,
                        'weighted_distance': weighted_distance,
                        'weight_product': weights[i] * weights[j]
                    })

            # Sort by weighted distance
            state_separations.sort(key=lambda x: x['weighted_distance'], reverse=True)

            separation_analysis['state_separations'][state] = {
                'separations': state_separations,
                'mean_separation': np.mean([s['weighted_distance'] for s in state_separations]),
                'max_separation': state_separations[0]['weighted_distance'] if state_separations else 0,
                'min_separation': state_separations[-1]['weighted_distance'] if state_separations else 0,
                'n_effective_components': self._count_effective_components(weights)
            }

        # Overall separation metrics
        all_separations = []
        for state_data in separation_analysis['state_separations'].values():
            all_separations.extend([s['weighted_distance'] for s in state_data['separations']])

        if all_separations:
            separation_analysis['overall_separation'] = {
                'mean_weighted_separation': np.mean(all_separations),
                'std_weighted_separation': np.std(all_separations),
                'max_weighted_separation': np.max(all_separations),
                'min_weighted_separation': np.min(all_separations)
            }

        return separation_analysis

    def _count_effective_components(self, weights: np.ndarray, threshold: float = 0.05) -> int:
        """
        Count the number of effective mixture components (weight > threshold).

        Args:
            weights: Mixture component weights
            threshold: Minimum weight threshold

        Returns:
            Number of effective components
        """
        return np.sum(weights > threshold)

    def get_mixture_responsibilities(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get mixture component responsibilities for each sample.

        Args:
            X: Input data

        Returns:
            Array of shape (n_samples, n_components, n_mix) with responsibilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting responsibilities")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get posterior probabilities for each mixture component
        # This requires accessing the internal structure of hmmlearn
        responsibilities = np.zeros((len(X_array), self.n_components, self.n_mix))

        for i in range(len(X_array)):
            sample = X_array[i:i+1]  # Keep 2D shape

            for state in range(self.n_components):
                for mix in range(self.n_mix):
                    # Compute responsibility using the model's internal parameters
                    weight = self.model_.weights_[state, mix]
                    mean = self.model_.means_[state, mix]

                    if self.covariance_type == "full":
                        cov = self.model_.covars_[state, mix]
                    else:
                        cov = np.diag(self.model_.covars_[state, mix])

                    # Compute Gaussian probability
                    diff = sample - mean
                    try:
                        inv_cov = np.linalg.inv(cov)
                        log_det = np.log(np.linalg.det(cov))
                        mahal = diff @ inv_cov @ diff.T
                        log_prob = -0.5 * (mahal + log_det + len(mean) * np.log(2 * np.pi))
                        prob = np.exp(log_prob)
                    except np.linalg.LinAlgError:
                        # Fall back to diagonal approximation
                        prob = 1e-10

                    responsibilities[i, state, mix] = weight * prob

        # Normalize responsibilities
        responsibilities = responsibilities / responsibilities.sum(axis=(1, 2), keepdims=True)

        return responsibilities
