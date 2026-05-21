"""
Base HMM Model Class

Provides the abstract base class for all Hidden Markov Model implementations
with common functionality for training, prediction, and evaluation.
"""

import abc
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from utils import get_logger

logger = get_logger(__name__)


class BaseHMMModel(BaseEstimator, abc.ABC):
    """
    Abstract base class for Hidden Markov Model implementations.

    Provides common functionality for training, evaluation, and persistence
    while allowing specific implementations to define their own algorithms.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Initialize base HMM model.

        Args:
            n_components: Number of hidden states
            covariance_type: Covariance type for emission distributions
            random_state: Random seed for reproducibility
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            verbose: Whether to print training progress
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Internal state
        self.model_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        self.training_history_ = {}
        self.convergence_info_ = {}

    @abc.abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying HMM model implementation."""
        pass

    @abc.abstractmethod
    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepare features for HMM training."""
        pass

    @abc.abstractmethod
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract model parameters for analysis."""
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> "BaseHMMModel":
        """
        Fit the HMM model to the training data.

        Args:
            X: Training features (DataFrame or array)
            y: Not used for unsupervised HMM, kept for compatibility
            feature_columns: List of feature column names (if X is DataFrame)

        Returns:
            self: Fitted model
        """
        logger.info(
            f"Fitting {self.__class__.__name__} with {self.n_components} states"
        )
        start_time = time.time()

        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                feature_columns = X.columns.tolist()
            self.feature_names_ = feature_columns
            X = X[feature_columns]

        # Prepare features
        X_array = self._prepare_features(X)
        logger.info(f"Training data shape: {X_array.shape}")

        # Validate input
        if len(X_array) < self.n_components:
            raise ValueError(
                f"Not enough data points ({len(X_array)}) for {self.n_components} states"
            )

        # Create and fit model
        self.model_ = self._create_model()

        try:
            # Fit the model
            self.model_.fit(X_array)

            # Record training history
            self.training_history_ = {
                "n_samples": len(X_array),
                "n_features": X_array.shape[1],
                "training_time": time.time() - start_time,
                "converged": getattr(self.model_, "monitor_", None)
                and getattr(self.model_.monitor_, "converged", False),
            }

            # Store convergence info if available
            if hasattr(self.model_, "monitor_"):
                monitor = self.model_.monitor_
                self.convergence_info_ = {
                    "converged": monitor.converged,
                    "n_iter": monitor.iter,
                    "history": getattr(monitor, "history", []),
                }

            self.is_fitted_ = True
            logger.info(
                f"Model fitted successfully in {self.training_history_['training_time']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to fit HMM model: {e}")
            raise

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict hidden states for the given data.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Array of predicted state indices
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        X_array = self._prepare_features(X)

        # Predict states
        states = self.model_.predict(X_array)

        logger.debug(f"Predicted states for {len(states)} samples")
        return states

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict state probabilities for the given data.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Array of state probabilities (n_samples, n_components)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get posterior probabilities
        postprob = self.model_.predict_proba(X_array)

        return postprob

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute log-likelihood of each sample under the model.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Array of log-likelihoods
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")

        # Prepare features
        X_array = self._prepare_features(X)

        # Compute log-likelihood
        log_likelihood = self.model_.score_samples(X_array)

        return log_likelihood

    def score(self, X: Union[pd.DataFrame, np.ndarray]) -> float:
        """
        Compute the total log-likelihood of the data under the model.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Total log-likelihood
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")

        # Prepare features
        X_array = self._prepare_features(X)

        # Compute total log-likelihood
        total_log_likelihood = self.model_.score(X_array)

        return total_log_likelihood

    def decode(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Find the best state sequence and its log probability.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Tuple of (state_sequence, log_probability)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before decoding")

        # Prepare features
        X_array = self._prepare_features(X)

        # Viterbi decoding
        logprob, state_sequence = self.model_.decode(X_array)

        return state_sequence, logprob

    def sample(
        self, n_samples: int = 1, random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples from the model.

        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for generation

        Returns:
            Tuple of (X, states) generated samples
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before sampling")

        # Generate samples
        X, states = self.model_.sample(n_samples, random_state=random_state)

        return X, states

    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get detailed model parameters for analysis.

        Returns:
            Dictionary with model parameters
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting parameters")

        params = self._extract_parameters()
        params.update(
            {
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "is_fitted": self.is_fitted_,
                "feature_names": self.feature_names_,
                "training_history": self.training_history_,
                "convergence_info": self.convergence_info_,
            }
        )

        return params

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the state transition matrix.

        Returns:
            Transition matrix of shape (n_components, n_components)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting transition matrix")

        return self.model_.transmat_.copy()

    def get_emission_parameters(self) -> Dict[str, Any]:
        """
        Get emission distribution parameters.

        Returns:
            Dictionary with emission parameters
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting emission parameters")

        return self._extract_parameters()

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        cv: int = 5,
        scoring: str = "neg_log_loss",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.

        Args:
            X: Input features
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv}-fold cross-validation")

        # Prepare features
        X_array = self._prepare_features(X)

        # Use time series split for financial data
        tscv = TimeSeriesSplit(n_splits=cv)

        # Clone model for CV
        model_clone = clone(self)

        # Perform cross-validation
        scores = cross_val_score(
            model_clone, X_array, cv=tscv, scoring=scoring, n_jobs=-1
        )

        cv_results = {
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scoring": scoring,
            "n_folds": cv,
        }

        logger.info(
            f"CV {scoring}: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}"
        )

        return cv_results

    def evaluate_model_quality(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model quality using various metrics.

        Args:
            X: Input features

        Returns:
            Dictionary with quality metrics
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before evaluation")

        # Prepare features
        X_array = self._prepare_features(X)

        # Get predictions and probabilities
        states = self.predict(X_array)
        probabilities = self.predict_proba(X_array)
        log_likelihoods = self.score_samples(X_array)

        # Calculate metrics
        quality_metrics = {
            "total_log_likelihood": self.score(X_array),
            "mean_log_likelihood": np.mean(log_likelihoods),
            "std_log_likelihood": np.std(log_likelihoods),
            "min_log_likelihood": np.min(log_likelihoods),
            "max_log_likelihood": np.max(log_likelihoods),
            "n_samples": len(X_array),
            "state_distribution": np.bincount(states, minlength=self.n_components)
            / len(states),
            "mean_state_probabilities": np.mean(probabilities, axis=0),
            "state_entropy": -np.sum(
                np.mean(probabilities, axis=0)
                * np.log(np.mean(probabilities, axis=0) + 1e-10)
            ),
        }

        # Calculate BIC and AIC if possible
        n_params = self._count_parameters()
        quality_metrics.update(
            {
                "bic": -2 * quality_metrics["total_log_likelihood"]
                + n_params * np.log(len(X_array)),
                "aic": -2 * quality_metrics["total_log_likelihood"] + 2 * n_params,
                "n_parameters": n_params,
            }
        )

        return quality_metrics

    def _count_parameters(self) -> int:
        """Count the number of free parameters in the model."""
        if not self.is_fitted_:
            return 0

        # Transition matrix parameters (n_components * (n_components - 1))
        n_transition = self.n_components * (self.n_components - 1)

        # Initial state probabilities (n_components - 1)
        n_initial = self.n_components - 1

        # Emission parameters (implementation-specific)
        n_emission = self._count_emission_parameters()

        return n_transition + n_initial + n_emission

    @abc.abstractmethod
    def _count_emission_parameters(self) -> int:
        """Count emission distribution parameters."""
        pass

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model parameters and state
        model_data = {
            "model": self.model_,
            "feature_names": self.feature_names_,
            "training_history": self.training_history_,
            "convergence_info": self.convergence_info_,
            "hyperparameters": {
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "tol": self.tol,
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Union[str, Path]) -> "BaseHMMModel":
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Restore model state
        self.model_ = model_data["model"]
        self.feature_names_ = model_data["feature_names"]
        self.training_history_ = model_data["training_history"]
        self.convergence_info_ = model_data["convergence_info"]

        # Restore hyperparameters
        hyperparams = model_data["hyperparameters"]
        self.n_components = hyperparams["n_components"]
        self.covariance_type = hyperparams["covariance_type"]
        self.random_state = hyperparams["random_state"]
        self.max_iter = hyperparams["max_iter"]
        self.tol = hyperparams["tol"]

        self.is_fitted_ = True

        logger.info(f"Model loaded from {filepath}")
        return self

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted_ else "unfitted"
        return f"{self.__class__.__name__}(n_components={self.n_components}, status={status})"
