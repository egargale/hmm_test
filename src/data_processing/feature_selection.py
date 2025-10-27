"""
Feature Selection and Quality Assessment Module

This module provides automated feature selection algorithms and quality assessment
tools for financial time series data used in HMM analysis.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    mutual_info_classif,
    mutual_info_regression,
)

from utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selectors."""

    def __init__(self, **params):
        self.params = params
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureSelector':
        """Fit the feature selector to the data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using selected features."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)

    def get_support(self) -> List[bool]:
        """Get boolean mask of selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return self.selected_features_

    def get_feature_names_out(self) -> List[str]:
        """Get names of selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return [name for name, selected in zip(self.feature_names_, self.selected_features_) if selected]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted_ or self.feature_scores_ is None:
            raise ValueError("Feature selector has not been fitted or does not provide scores.")
        return pd.DataFrame({
            'feature': self.feature_names_,
            'score': self.feature_scores_
        }).sort_values('score', ascending=False)


class CorrelationFeatureSelector(BaseFeatureSelector):
    """Feature selector based on correlation analysis."""

    def __init__(self, threshold: float = 0.95, method: str = 'pearson',
                 target_correlation: bool = True, target_threshold: float = 0.1):
        """
        Initialize correlation-based feature selector.

        Args:
            threshold: Correlation threshold for removing redundant features
            method: Correlation method ('pearson', 'spearman', 'kendall')
            target_correlation: Whether to consider correlation with target
            target_threshold: Minimum correlation with target variable
        """
        super().__init__(
            threshold=threshold,
            method=method,
            target_correlation=target_correlation,
            target_threshold=target_threshold
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationFeatureSelector':
        """Fit the correlation-based selector."""
        self.feature_names_ = X.columns.tolist()

        # Remove highly correlated features
        corr_matrix = X.corr(method=self.params['method']).abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to remove
        to_remove = set()
        for col in upper_triangle.columns:
            if any(upper_triangle[col] > self.params['threshold']):
                correlated_cols = upper_triangle[col][upper_triangle[col] > self.params['threshold']].index.tolist()
                to_remove.update(correlated_cols[1:])  # Keep the first one

        # Consider target correlation if target is provided
        if self.params['target_correlation'] and y is not None:
            target_corr = X.corrwith(y, method=self.params['method']).abs()
            low_corr_features = target_corr[target_corr < self.params['target_threshold']].index.tolist()
            to_remove.update(low_corr_features)

        # Create selected features mask
        self.selected_features_ = [feature not in to_remove for feature in self.feature_names_]
        self.feature_scores_ = 1 - corr_matrix.max(axis=1)  # Lower correlation = higher score
        self.is_fitted_ = True

        logger.info(f"Correlation selector: removed {len(to_remove)} features, kept {sum(self.selected_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return X[self.get_feature_names_out()]


class VarianceFeatureSelector(BaseFeatureSelector):
    """Feature selector based on variance threshold."""

    def __init__(self, threshold: float = 0.01):
        """
        Initialize variance-based feature selector.

        Args:
            threshold: Variance threshold below which features are removed
        """
        super().__init__(threshold=threshold)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VarianceFeatureSelector':
        """Fit the variance-based selector."""
        self.feature_names_ = X.columns.tolist()

        # Calculate variance for each feature
        variances = X.var()
        self.feature_scores_ = variances.values
        self.selected_features_ = (variances >= self.params['threshold']).values
        self.is_fitted_ = True

        removed_count = sum(~np.array(self.selected_features_))
        logger.info(f"Variance selector: removed {removed_count} features, kept {sum(self.selected_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return X[self.get_feature_names_out()]


class MutualInformationFeatureSelector(BaseFeatureSelector):
    """Feature selector based on mutual information."""

    def __init__(self, k: int = 10, task_type: str = 'regression', random_state: int = 42):
        """
        Initialize mutual information-based feature selector.

        Args:
            k: Number of features to select
            task_type: 'regression' or 'classification'
            random_state: Random state for reproducibility
        """
        super().__init__(k=k, task_type=task_type, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MutualInformationFeatureSelector':
        """Fit the mutual information-based selector."""
        if y is None:
            raise ValueError("Target variable y is required for mutual information selection.")

        self.feature_names_ = X.columns.tolist()

        # Calculate mutual information scores
        if self.params['task_type'] == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=self.params['random_state'])
        else:
            mi_scores = mutual_info_classif(X, y, random_state=self.params['random_state'])

        self.feature_scores_ = mi_scores

        # Select top k features
        k = min(self.params['k'], len(X.columns))
        top_indices = np.argsort(mi_scores)[-k:]
        self.selected_features_ = [i in top_indices for i in range(len(X.columns))]
        self.is_fitted_ = True

        logger.info(f"Mutual information selector: selected {k} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return X[self.get_feature_names_out()]


class RecursiveFeatureEliminationSelector(BaseFeatureSelector):
    """Feature selector using recursive feature elimination."""

    def __init__(self, n_features_to_select: int = 10, step: float = 0.1,
                 estimator_type: str = 'random_forest', cv: int = 5, random_state: int = 42):
        """
        Initialize RFE-based feature selector.

        Args:
            n_features_to_select: Number of features to select
            step: Step size for feature elimination
            estimator_type: Type of estimator to use ('random_forest', 'linear')
            cv: Cross-validation folds
            random_state: Random state for reproducibility
        """
        super().__init__(
            n_features_to_select=n_features_to_select,
            step=step,
            estimator_type=estimator_type,
            cv=cv,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RecursiveFeatureEliminationSelector':
        """Fit the RFE-based selector."""
        if y is None:
            raise ValueError("Target variable y is required for RFE selection.")

        self.feature_names_ = X.columns.tolist()

        # Determine if classification or regression
        is_classification = len(y.unique()) < len(y) * 0.1  # Heuristic

        # Create estimator
        if self.params['estimator_type'] == 'random_forest':
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.params['random_state'])
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.params['random_state'])
        else:
            from sklearn.linear_model import LinearCV, LogisticRegressionCV
            if is_classification:
                estimator = LogisticRegressionCV(cv=self.params['cv'], random_state=self.params['random_state'])
            else:
                estimator = LinearCV()

        # Perform RFE
        n_features = min(self.params['n_features_to_select'], len(X.columns))
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=self.params['step'])
        rfe.fit(X, y)

        self.selected_features_ = rfe.support_
        self.feature_scores_ = rfe.ranking_
        self.is_fitted_ = True

        logger.info(f"RFE selector: selected {n_features} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if not self.is_fitted_:
            raise ValueError("Feature selector has not been fitted yet.")
        return X[self.get_feature_names_out()]


class FeatureQualityScorer:
    """Comprehensive feature quality assessment system."""

    def __init__(self, completeness_weight: float = 0.3, stability_weight: float = 0.3,
                 predictive_weight: float = 0.2, uniqueness_weight: float = 0.2):
        """
        Initialize feature quality scorer.

        Args:
            completeness_weight: Weight for completeness score
            stability_weight: Weight for stability score
            predictive_weight: Weight for predictive power score
            uniqueness_weight: Weight for uniqueness score
        """
        self.completeness_weight = completeness_weight
        self.stability_weight = stability_weight
        self.predictive_weight = predictive_weight
        self.uniqueness_weight = uniqueness_weight

    def score_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate comprehensive quality scores for all features.

        Args:
            X: Feature matrix
            y: Target variable (optional for predictive power)

        Returns:
            DataFrame with quality scores and breakdown
        """
        quality_scores = []

        for feature in X.columns:
            feature_data = X[feature]

            # Skip if feature is all NaN
            if feature_data.isnull().all():
                quality_scores.append({
                    'feature': feature,
                    'overall_score': 0.0,
                    'completeness_score': 0.0,
                    'stability_score': 0.0,
                    'predictive_score': 0.0,
                    'uniqueness_score': 0.0
                })
                continue

            # Calculate individual quality components
            completeness_score = self._calculate_completeness_score(feature_data)
            stability_score = self._calculate_stability_score(feature_data)
            predictive_score = self._calculate_predictive_score(feature_data, y) if y is not None else 0.5
            uniqueness_score = self._calculate_uniqueness_score(feature_data, X)

            # Calculate overall score
            overall_score = (
                completeness_score * self.completeness_weight +
                stability_score * self.stability_weight +
                predictive_score * self.predictive_weight +
                uniqueness_score * self.uniqueness_weight
            )

            quality_scores.append({
                'feature': feature,
                'overall_score': overall_score,
                'completeness_score': completeness_score,
                'stability_score': stability_score,
                'predictive_score': predictive_score,
                'uniqueness_score': uniqueness_score
            })

        quality_df = pd.DataFrame(quality_scores)
        quality_df = quality_df.sort_values('overall_score', ascending=False)

        logger.info(f"Calculated quality scores for {len(quality_df)} features")
        return quality_df

    def _calculate_completeness_score(self, feature_data: pd.Series) -> float:
        """Calculate completeness score based on missing data."""
        total_points = len(feature_data)
        valid_points = feature_data.count()
        return valid_points / total_points if total_points > 0 else 0.0

    def _calculate_stability_score(self, feature_data: pd.Series, window: int = 100) -> float:
        """Calculate stability score based on rolling statistics."""
        if len(feature_data) < window * 2:
            return 0.5  # Default score for insufficient data

        # Calculate rolling statistics
        rolling_mean = feature_data.rolling(window=window).mean()
        rolling_std = feature_data.rolling(window=window).std()

        # Calculate coefficient of variation
        cv = rolling_std / rolling_mean.abs()
        cv = cv.dropna()

        # Lower CV = higher stability
        if len(cv) > 0:
            avg_cv = cv.mean()
            stability_score = max(0, 1 - avg_cv)
        else:
            stability_score = 0.5

        return stability_score

    def _calculate_predictive_score(self, feature_data: pd.Series, target: pd.Series) -> float:
        """Calculate predictive power score based on correlation with target."""
        if target is None:
            return 0.5

        # Align data
        aligned_data = pd.concat([feature_data, target], axis=1).dropna()
        if len(aligned_data) < 10:
            return 0.5

        feature_clean = aligned_data.iloc[:, 0]
        target_clean = aligned_data.iloc[:, 1]

        try:
            # Calculate correlation
            correlation = abs(feature_clean.corr(target_clean))
            return min(correlation, 1.0)
        except:
            return 0.5

    def _calculate_uniqueness_score(self, feature_data: pd.Series, X: pd.DataFrame) -> float:
        """Calculate uniqueness score based on correlation with other features."""
        correlations = []
        for other_feature in X.columns:
            if other_feature != feature_data.name:
                try:
                    corr = abs(feature_data.corr(X[other_feature]))
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue

        if correlations:
            avg_correlation = np.mean(correlations)
            uniqueness_score = max(0, 1 - avg_correlation)
        else:
            uniqueness_score = 1.0

        return uniqueness_score

    def filter_by_quality(self, X: pd.DataFrame, min_score: float = 0.5,
                        y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter features based on quality scores.

        Args:
            X: Feature matrix
            min_score: Minimum quality score threshold
            y: Target variable

        Returns:
            Tuple of (filtered_features, quality_report)
        """
        quality_report = self.score_features(X, y)
        high_quality_features = quality_report[quality_report['overall_score'] >= min_score]['feature'].tolist()

        filtered_X = X[high_quality_features] if high_quality_features else pd.DataFrame()

        logger.info(f"Filtered {len(high_quality_features)} features with quality >= {min_score}")
        return filtered_X, quality_report


class FeatureSelectionPipeline:
    """Pipeline for automated feature selection with multiple methods."""

    def __init__(self, selectors: List[BaseFeatureSelector],
                 quality_threshold: float = 0.5, enable_quality_filtering: bool = True):
        """
        Initialize feature selection pipeline.

        Args:
            selectors: List of feature selectors to apply in sequence
            quality_threshold: Minimum quality score for features
            enable_quality_filtering: Whether to apply quality-based filtering
        """
        self.selectors = selectors
        self.quality_threshold = quality_threshold
        self.enable_quality_filtering = enable_quality_filtering
        self.quality_scorer = FeatureQualityScorer()
        self.selected_features_ = None
        self.selection_report_ = None

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit all selectors and transform the data.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Transformed feature matrix with selected features
        """
        current_X = X.copy()
        selection_steps = []

        for i, selector in enumerate(self.selectors):
            logger.info(f"Applying selector {i+1}/{len(self.selectors)}: {type(selector).__name__}")

            # Fit and transform
            current_X = selector.fit_transform(current_X, y)

            # Record selection step
            step_info = {
                'step': i + 1,
                'selector': type(selector).__name__,
                'features_in': len(selector.feature_names_),
                'features_out': len(selector.get_feature_names_out()),
                'selected_features': selector.get_feature_names_out()
            }
            selection_steps.append(step_info)

        # Apply quality filtering if enabled
        if self.enable_quality_filtering and len(current_X) > 0:
            logger.info("Applying quality-based filtering")
            current_X, quality_report = self.quality_scorer.filter_by_quality(
                current_X, self.quality_threshold, y
            )
            self.quality_report_ = quality_report
        else:
            self.quality_report_ = None

        self.selected_features_ = current_X.columns.tolist()
        self.selection_report_ = {
            'pipeline_steps': selection_steps,
            'final_features': self.selected_features_,
            'quality_report': self.quality_report_
        }

        logger.info(f"Feature selection pipeline completed. Final features: {len(self.selected_features_)}")
        return current_X

    def get_selection_report(self) -> Dict[str, Any]:
        """Get comprehensive selection report."""
        if self.selection_report_ is None:
            raise ValueError("Pipeline has not been fitted yet.")
        return self.selection_report_
