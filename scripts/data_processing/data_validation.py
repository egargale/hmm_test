"""
Data Validation Module

Provides comprehensive data quality checks and validation for OHLCV data and
derived features, including outlier detection, missing value handling, and
data quality reporting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)


def validate_data(
    df: pd.DataFrame,
    ohlcv_columns: Optional[List[str]] = None,
    outlier_detection: bool = True,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    missing_value_strategy: str = "forward_fill",
    generate_report: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform comprehensive data validation and quality checks.

    Args:
        df: DataFrame to validate (OHLCV data with optional features)
        ohlcv_columns: List of core OHLCV columns to validate
        outlier_detection: Whether to detect and handle outliers
        outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        outlier_threshold: Threshold for outlier detection
        missing_value_strategy: Strategy for handling missing values
        generate_report: Whether to generate detailed validation report

    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and validation report

    Raises:
        ValueError: If required OHLCV columns are missing
    """
    logger.info(
        f"Starting data validation for DataFrame with {len(df)} rows, {len(df.columns)} columns"
    )

    # Set default OHLCV columns if not provided
    if ohlcv_columns is None:
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

    # Initialize validation report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "original_shape": df.shape,
        "issues_found": [],
        "cleaning_actions": [],
        "statistics": {},
        "quality_score": None,
    }

    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # 1. Validate required columns
    df_clean, column_issues = _validate_columns(df_clean, ohlcv_columns)
    report["issues_found"].extend(column_issues)

    # 2. Validate data types
    df_clean, type_issues = _validate_data_types(df_clean, ohlcv_columns)
    report["issues_found"].extend(type_issues)

    # 3. Validate OHLCV consistency
    df_clean, consistency_issues = _validate_ohlcv_consistency(df_clean)
    report["issues_found"].extend(consistency_issues)

    # 4. Handle missing values
    df_clean, missing_issues = _handle_missing_values(
        df_clean, missing_value_strategy, ohlcv_columns
    )
    report["issues_found"].extend(missing_issues)

    # 5. Detect and handle outliers
    if outlier_detection:
        df_clean, outlier_issues = _handle_outliers(
            df_clean, ohlcv_columns, outlier_method, outlier_threshold
        )
        report["issues_found"].extend(outlier_issues)

    # 6. Validate datetime index
    df_clean, datetime_issues = _validate_datetime_index(df_clean)
    report["issues_found"].extend(datetime_issues)

    # 7. Generate statistics
    report["statistics"] = _generate_statistics(df_clean, ohlcv_columns)

    # 8. Calculate quality score
    report["quality_score"] = _calculate_quality_score(report)

    # 9. Final shape
    report["final_shape"] = df_clean.shape

    logger.info(f"Validation completed. Quality score: {report['quality_score']:.2f}")
    logger.info(
        f"Issues found: {len(report['issues_found'])}, Actions taken: {len(report['cleaning_actions'])}"
    )

    return df_clean, report


def _validate_columns(
    df: pd.DataFrame, required_columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate required columns exist."""
    issues = []

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(
            {
                "type": "missing_columns",
                "severity": "critical",
                "description": f"Missing required columns: {missing_columns}",
                "columns": missing_columns,
            }
        )
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        issues.append(
            {
                "type": "duplicate_columns",
                "severity": "warning",
                "description": f"Duplicate columns found: {duplicate_columns}",
                "columns": duplicate_columns,
            }
        )

    return df, issues


def _validate_data_types(
    df: pd.DataFrame, ohlcv_columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate and correct data types."""
    issues = []

    # Check OHLCV columns are numeric
    for col in ohlcv_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                issues.append(
                    {
                        "type": "type_conversion",
                        "severity": "info",
                        "description": f"Converted {col} to numeric",
                        "column": col,
                    }
                )
            except Exception as e:
                issues.append(
                    {
                        "type": "type_conversion_error",
                        "severity": "error",
                        "description": f"Failed to convert {col} to numeric: {e}",
                        "column": col,
                    }
                )

    return df, issues


def _validate_ohlcv_consistency(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate OHLCV data consistency."""
    issues = []

    # Check for negative prices
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if col in df.columns:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                issues.append(
                    {
                        "type": "negative_prices",
                        "severity": "warning",
                        "description": f"Found {negative_count} non-positive values in {col}",
                        "column": col,
                        "count": negative_count,
                    }
                )

    # Check OHLC consistency
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        # High should be >= open,close
        high_violations = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        if high_violations > 0:
            issues.append(
                {
                    "type": "ohlcv_consistency",
                    "severity": "warning",
                    "description": f"Found {high_violations} rows where high < max(open,close)",
                    "count": high_violations,
                }
            )

        # Low should be <= open,close
        low_violations = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if low_violations > 0:
            issues.append(
                {
                    "type": "ohlcv_consistency",
                    "severity": "warning",
                    "description": f"Found {low_violations} rows where low > min(open,close)",
                    "count": low_violations,
                }
            )

    # Check for negative volume
    if "volume" in df.columns:
        negative_volume = (df["volume"] < 0).sum()
        if negative_volume > 0:
            issues.append(
                {
                    "type": "negative_volume",
                    "severity": "warning",
                    "description": f"Found {negative_volume} negative volume values",
                    "count": negative_volume,
                }
            )

    return df, issues


def _handle_missing_values(
    df: pd.DataFrame, strategy: str, ohlcv_columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Handle missing values in the dataset."""
    issues = []

    # Count missing values before handling
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        return df, issues

    issues.append(
        {
            "type": "missing_values",
            "severity": "info",
            "description": f"Found {total_missing} missing values in dataset",
            "missing_counts": missing_counts[missing_counts > 0].to_dict(),
        }
    )

    original_shape = df.shape

    # Apply missing value strategy
    if strategy == "drop":
        # Drop rows with missing values in OHLCV columns
        df = df.dropna(subset=ohlcv_columns)
        issues.append(
            {
                "type": "missing_values_handled",
                "severity": "info",
                "description": f"Dropped {original_shape[0] - len(df)} rows with missing values",
                "strategy": strategy,
            }
        )

    elif strategy == "forward_fill":
        # Forward fill missing values
        df = df.ffill()
        # Backward fill any remaining NaNs
        df = df.bfill()
        issues.append(
            {
                "type": "missing_values_handled",
                "severity": "info",
                "description": "Applied forward fill and backward fill for missing values",
                "strategy": strategy,
            }
        )

    elif strategy == "interpolate":
        # Interpolate missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].interpolate(method="linear")
        issues.append(
            {
                "type": "missing_values_handled",
                "severity": "info",
                "description": "Applied linear interpolation for missing values",
                "strategy": strategy,
            }
        )

    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    return df, issues


def _handle_outliers(
    df: pd.DataFrame, ohlcv_columns: List[str], method: str, threshold: float
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Detect and handle outliers in the data."""
    issues = []

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    outlier_counts = {}
    total_outliers = 0

    for col in numeric_columns:
        if method == "iqr":
            outliers = _detect_outliers_iqr(df[col], threshold)
        elif method == "zscore":
            outliers = _detect_outliers_zscore(df[col], threshold)
        elif method == "isolation_forest":
            outliers = _detect_outliers_isolation_forest(df[col])
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_count = outliers.sum()
        if outlier_count > 0:
            outlier_counts[col] = outlier_count
            total_outliers += outlier_count

    if total_outliers > 0:
        issues.append(
            {
                "type": "outliers_detected",
                "severity": "warning",
                "description": f"Detected {total_outliers} outliers using {method} method",
                "method": method,
                "threshold": threshold,
                "outlier_counts": outlier_counts,
            }
        )

        # Flag outliers in a new column instead of removing them
        df["is_outlier"] = False
        for col, _count in outlier_counts.items():
            if method == "iqr":
                outliers = _detect_outliers_iqr(df[col], threshold)
            elif method == "zscore":
                outliers = _detect_outliers_zscore(df[col], threshold)
            elif method == "isolation_forest":
                outliers = _detect_outliers_isolation_forest(df[col])

            df.loc[outliers, "is_outlier"] = True

        issues.append(
            {
                "type": "outliers_flagged",
                "severity": "info",
                "description": f"Flagged {total_outliers} outliers in 'is_outlier' column",
                "action": "flagged",
            }
        )

    return df, issues


def _detect_outliers_iqr(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (series < lower_bound) | (series > upper_bound)


def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def _detect_outliers_isolation_forest(series: pd.Series) -> pd.Series:
    """Detect outliers using Isolation Forest."""
    try:
        from sklearn.ensemble import IsolationForest

        # Reshape data for sklearn
        data = series.values.reshape(-1, 1)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(data)

        return outliers == -1  # -1 indicates outliers
    except ImportError:
        logger.warning("sklearn not available, falling back to IQR method")
        return _detect_outliers_iqr(series)


def _validate_datetime_index(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Validate datetime index."""
    issues = []

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            issues.append(
                {
                    "type": "datetime_conversion",
                    "severity": "info",
                    "description": "Converted index to datetime",
                }
            )
        except Exception as e:
            issues.append(
                {
                    "type": "datetime_conversion_error",
                    "severity": "error",
                    "description": f"Failed to convert index to datetime: {e}",
                }
            )

    # Check for duplicate timestamps
    if isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(
                {
                    "type": "duplicate_timestamps",
                    "severity": "warning",
                    "description": f"Found {duplicates} duplicate timestamps",
                    "count": duplicates,
                }
            )

    return df, issues


def _generate_statistics(df: pd.DataFrame, ohlcv_columns: List[str]) -> Dict[str, Any]:
    """Generate statistics for the validation report."""
    stats = {
        "shape": df.shape,
        "column_types": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }

    # OHLCV statistics
    ohlcv_stats = {}
    for col in ohlcv_columns:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                ohlcv_stats[col] = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "missing_count": df[col].isnull().sum(),
                }

    stats["ohlcv_statistics"] = ohlcv_stats

    # Feature statistics
    feature_columns = [
        col for col in df.columns if col not in ohlcv_columns and col != "is_outlier"
    ]
    if feature_columns:
        feature_stats = {
            "total_features": len(feature_columns),
            "feature_types": df[feature_columns].dtypes.astype(str).to_dict(),
        }
        stats["feature_statistics"] = feature_stats

    return stats


def _calculate_quality_score(report: Dict[str, Any]) -> float:
    """Calculate overall data quality score (0-100)."""
    base_score = 100.0

    # Deduct points for issues
    for issue in report["issues_found"]:
        severity = issue.get("severity", "info")
        if severity == "critical":
            base_score -= 20
        elif severity == "error":
            base_score -= 10
        elif severity == "warning":
            base_score -= 5
        elif severity == "info":
            base_score -= 1

    # Deduct points for missing data
    if "statistics" in report and "ohlcv_statistics" in report["statistics"]:
        total_values = sum(
            stats["count"]
            for stats in report["statistics"]["ohlcv_statistics"].values()
        )
        total_missing = sum(
            stats["missing_count"]
            for stats in report["statistics"]["ohlcv_statistics"].values()
        )
        if total_values > 0:
            missing_ratio = total_missing / total_values
            base_score -= missing_ratio * 20

    # Ensure score doesn't go below 0
    return max(0.0, min(100.0, base_score))


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Validation Time: {report['validation_timestamp']}")
    print(f"Original Shape: {report['original_shape']}")
    print(f"Final Shape: {report['final_shape']}")
    print(f"Quality Score: {report['quality_score']:.2f}/100")

    if report["issues_found"]:
        print(f"\nIssues Found ({len(report['issues_found'])}):")
        for i, issue in enumerate(report["issues_found"], 1):
            severity_icon = {
                "critical": "🚨",
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }.get(issue.get("severity", "info"), "ℹ️")
            print(
                f"  {i}. {severity_icon} [{issue.get('severity', 'info').upper()}] {issue['description']}"
            )

    if report["statistics"]:
        print("\nStatistics:")
        print(f"  - Memory Usage: {report['statistics']['memory_usage_mb']:.2f} MB")
        if "ohlcv_statistics" in report["statistics"]:
            print(f"  - OHLCV Columns: {len(report['statistics']['ohlcv_statistics'])}")
        if "feature_statistics" in report["statistics"]:
            print(
                f"  - Feature Columns: {report['statistics']['feature_statistics']['total_features']}"
            )

    print("=" * 60)
