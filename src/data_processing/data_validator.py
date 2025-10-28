"""
Enhanced data validation and quality assessment system.

This module provides comprehensive data validation, including OHLC consistency checks,
outlier detection, missing data handling, and data quality scoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    level: ValidationLevel
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    suggested_action: Optional[str] = None
    data_value: Optional[Any] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    is_valid: bool
    total_rows: int
    valid_rows: int
    issues: List[ValidationIssue]
    quality_score: float
    summary: Dict[str, Any]
    recommendations: List[str]


@dataclass
class OutlierReport:
    """Outlier detection report."""

    total_outliers: int
    outlier_percentage: float
    outlier_indices: List[int]
    outlier_values: List[float]
    method: str
    threshold: float
    summary: Dict[str, Any]


@dataclass
class RangeReport:
    """Data range validation report."""

    column: str
    min_value: float
    max_value: float
    range_violations: List[int]
    mean_value: float
    std_value: float
    is_valid_range: bool


class DataValidator:
    """
    Enhanced data validator with comprehensive validation capabilities.

    Features:
    - OHLC consistency validation
    - Outlier detection (IQR, Z-score, Isolation Forest)
    - Missing data analysis and handling
    - Data quality scoring
    - Range validation
    - Duplicate detection
    - Data type validation
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.issues: List[ValidationIssue] = []

    def validate_dataset(self, df: pd.DataFrame, **kwargs) -> ValidationReport:
        """
        Perform comprehensive validation on the dataset.

        Args:
            df: Input DataFrame
            **kwargs: Additional validation parameters

        Returns:
            Comprehensive validation report
        """
        logger.info(f"Starting comprehensive validation for {len(df)} rows")
        self.issues = []

        try:
            # Basic structure validation
            self._validate_basic_structure(df)

            # OHLC consistency validation
            if self._has_ohlc_columns(df):
                self._validate_ohlc_consistency(df)

            # Data type validation
            self._validate_data_types(df)

            # Missing data analysis
            self._analyze_missing_data(df)

            # Duplicate detection
            self._detect_duplicates(df)

            # Range validation
            self._validate_ranges(df)

            # Outlier detection
            if kwargs.get("detect_outliers", True):
                self._detect_outliers(df, method=kwargs.get("outlier_method", "iqr"))

            # Calculate quality score
            quality_score = self._calculate_quality_score(df, len(self.issues))

            # Generate summary and recommendations
            summary = self._generate_summary(df, quality_score)
            recommendations = self._generate_recommendations()

            is_valid = not any(
                issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
                for issue in self.issues
            )

            valid_rows = len(df) - len(
                {
                    idx
                    for issue in self.issues
                    if issue.row_indices
                    for idx in issue.row_indices
                }
            )

            report = ValidationReport(
                is_valid=is_valid,
                total_rows=len(df),
                valid_rows=valid_rows,
                issues=self.issues,
                quality_score=quality_score,
                summary=summary,
                recommendations=recommendations,
            )

            logger.info(
                f"Validation completed: Quality Score = {quality_score:.3f}, "
                f"Issues = {len(self.issues)}"
            )

            return report

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    def _validate_basic_structure(self, df: pd.DataFrame) -> None:
        """Validate basic DataFrame structure."""
        if df.empty:
            self.issues.append(
                ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    message="DataFrame is empty",
                    suggested_action="Check data source and loading process",
                )
            )

        if len(df.columns) == 0:
            self.issues.append(
                ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    message="No columns found in DataFrame",
                    suggested_action="Check CSV format and column headers",
                )
            )

        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            self.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Empty columns detected: {empty_columns}",
                    suggested_action="Consider removing or filling empty columns",
                )
            )

    def _has_ohlc_columns(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has OHLC columns."""
        ohlc_patterns = ["open", "high", "low", "close"]
        df_cols_lower = [col.lower() for col in df.columns]

        return any(pattern in df_cols_lower for pattern in ohlc_patterns)

    def _validate_ohlc_consistency(self, df: pd.DataFrame) -> None:
        """Validate OHLC data consistency."""
        try:
            # Find OHLC columns (case-insensitive)
            ohlc_cols = {}
            for pattern in ["open", "high", "low", "close"]:
                matching_cols = [
                    col for col in df.columns if pattern.lower() == col.lower()
                ]
                if matching_cols:
                    ohlc_cols[pattern] = matching_cols[0]

            if len(ohlc_cols) < 4:
                self.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message="Incomplete OHLC columns found",
                        suggested_action="Ensure all OHLC columns are present",
                    )
                )
                return

            # Convert to numeric
            ohlc_data = {}
            for pattern, col in ohlc_cols.items():
                try:
                    ohlc_data[pattern] = pd.to_numeric(df[col], errors="coerce")
                except ValueError:
                    self.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            message=f"Non-numeric data in {col} column",
                            column=col,
                            suggested_action="Clean and convert to numeric values",
                        )
                    )
                    return

            # Validate OHLC relationships: High >= Open, Low <= Open, etc.
            open_prices = ohlc_data["open"]
            high_prices = ohlc_data["high"]
            low_prices = ohlc_data["low"]
            close_prices = ohlc_data["close"]

            # Check High >= max(Open, Close)
            high_violations = high_prices < np.maximum(open_prices, close_prices)
            if high_violations.any():
                violation_indices = df.index[high_violations].tolist()
                self.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"High prices lower than Open/Close in {high_violations.sum()} rows",
                        column=ohlc_cols["high"],
                        row_indices=violation_indices,
                        suggested_action="Correct High price values or verify data quality",
                    )
                )

            # Check Low <= min(Open, Close)
            low_violations = low_prices > np.minimum(open_prices, close_prices)
            if low_violations.any():
                violation_indices = df.index[low_violations].tolist()
                self.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        message=f"Low prices higher than Open/Close in {low_violations.sum()} rows",
                        column=ohlc_cols["low"],
                        row_indices=violation_indices,
                        suggested_action="Correct Low price values or verify data quality",
                    )
                )

            # Check for negative prices (usually invalid for stock data)
            for pattern, col in ohlc_cols.items():
                negative_prices = ohlc_data[pattern] < 0
                if negative_prices.any():
                    violation_indices = df.index[negative_prices].tolist()
                    self.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            message=f"Negative {pattern} prices in {negative_prices.sum()} rows",
                            column=col,
                            row_indices=violation_indices,
                            suggested_action="Verify if negative prices are valid for this data",
                        )
                    )

            # Check for zero prices with non-zero volume
            if "volume" in df.columns:
                volume_data = pd.to_numeric(df["volume"], errors="coerce")
                for pattern, col in ohlc_cols.items():
                    zero_price_nonzero_volume = (ohlc_data[pattern] == 0) & (
                        volume_data > 0
                    )
                    if zero_price_nonzero_volume.any():
                        self.issues.append(
                            ValidationIssue(
                                level=ValidationLevel.WARNING,
                                message=f"Zero {pattern} prices with non-zero volume detected",
                                column=col,
                                suggested_action="Verify data quality for zero price periods",
                            )
                        )

        except Exception as e:
            logger.warning(f"OHLC validation failed: {e}")

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types and convert if necessary."""
        for col in df.columns:
            if df[col].dtype == "object":
                # Try to convert to appropriate types
                try:
                    # Try numeric first
                    numeric_data = pd.to_numeric(df[col], errors="coerce")
                    if not numeric_data.isna().all():
                        # Successfully converted to numeric
                        invalid_count = numeric_data.isna().sum() - df[col].isna().sum()
                        if invalid_count > 0:
                            self.issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.WARNING,
                                    message=f"Column {col} contains non-numeric values mixed with numbers",
                                    column=col,
                                    suggested_action="Clean mixed data types",
                                )
                            )
                except Exception:
                    # Try datetime
                    try:
                        datetime_data = pd.to_datetime(df[col], errors="coerce")
                        if not datetime_data.isna().all():
                            self.issues.append(
                                ValidationIssue(
                                    level=ValidationLevel.INFO,
                                    message=f"Column {col} appears to contain datetime data",
                                    column=col,
                                    suggested_action="Convert to datetime type if appropriate",
                                )
                            )
                    except Exception:
                        pass

    def _analyze_missing_data(self, df: pd.DataFrame) -> None:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        for col, (count, pct) in enumerate(zip(missing_counts, missing_percentages)):
            if count > 0:
                level = ValidationLevel.WARNING if pct < 10 else ValidationLevel.ERROR
                self.issues.append(
                    ValidationIssue(
                        level=level,
                        message=f"Column {df.columns[col]} has {count} missing values ({pct:.1f}%)",
                        column=df.columns[col],
                        suggested_action="Consider imputation or data collection improvement",
                    )
                )

        # Check for missing data patterns
        if df.isnull().any(axis=1).any():
            completely_missing_rows = df.isnull().all(axis=1).sum()
            if completely_missing_rows > 0:
                self.issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"{completely_missing_rows} rows are completely empty",
                        suggested_action="Remove completely empty rows",
                    )
                )

    def _detect_duplicates(self, df: pd.DataFrame) -> None:
        """Detect duplicate rows and index values."""
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self.issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Found {duplicate_rows} duplicate rows",
                    suggested_action="Remove duplicates unless they are valid",
                )
            )

        # Check for duplicate index values
        if df.index.duplicated().any():
            dup_indices = df.index[df.index.duplicated()].tolist()
            self.issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    message=f"Duplicate index values found: {len(set(dup_indices))} unique indices",
                    suggested_action="Fix duplicate timestamps or reset index",
                )
            )

    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate data ranges for price and volume columns."""
        for col in df.columns:
            col_lower = col.lower()

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            data = df[col].dropna()
            if len(data) == 0:
                continue

            min_val, max_val = data.min(), data.max()
            mean_val, std_val = data.mean(), data.std()

            # Price column validation
            if any(
                price_word in col_lower
                for price_word in ["open", "high", "low", "close"]
            ):
                # Check for extreme price values
                if max_val > 0 and min_val > 0:
                    price_ratio = max_val / min_val
                    if price_ratio > 1000:  # 1000x price difference
                        self.issues.append(
                            ValidationIssue(
                                level=ValidationLevel.WARNING,
                                message=f"Extreme price range in {col}: {min_val:.2f} to {max_val:.2f}",
                                column=col,
                                suggested_action="Verify for data errors or stock splits",
                            )
                        )

                # Check for unusually high volatility
                if std_val > 0 and mean_val > 0:
                    cv = std_val / mean_val  # Coefficient of variation
                    if cv > 0.5:  # High volatility threshold
                        self.issues.append(
                            ValidationIssue(
                                level=ValidationLevel.INFO,
                                message=f"High volatility in {col}: CV = {cv:.3f}",
                                column=col,
                                suggested_action="Verify if high volatility is expected",
                            )
                        )

            # Volume column validation
            elif "volume" in col_lower or "vol" in col_lower:
                if min_val < 0:
                    self.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.ERROR,
                            message=f"Negative volume values in {col}",
                            column=col,
                            suggested_action="Correct negative volume values",
                        )
                    )

                # Check for zero volume periods
                zero_volume_pct = (data == 0).sum() / len(data) * 100
                if zero_volume_pct > 50:
                    self.issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            message=f"High zero volume percentage in {col}: {zero_volume_pct:.1f}%",
                            column=col,
                            suggested_action="Verify if extended zero volume periods are correct",
                        )
                    )

    def _detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> None:
        """Detect outliers in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].dtype in ["int64", "float64"]:
                data = df[col].dropna()
                if len(data) < 10:  # Need sufficient data for outlier detection
                    continue

                try:
                    if method == "iqr":
                        outliers = self._detect_outliers_iqr(data, threshold)
                    elif method == "zscore":
                        outliers = self._detect_outliers_zscore(data, threshold)
                    elif method == "isolation_forest":
                        outliers = self._detect_outliers_isolation_forest(data)
                    else:
                        continue

                    if outliers.any():
                        outlier_count = outliers.sum()
                        outlier_pct = (outlier_count / len(data)) * 100

                        level = (
                            ValidationLevel.WARNING
                            if outlier_pct < 5
                            else ValidationLevel.ERROR
                        )
                        self.issues.append(
                            ValidationIssue(
                                level=level,
                                message=f"{outlier_count} outliers detected in {col} ({outlier_pct:.1f}%)",
                                column=col,
                                row_indices=df.index[
                                    df[col].notna() & outliers
                                ].tolist(),
                                suggested_action="Review outliers for data quality issues",
                            )
                        )

                except Exception as e:
                    logger.warning(f"Outlier detection failed for {col}: {e}")

    def _detect_outliers_iqr(
        self, data: pd.Series, multiplier: float = 1.5
    ) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (data < lower_bound) | (data > upper_bound)

    def _detect_outliers_zscore(
        self, data: pd.Series, threshold: float = 3.0
    ) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    def _detect_outliers_isolation_forest(self, data: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest

            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)

            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)

            # Return outliers (label = -1)
            return pd.Series(outlier_labels == -1, index=data.index)

        except ImportError:
            logger.warning("scikit-learn not available, falling back to IQR method")
            return self._detect_outliers_iqr(data)

    def _calculate_quality_score(self, df: pd.DataFrame, issue_count: int) -> float:
        """Calculate overall data quality score."""
        base_score = 1.0

        # Deduct points for issues
        critical_issues = sum(
            1 for issue in self.issues if issue.level == ValidationLevel.CRITICAL
        )
        error_issues = sum(
            1 for issue in self.issues if issue.level == ValidationLevel.ERROR
        )
        warning_issues = sum(
            1 for issue in self.issues if issue.level == ValidationLevel.WARNING
        )

        # Weight deductions by severity
        score_deduction = (
            (critical_issues * 0.3) + (error_issues * 0.2) + (warning_issues * 0.1)
        )

        # Deduct for missing data
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        score_deduction += missing_pct * 0.01

        # Deduct for duplicates
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        score_deduction += duplicate_pct * 0.005

        quality_score = max(0.0, base_score - score_deduction)
        return quality_score

    def _generate_summary(
        self, df: pd.DataFrame, quality_score: float
    ) -> Dict[str, Any]:
        """Generate validation summary."""
        issue_counts = {
            "critical": sum(
                1 for issue in self.issues if issue.level == ValidationLevel.CRITICAL
            ),
            "error": sum(
                1 for issue in self.issues if issue.level == ValidationLevel.ERROR
            ),
            "warning": sum(
                1 for issue in self.issues if issue.level == ValidationLevel.WARNING
            ),
            "info": sum(
                1 for issue in self.issues if issue.level == ValidationLevel.INFO
            ),
        }

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score),
            "total_issues": len(self.issues),
            "issue_breakdown": issue_counts,
            "missing_data_percentage": (
                df.isnull().sum().sum() / (len(df) * len(df.columns))
            )
            * 100,
            "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime64"]).columns),
        }

    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade from score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []

        # General recommendations based on issues found
        if any(issue.level == ValidationLevel.CRITICAL for issue in self.issues):
            recommendations.append("Address critical issues before processing data")

        if any("missing" in issue.message.lower() for issue in self.issues):
            recommendations.append("Implement missing data imputation strategy")

        if any("duplicate" in issue.message.lower() for issue in self.issues):
            recommendations.append("Remove or investigate duplicate records")

        if any("outlier" in issue.message.lower() for issue in self.issues):
            recommendations.append("Review and handle outliers appropriately")

        if any("type" in issue.message.lower() for issue in self.issues):
            recommendations.append("Ensure consistent data types across columns")

        # Data-specific recommendations
        ohlc_issues = [
            issue
            for issue in self.issues
            if any(
                col in issue.column.lower() if issue.column else ""
                for col in ["open", "high", "low", "close"]
            )
        ]
        if ohlc_issues:
            recommendations.append("Verify OHLC data consistency and relationships")

        return recommendations

    def handle_missing_data(
        self, df: pd.DataFrame, strategy: str = "interpolate"
    ) -> pd.DataFrame:
        """
        Handle missing data using various strategies.

        Args:
            df: Input DataFrame
            strategy: Handling strategy ('drop', 'interpolate', 'forward_fill', 'backward_fill', 'mean')

        Returns:
            DataFrame with handled missing data
        """
        logger.info(f"Handling missing data with strategy: {strategy}")

        df_clean = df.copy()

        if strategy == "drop":
            df_clean = df_clean.dropna()
        elif strategy == "interpolate":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear")
        elif strategy == "forward_fill":
            df_clean = df_clean.fillna(method="ffill")
        elif strategy == "backward_fill":
            df_clean = df_clean.fillna(method="bfill")
        elif strategy == "mean":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].mean()
            )

        return df_clean

    def detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", columns: Optional[List[str]] = None
    ) -> OutlierReport:
        """
        Detect outliers in specified columns.

        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            columns: Columns to check (default: all numeric columns)

        Returns:
            OutlierReport with detection results
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        all_outliers = []
        all_outlier_values = []
        total_outliers = 0

        for col in columns:
            if col not in df.columns:
                continue

            data = df[col].dropna()
            if len(data) < 10:
                continue

            if method == "iqr":
                outliers = self._detect_outliers_iqr(data)
            elif method == "zscore":
                outliers = self._detect_outliers_zscore(data)
            else:
                outliers = pd.Series(False, index=data.index)

            outlier_indices = data.index[outliers].tolist()
            outlier_values = data[outliers].tolist()

            all_outliers.extend(outlier_indices)
            all_outlier_values.extend(outlier_values)
            total_outliers += len(outlier_indices)

        total_cells = len(df) * len(columns)
        outlier_percentage = (
            (total_outliers / total_cells) * 100 if total_cells > 0 else 0
        )

        return OutlierReport(
            total_outliers=total_outliers,
            outlier_percentage=outlier_percentage,
            outlier_indices=list(set(all_outliers)),
            outlier_values=all_outlier_values,
            method=method,
            threshold=1.5 if method == "iqr" else 3.0,
            summary={
                "columns_checked": columns,
                "total_cells_checked": total_cells,
                "unique_outlier_rows": len(set(all_outliers)),
            },
        )
