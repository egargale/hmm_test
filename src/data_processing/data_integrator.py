"""
Enhanced data integration system for seamless CSV processing and feature engineering.

This module provides integration between CSV processing, format standardization,
and enhanced feature engineering capabilities from Phase 2.1.2.
"""

import pytz
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from ..utils.logging_config import get_logger
from .csv_format_detector import CSVFormat, DetectionResult
from . import feature_engineering
from .performance_optimizer import PerformanceOptimizer

logger = get_logger(__name__)


@dataclass
class ProcessingMetadata:
    """Metadata for data processing operations."""
    source_file: Path
    original_format: str
    processed_format: str
    rows_original: int
    rows_processed: int
    columns_original: int
    columns_processed: int
    processing_time: float
    quality_score: Optional[float] = None
    feature_count: int = 0
    timezone: Optional[str] = None
    data_quality_issues: List[str] = None

    def __post_init__(self):
        if self.data_quality_issues is None:
            self.data_quality_issues = []


@dataclass
class IntegrationResult:
    """Result of data integration process."""
    data: pd.DataFrame
    metadata: ProcessingMetadata
    format_info: CSVFormat
    issues: List[str]
    recommendations: List[str]
    performance_metrics: Dict[str, Any]


class DataIntegrator:
    """
    Enhanced data integrator for seamless CSV processing and feature integration.

    Features:
    - Seamless integration with enhanced feature engineering (Phase 2.1.2)
    - Format standardization across 10+ CSV formats
    - Metadata extraction and preservation
    - Timezone handling and normalization
    - Data lineage tracking
    - Quality metrics calculation
    - Performance optimization integration
    """

    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        self.feature_config = feature_config or {}
        self.performance_optimizer = PerformanceOptimizer()

        # Processing lineage
        self.processing_history: List[ProcessingMetadata] = []

        # Format mappings
        self.standard_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        self.optional_columns = ['adj_close', 'symbol', 'exchange']

        logger.info("Data integrator initialized with enhanced feature engineering support")

    def integrate_with_features(self, df: pd.DataFrame, detection_result: DetectionResult,
                              feature_config: Optional[Dict[str, Any]] = None) -> IntegrationResult:
        """
        Integrate CSV data with enhanced feature engineering.

        Args:
            df: Input DataFrame
            detection_result: Format detection result
            feature_config: Feature engineering configuration

        Returns:
            IntegrationResult with processed data and metadata
        """
        start_time = datetime.now()
        logger.info(f"Starting data integration for {len(df)} rows")

        issues = []
        recommendations = []

        try:
            # Standardize format
            df_standardized, format_issues = self.standardize_format(df, detection_result.format)
            issues.extend(format_issues)

            # Handle timezone
            df_tz, tz_issues = self.normalize_timezones(df_standardized)
            issues.extend(tz_issues)

            # Apply enhanced feature engineering from Phase 2.1.2
            df_features, feature_metadata = self._apply_enhanced_features(
                df_tz, feature_config or {}
            )

            # Extract metadata
            metadata = self._extract_processing_metadata(
                df, df_features, detection_result, start_time, feature_metadata
            )

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df_features)

            # Performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'rows_per_second': len(df_features) / processing_time if processing_time > 0 else 0,
                'feature_generation_time': feature_metadata.get('processing_time', 0),
                'quality_score': quality_metrics.get('overall_score', 0)
            }

            # Generate recommendations
            recommendations.extend(self._generate_integration_recommendations(
                df_features, metadata, quality_metrics
            ))

            result = IntegrationResult(
                data=df_features,
                metadata=metadata,
                format_info=detection_result.format,
                issues=issues,
                recommendations=recommendations,
                performance_metrics=performance_metrics
            )

            # Store processing lineage
            self.processing_history.append(metadata)

            logger.info(f"Data integration completed: {len(df_features)} rows, "
                       f"{len(df_features.columns)} columns, "
                       f"{feature_metadata.get('feature_count', 0)} features added")

            return result

        except Exception as e:
            logger.error(f"Data integration failed: {e}")
            raise

    def standardize_format(self, df: pd.DataFrame, csv_format: CSVFormat,
                          target_format: str = 'standard') -> Tuple[pd.DataFrame, List[str]]:
        """
        Standardize DataFrame to target format.

        Args:
            df: Input DataFrame
            csv_format: Detected CSV format
            target_format: Target format ('standard')

        Returns:
            Tuple of (standardized_df, issues)
        """
        issues = []
        df_standardized = df.copy()

        try:
            # Apply column mapping
            if csv_format.column_mapping:
                df_standardized = df_standardized.rename(columns=csv_format.column_mapping)

                # Check for unmapped columns
                mapped_columns = set(csv_format.column_mapping.values())
                available_columns = set(df_standardized.columns)
                unmapped = available_columns - mapped_columns

                if unmapped:
                    issues.append(f"Unmapped columns preserved: {list(unmapped)}")

            # Standardize column names to lowercase
            df_standardized.columns = [col.lower().replace(' ', '_') for col in df_standardized.columns]

            # Ensure required columns exist
            missing_standard = [col for col in self.standard_columns if col not in df_standardized.columns]
            if missing_standard:
                issues.append(f"Missing standard columns: {missing_standard}")

            # Convert to standard datetime format
            df_standardized = self._standardize_datetime(df_standardized, csv_format)

            # Standardize numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df_standardized.columns:
                    df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce')

            # Handle adj_close -> close if close missing
            if 'close' not in df_standardized.columns and 'adj_close' in df_standardized.columns:
                df_standardized['close'] = df_standardized['adj_close']
                issues.append("Using adj_close as close price")

            # Set datetime index
            if 'datetime' in df_standardized.columns:
                df_standardized = df_standardized.set_index('datetime')

            # Sort by datetime
            if df_standardized.index.name == 'datetime':
                df_standardized = df_standardized.sort_index()

            logger.debug(f"Format standardized: {len(df_standardized)} rows, {len(df_standardized.columns)} columns")

        except Exception as e:
            issues.append(f"Format standardization failed: {e}")
            raise

        return df_standardized, issues

    def _standardize_datetime(self, df: pd.DataFrame, csv_format: CSVFormat) -> pd.DataFrame:
        """Standardize datetime columns."""
        df_datetime = df.copy()

        try:
            if csv_format.date_column and csv_format.time_column:
                # Split datetime columns
                if csv_format.date_column in df.columns and csv_format.time_column in df.columns:
                    df_datetime['datetime'] = pd.to_datetime(
                        df_datetime[csv_format.date_column] + ' ' + df_datetime[csv_format.time_column]
                    )
                    df_datetime = df_datetime.drop([csv_format.date_column, csv_format.time_column], axis=1)

            elif csv_format.datetime_format == 'unix':
                # Unix timestamp conversion
                datetime_col = self._find_datetime_column(df)
                if datetime_col:
                    df_datetime['datetime'] = pd.to_datetime(df_datetime[datetime_col], unit='s')
                    df_datetime = df_datetime.drop(datetime_col, axis=1)

            elif csv_format.datetime_format == 'ISO8601':
                # ISO8601 format
                datetime_col = self._find_datetime_column(df)
                if datetime_col:
                    df_datetime['datetime'] = pd.to_datetime(df_datetime[datetime_col])
                    df_datetime = df_datetime.drop(datetime_col, axis=1)

            else:
                # Standard datetime column
                datetime_col = self._find_datetime_column(df)
                if datetime_col and datetime_col != 'datetime':
                    df_datetime['datetime'] = pd.to_datetime(df_datetime[datetime_col])
                    if datetime_col != 'datetime':
                        df_datetime = df_datetime.drop(datetime_col, axis=1)

        except Exception as e:
            logger.warning(f"Datetime standardization failed: {e}")

        return df_datetime

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column in DataFrame."""
        datetime_patterns = ['datetime', 'date', 'time', 'timestamp']

        for col in df.columns:
            if any(pattern in col.lower() for pattern in datetime_patterns):
                return col

        return None

    def normalize_timezones(self, df: pd.DataFrame, target_timezone: str = 'UTC') -> Tuple[pd.DataFrame, List[str]]:
        """
        Normalize timezone information.

        Args:
            df: Input DataFrame with datetime index
            target_timezone: Target timezone

        Returns:
            Tuple of (timezone_normalized_df, issues)
        """
        issues = []

        if df.index.name != 'datetime' or not pd.api.types.is_datetime64_any_dtype(df.index):
            return df, issues

        try:
            df_tz = df.copy()

            # Check if datetime has timezone info
            if df_tz.index.tz is None:
                # Assume timezone based on data characteristics or default to UTC
                df_tz.index = df_tz.index.tz_localize('UTC')
                issues.append("Assumed UTC timezone for datetime data")
            else:
                # Convert to target timezone
                df_tz.index = df_tz.index.tz_convert(target_timezone)

            logger.debug(f"Timezone normalized to {target_timezone}")

        except Exception as e:
            issues.append(f"Timezone normalization failed: {e}")
            return df, issues

        return df_tz, issues

    def _apply_enhanced_features(self, df: pd.DataFrame,
                               feature_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply enhanced feature engineering from Phase 2.1.2."""
        try:
            start_time = datetime.now()

            # Use the enhanced feature engineer from Phase 2.1.2
            df_features = feature_engineering.add_features(df, feature_config)

            processing_time = (datetime.now() - start_time).total_seconds()
            feature_count = len(df_features.columns) - len(df.columns)

            feature_metadata = {
                'feature_count': feature_count,
                'processing_time': processing_time,
                'total_features': len(df_features.columns),
                'original_columns': len(df.columns),
                'new_features': [col for col in df_features.columns if col not in df.columns]
            }

            logger.debug(f"Enhanced features applied: {feature_count} new features in {processing_time:.3f}s")

            return df_features, feature_metadata

        except Exception as e:
            logger.error(f"Enhanced feature application failed: {e}")
            raise

    def extract_metadata(self, file_path: Path, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from file and DataFrame.

        Args:
            file_path: Source file path
            df: Processed DataFrame

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # File metadata
        try:
            stat = file_path.stat()
            metadata['file_info'] = {
                'name': file_path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': file_path.suffix
            }
        except Exception as e:
            logger.warning(f"File metadata extraction failed: {e}")

        # Data metadata
        try:
            metadata['data_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'column_types': df.dtypes.to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }

            # Date range
            if df.index.name == 'datetime' and pd.api.types.is_datetime64_any_dtype(df.index):
                metadata['date_range'] = {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat(),
                    'duration_days': (df.index.max() - df.index.min()).days
                }

        except Exception as e:
            logger.warning(f"Data metadata extraction failed: {e}")

        return metadata

    def _extract_processing_metadata(self, original_df: pd.DataFrame,
                                   processed_df: pd.DataFrame,
                                   detection_result: DetectionResult,
                                   start_time: datetime,
                                   feature_metadata: Dict[str, Any]) -> ProcessingMetadata:
        """Extract comprehensive processing metadata."""
        processing_time = (datetime.now() - start_time).total_seconds()

        return ProcessingMetadata(
            source_file=Path("unknown"),  # Will be set by caller
            original_format=detection_result.format.format_type,
            processed_format='standard_enhanced',
            rows_original=len(original_df),
            rows_processed=len(processed_df),
            columns_original=len(original_df.columns),
            columns_processed=len(processed_df.columns),
            processing_time=processing_time,
            feature_count=feature_metadata.get('feature_count', 0),
            timezone='UTC',  # Default
            data_quality_issues=detection_result.issues
        )

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        quality_metrics = {}

        try:
            # Missing data metrics
            missing_data = df.isnull().sum()
            total_cells = len(df) * len(df.columns)
            missing_percentage = (missing_data.sum() / total_cells) * 100

            quality_metrics['missing_data'] = {
                'total_missing': missing_data.sum(),
                'missing_percentage': missing_percentage,
                'columns_with_missing': missing_data[missing_data > 0].to_dict()
            }

            # Data consistency metrics (OHLC)
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_issues = 0
                # High >= max(Open, Close)
                high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
                # Low <= min(Open, Close)
                low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
                ohlc_issues = high_violations + low_violations

                quality_metrics['ohlc_consistency'] = {
                    'total_violations': ohlc_issues,
                    'violation_percentage': (ohlc_issues / len(df)) * 100,
                    'high_violations': high_violations,
                    'low_violations': low_violations
                }

            # Numeric range metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Check for extreme values
                extreme_values = {}
                for col in numeric_cols:
                    if df[col].dtype in ['float64', 'int64']:
                        q1, q3 = df[col].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
                        if outliers > 0:
                            extreme_values[col] = outliers

                quality_metrics['outliers'] = extreme_values

            # Overall quality score
            base_score = 1.0
            score_deductions = 0

            # Deduct for missing data
            score_deductions += (missing_percentage / 100) * 0.3

            # Deduct for OHLC issues
            if 'ohlc_consistency' in quality_metrics:
                ohlc_violation_pct = quality_metrics['ohlc_consistency']['violation_percentage']
                score_deductions += (ohlc_violation_pct / 100) * 0.4

            # Deduct for outliers
            if 'outliers' in quality_metrics:
                total_outliers = sum(quality_metrics['outliers'].values())
                outlier_percentage = (total_outliers / (len(df) * len(numeric_cols))) * 100
                score_deductions += (outlier_percentage / 100) * 0.2

            quality_score = max(0.0, base_score - score_deductions)
            quality_metrics['overall_score'] = quality_score

        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            quality_metrics['overall_score'] = 0.5  # Default score

        return quality_metrics

    def _generate_integration_recommendations(self, df: pd.DataFrame,
                                            metadata: ProcessingMetadata,
                                            quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate integration-specific recommendations."""
        recommendations = []

        # Quality-based recommendations
        if quality_metrics.get('overall_score', 1.0) < 0.8:
            recommendations.append("Data quality score is below 80%, consider data cleaning")

        missing_pct = quality_metrics.get('missing_data', {}).get('missing_percentage', 0)
        if missing_pct > 5:
            recommendations.append(f"High missing data percentage ({missing_pct:.1f}%), consider imputation")

        # Performance-based recommendations
        if metadata.processing_time > 10:
            recommendations.append("Long processing time, consider enabling performance optimizations")

        # Feature-based recommendations
        if metadata.feature_count < 20:
            recommendations.append("Consider enabling more feature engineering options")

        # Data consistency recommendations
        if 'ohlc_consistency' in quality_metrics:
            violations = quality_metrics['ohlc_consistency']['total_violations']
            if violations > 0:
                recommendations.append(f"OHLC consistency issues detected ({violations} violations)")

        return recommendations

    def get_processing_lineage(self) -> List[Dict[str, Any]]:
        """Get processing history/lineage information."""
        return [
            {
                'source_file': meta.source_file.name,
                'original_format': meta.original_format,
                'rows_processed': meta.rows_processed,
                'processing_time': meta.processing_time,
                'feature_count': meta.feature_count,
                'quality_issues': len(meta.data_quality_issues)
            }
            for meta in self.processing_history
        ]

    def clear_processing_history(self) -> None:
        """Clear processing history."""
        self.processing_history.clear()
        logger.info("Processing history cleared")

    def get_supported_features(self) -> Dict[str, Any]:
        """Get information about supported features and capabilities."""
        return {
            'supported_formats': [
                'standard_ohlcv', 'split_datetime', 'tradingview', 'yahoo_finance',
                'alpha_vantage', 'iso_8601', 'unix_timestamp', 'multi_asset'
            ],
            'feature_engineering_categories': [
                'basic_indicators', 'enhanced_momentum', 'enhanced_volatility',
                'enhanced_trend', 'enhanced_volume', 'time_features'
            ],
            'optimization_features': [
                'parallel_processing', 'memory_mapping', 'adaptive_chunking',
                'data_type_optimization', 'vectorized_operations'
            ],
            'validation_features': [
                'ohlc_consistency', 'outlier_detection', 'missing_data_analysis',
                'range_validation', 'quality_scoring'
            ]
        }