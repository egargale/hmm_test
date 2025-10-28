"""
Pipeline stage implementations for the unified data pipeline.

This module provides individual stage implementations that integrate the enhanced
capabilities from Phases 2.1.1-2.1.3 into a unified processing pipeline.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.logging_config import get_logger

from . import feature_engineering
from .csv_format_detector import CSVFormatDetector, DetectionResult
from .data_validator import DataValidator

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    success: bool
    data: Any
    metadata: Dict[str, Any]
    issues: List[str]
    processing_time: float
    stage_name: str


@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""

    config: Dict[str, Any]
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    processing_history: List[Dict[str, Any]]

    def add_stage_result(self, stage_name: str, result: StageResult) -> None:
        """Add stage result to processing history."""
        self.processing_history.append(
            {
                "stage_name": stage_name,
                "timestamp": datetime.now().isoformat(),
                "success": result.success,
                "processing_time": result.processing_time,
                "issues": result.issues,
                "metadata": result.metadata,
            }
        )


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{name}")

    @abstractmethod
    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Process data through this stage."""
        pass

    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for this stage."""
        return True, []

    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about this stage."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
        }


class FormatDetectionStage(PipelineStage):
    """Pipeline stage for CSV format detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FormatDetection", config)
        self.detector = CSVFormatDetector()

    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Detect CSV format from input data."""
        start_time = time.time()
        issues = []

        try:
            self.logger.info("Starting format detection")

            # Handle different input types
            if isinstance(data, (str, tuple)) and len(data) == 2:
                # (file_path, sample_size) tuple
                file_path, sample_size = data
                detection_result = self.detector.detect_format(file_path, sample_size)
            else:
                # Assume it's a file path
                detection_result = self.detector.detect_format(data)

            processing_time = time.time() - start_time

            # Add detection issues to pipeline issues
            issues.extend(detection_result.issues)

            # Update context with format information
            context.metadata["csv_format"] = detection_result.format
            context.metadata["format_confidence"] = detection_result.confidence
            context.metadata["sample_data"] = detection_result.sample_data

            self.logger.info(
                f"Format detection completed: {detection_result.format.format_type} "
                f"(confidence: {detection_result.confidence:.2f})"
            )

            return StageResult(
                success=True,
                data=detection_result,
                metadata={
                    "detected_format": detection_result.format.format_type,
                    "confidence": detection_result.confidence,
                    "sample_rows": len(detection_result.sample_data),
                    "format_details": detection_result.format.__dict__,
                },
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Format detection failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return StageResult(
                success=False,
                data=None,
                metadata={},
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input for format detection."""
        issues = []

        if isinstance(data, str):
            from pathlib import Path

            if not Path(data).exists():
                issues.append(f"File not found: {data}")
        elif isinstance(data, tuple) and len(data) == 2:
            file_path, sample_size = data
            from pathlib import Path

            if not Path(file_path).exists():
                issues.append(f"File not found: {file_path}")
            if not isinstance(sample_size, int) or sample_size <= 0:
                issues.append("Sample size must be a positive integer")
        else:
            issues.append("Input must be file path or (file_path, sample_size) tuple")

        return len(issues) == 0, issues


class DataLoadingStage(PipelineStage):
    """Pipeline stage for loading and standardizing data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DataLoading", config)
        self.encoding = config.get("encoding", "utf-8") if config else "utf-8"

    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Load and standardize data from various sources."""
        start_time = time.time()
        issues = []

        try:
            self.logger.info("Starting data loading and standardization")

            # Get format information from previous stage
            detection_result = context.metadata.get("detection_result")
            if not detection_result:
                # If no detection result, try to load directly
                df = self._load_data_directly(data)
            else:
                df = self._load_with_format_info(data, detection_result)

            # Basic standardization
            df_standardized = self._standardize_dataframe(df)

            processing_time = time.time() - start_time

            # Update context
            context.metadata["original_shape"] = df.shape
            context.metadata["standardized_shape"] = df_standardized.shape
            context.metadata["columns"] = df_standardized.columns.tolist()

            self.logger.info(
                f"Data loading completed: {len(df_standardized)} rows, "
                f"{len(df_standardized.columns)} columns"
            )

            return StageResult(
                success=True,
                data=df_standardized,
                metadata={
                    "rows_loaded": len(df_standardized),
                    "columns_loaded": len(df_standardized.columns),
                    "data_types": df_standardized.dtypes.to_dict(),
                    "memory_usage_mb": df_standardized.memory_usage(deep=True).sum()
                    / (1024 * 1024),
                },
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Data loading failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return StageResult(
                success=False,
                data=None,
                metadata={},
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

    def _load_data_directly(self, data_source: Any) -> pd.DataFrame:
        """Load data directly without format information."""
        from pathlib import Path

        if isinstance(data_source, (str, Path)):
            return pd.read_csv(data_source, encoding=self.encoding)
        elif isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _load_with_format_info(
        self, data_source: Any, detection_result: DetectionResult
    ) -> pd.DataFrame:
        """Load data using format detection information."""
        from pathlib import Path

        if isinstance(data_source, (str, Path)):
            csv_format = detection_result.format

            # Read CSV with detected format
            df = pd.read_csv(
                data_source,
                encoding=csv_format.encoding,
                delimiter=csv_format.delimiter,
                dtype=str,  # Keep as string for pattern matching
            )

            # Apply column mapping if available
            if csv_format.column_mapping:
                df = df.rename(columns=csv_format.column_mapping)

            return df
        else:
            return self._load_data_directly(data_source)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic standardization to DataFrame."""
        df_standardized = df.copy()

        # Strip whitespace from column names
        df_standardized.columns = [
            col.strip().lower().replace(" ", "_") for col in df_standardized.columns
        ]

        # Convert datetime column if present
        datetime_cols = [
            col for col in df_standardized.columns if "datetime" in col or "date" in col
        ]
        if datetime_cols:
            datetime_col = datetime_cols[0]
            try:
                df_standardized[datetime_col] = pd.to_datetime(
                    df_standardized[datetime_col]
                )
                df_standardized = df_standardized.set_index(datetime_col)
            except Exception as e:
                self.logger.warning(f"Failed to convert datetime column: {e}")

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df_standardized.columns:
                df_standardized[col] = pd.to_numeric(
                    df_standardized[col], errors="coerce"
                )

        return df_standardized


class DataValidationStage(PipelineStage):
    """Pipeline stage for data validation and quality assessment."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DataValidation", config)
        self.validator = DataValidator(
            strict_mode=config.get("strict_mode", False) if config else False
        )

    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Validate data and assess quality."""
        start_time = time.time()
        issues = []

        try:
            self.logger.info("Starting data validation")

            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data validation requires pandas DataFrame")

            # Run comprehensive validation
            validation_report = self.validator.validate_dataset(
                data,
                detect_outliers=self.config.get("detect_outliers", True)
                if self.config
                else True,
                outlier_method=self.config.get("outlier_method", "iqr")
                if self.config
                else "iqr",
            )

            processing_time = time.time() - start_time

            # Add validation issues to pipeline issues
            for issue in validation_report.issues:
                issues.append(f"{issue.level.value}: {issue.message}")

            # Update context with validation information
            context.metadata["validation_report"] = validation_report
            context.metadata["quality_score"] = validation_report.quality_score
            context.metadata["validation_summary"] = validation_report.summary

            self.logger.info(
                f"Data validation completed: Quality Score = {validation_report.quality_score:.3f}"
            )

            return StageResult(
                success=validation_report.is_valid,
                data=data,  # Data unchanged, validation is metadata
                metadata={
                    "quality_score": validation_report.quality_score,
                    "total_issues": len(validation_report.issues),
                    "valid_rows": validation_report.valid_rows,
                    "total_rows": validation_report.total_rows,
                    "validation_summary": validation_report.summary,
                },
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Data validation failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return StageResult(
                success=False,
                data=data,
                metadata={},
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )


class FeatureEngineeringStage(PipelineStage):
    """Pipeline stage for enhanced feature engineering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FeatureEngineering", config)
        self.feature_config = config or {}

    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Apply enhanced feature engineering."""
        start_time = time.time()
        issues = []

        try:
            self.logger.info("Starting feature engineering")

            if not isinstance(data, pd.DataFrame):
                raise ValueError("Feature engineering requires pandas DataFrame")

            # Apply enhanced features using Phase 2.1.2 capabilities
            df_with_features = feature_engineering.add_features(
                data, self.feature_config
            )

            processing_time = time.time() - start_time

            # Calculate feature statistics
            feature_count = len(df_with_features.columns) - len(data.columns)
            new_features = [
                col for col in df_with_features.columns if col not in data.columns
            ]

            # Update context
            context.metadata["feature_count"] = feature_count
            context.metadata["new_features"] = new_features
            context.metadata["total_features"] = len(df_with_features.columns)

            self.logger.info(
                f"Feature engineering completed: {feature_count} new features added"
            )

            return StageResult(
                success=True,
                data=df_with_features,
                metadata={
                    "feature_count": feature_count,
                    "new_features": new_features,
                    "total_features": len(df_with_features.columns),
                    "original_columns": len(data.columns),
                    "feature_categories": list(self.feature_config.keys()),
                },
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Feature engineering failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return StageResult(
                success=False,
                data=data,  # Return original data on failure
                metadata={},
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )


class QualityAssessmentStage(PipelineStage):
    """Pipeline stage for final quality assessment and metrics calculation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("QualityAssessment", config)

    def process(self, data: Any, context: PipelineContext) -> StageResult:
        """Perform final quality assessment and generate metrics."""
        start_time = time.time()
        issues = []

        try:
            self.logger.info("Starting final quality assessment")

            if not isinstance(data, pd.DataFrame):
                raise ValueError("Quality assessment requires pandas DataFrame")

            # Calculate comprehensive quality metrics
            quality_metrics = self._calculate_quality_metrics(data, context)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                data, context, quality_metrics
            )

            processing_time = time.time() - start_time

            # Update context with final assessment
            context.metadata["final_quality_metrics"] = quality_metrics
            context.metadata["recommendations"] = recommendations
            context.metrics.update(quality_metrics)

            self.logger.info(
                f"Quality assessment completed: Overall score = {quality_metrics.get('overall_score', 0):.3f}"
            )

            return StageResult(
                success=True,
                data=data,
                metadata={
                    "quality_metrics": quality_metrics,
                    "recommendations": recommendations,
                    "data_shape": data.shape,
                    "feature_count": len(data.columns),
                    "processing_complete": True,
                },
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Quality assessment failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return StageResult(
                success=False,
                data=data,
                metadata={},
                issues=issues,
                processing_time=processing_time,
                stage_name=self.name,
            )

    def _calculate_quality_metrics(
        self, df: pd.DataFrame, context: PipelineContext
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        metrics = {}

        # Data completeness
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        metrics["completeness_score"] = completeness_score

        # Feature quality (if validation was performed)
        if "validation_report" in context.metadata:
            validation_score = context.metadata["validation_report"].quality_score
            metrics["validation_score"] = validation_score

        # Feature diversity
        numeric_columns = df.select_dtypes(include=["number"]).columns
        metrics["feature_diversity"] = {
            "total_features": len(df.columns),
            "numeric_features": len(numeric_columns),
            "datetime_features": len(df.select_dtypes(include=["datetime64"]).columns),
            "object_features": len(df.select_dtypes(include=["object"]).columns),
        }

        # Data range consistency (for OHLCV data)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_ohlcv = [col for col in ohlcv_cols if col in df.columns]
        metrics["ohlcv_coverage"] = len(available_ohlcv) / len(ohlcv_cols)

        # Calculate overall score
        scores = [
            completeness_score,
            metrics.get("validation_score", 0.8),  # Default if no validation
            metrics["ohlcv_coverage"],
        ]
        metrics["overall_score"] = sum(scores) / len(scores)

        return metrics

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        context: PipelineContext,
        quality_metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Completeness recommendations
        if quality_metrics.get("completeness_score", 1) < 0.95:
            recommendations.append(
                "Consider handling missing data to improve completeness"
            )

        # Feature recommendations
        if quality_metrics.get("ohlcv_coverage", 0) < 1.0:
            recommendations.append(
                "Some OHLCV columns are missing - verify data source"
            )

        # Quality recommendations
        if quality_metrics.get("overall_score", 1) < 0.8:
            recommendations.append(
                "Overall data quality is below 80% - review validation issues"
            )

        # Feature engineering recommendations
        if (
            "feature_count" in context.metadata
            and context.metadata["feature_count"] < 20
        ):
            recommendations.append("Consider enabling more feature engineering options")

        # Performance recommendations
        total_processing_time = sum(
            h.get("processing_time", 0) for h in context.processing_history
        )
        if total_processing_time > 10:
            recommendations.append(
                "Consider performance optimizations for large datasets"
            )

        return recommendations


class PipelineStageFactory:
    """Factory for creating pipeline stages."""

    @staticmethod
    def create_stage(
        stage_type: str, config: Optional[Dict[str, Any]] = None
    ) -> PipelineStage:
        """Create a pipeline stage by type."""
        stages = {
            "format_detection": FormatDetectionStage,
            "data_loading": DataLoadingStage,
            "data_validation": DataValidationStage,
            "feature_engineering": FeatureEngineeringStage,
            "quality_assessment": QualityAssessmentStage,
        }

        if stage_type not in stages:
            raise ValueError(
                f"Unknown stage type: {stage_type}. Available: {list(stages.keys())}"
            )

        return stages[stage_type](config)

    @staticmethod
    def get_available_stages() -> List[str]:
        """Get list of available stage types."""
        return [
            "format_detection",
            "data_loading",
            "data_validation",
            "feature_engineering",
            "quality_assessment",
        ]
