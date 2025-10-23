"""
Output management system for the unified data pipeline.

This module provides comprehensive output management, supporting multiple
formats and automatic metadata preservation.
"""

from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime

import pandas as pd

from ..utils.logging_config import get_logger
from .pipeline_config import OutputFormat

logger = get_logger(__name__)


@dataclass
class OutputPackage:
    """Complete output package with data, metadata, and reports."""
    data: Any
    metadata: Dict[str, Any]
    quality_report: Optional[Dict[str, Any]] = None
    processing_log: Optional[List[Dict[str, Any]]] = None
    pipeline_summary: Optional[Dict[str, Any]] = None


@dataclass
class OutputResult:
    """Result of output operation."""
    success: bool
    output_path: Optional[Path] = None
    format: Optional[str] = None
    size_mb: Optional[float] = None
    processing_time: float = 0.0
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class OutputManager:
    """
    Manages pipeline output in multiple formats with metadata preservation.

    Supports:
    - pandas DataFrame (in-memory)
    - CSV files
    - Parquet files
    - JSON files
    - Pickle files
    - Automatic metadata preservation
    - Quality report generation
    """

    def __init__(self, default_format: OutputFormat = OutputFormat.DATAFRAME):
        self.default_format = default_format
        self.logger = get_logger(f"{__name__}.OutputManager")

    def save_dataframe(self, df: pd.DataFrame, path: Path, format: str = "csv",
                      compression: Optional[str] = None, **kwargs) -> OutputResult:
        """
        Save DataFrame to specified format.

        Args:
            df: DataFrame to save
            path: Output file path
            format: Output format (csv, parquet, json, pickle)
            compression: Compression method
            **kwargs: Additional arguments for pandas save functions

        Returns:
            OutputResult with operation details
        """
        start_time = datetime.now()
        issues = []

        try:
            self.logger.info(f"Saving DataFrame to {format.upper()}: {path}")

            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save based on format
            if format.lower() == "csv":
                df.to_csv(path, compression=compression, **kwargs)
            elif format.lower() == "parquet":
                df.to_parquet(path, compression=compression, **kwargs)
            elif format.lower() == "json":
                df.to_json(path, compression=compression, **kwargs)
            elif format.lower() == "pickle":
                df.to_pickle(path, compression=compression, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Calculate file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Successfully saved {len(df)} rows to {path} ({file_size_mb:.2f} MB)")

            return OutputResult(
                success=True,
                output_path=path,
                format=format,
                size_mb=file_size_mb,
                processing_time=processing_time,
                issues=issues
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to save DataFrame to {path}: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return OutputResult(
                success=False,
                processing_time=processing_time,
                issues=issues
            )

    def save_metadata(self, metadata: Dict[str, Any], path: Path,
                     format: str = "json") -> bool:
        """
        Save metadata to file.

        Args:
            metadata: Metadata dictionary
            path: Output file path
            format: Output format (json, pickle)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Saving metadata to {path}")

            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                with open(path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            elif format.lower() == "pickle":
                with open(path, 'wb') as f:
                    pickle.dump(metadata, f)
            else:
                raise ValueError(f"Unsupported metadata format: {format}")

            self.logger.info(f"Metadata saved successfully to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save metadata to {path}: {e}")
            return False

    def save_quality_report(self, quality_report: Dict[str, Any], path: Path) -> bool:
        """
        Save quality report to file.

        Args:
            quality_report: Quality report dictionary
            path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Saving quality report to {path}")

            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JSON with formatting
            with open(path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)

            self.logger.info(f"Quality report saved successfully to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save quality report to {path}: {e}")
            return False

    def create_output_package(self, result_data: Any, metadata: Dict[str, Any],
                           quality_report: Optional[Dict[str, Any]] = None,
                           processing_log: Optional[List[Dict[str, Any]]] = None,
                           pipeline_summary: Optional[Dict[str, Any]] = None) -> OutputPackage:
        """
        Create a complete output package.

        Args:
            result_data: Main result data (usually DataFrame)
            metadata: Processing metadata
            quality_report: Optional quality assessment report
            processing_log: Optional processing log
            pipeline_summary: Optional pipeline execution summary

        Returns:
            Complete OutputPackage
        """
        return OutputPackage(
            data=result_data,
            metadata=metadata,
            quality_report=quality_report,
            processing_log=processing_log,
            pipeline_summary=pipeline_summary
        )

    def save_output_package(self, package: OutputPackage, base_path: Path,
                          data_format: str = "csv", include_metadata: bool = True,
                          include_quality_report: bool = True) -> Dict[str, OutputResult]:
        """
        Save complete output package to files.

        Args:
            package: OutputPackage to save
            base_path: Base directory for output files
            data_format: Format for main data file
            include_metadata: Whether to save metadata
            include_quality_report: Whether to save quality report

        Returns:
            Dictionary with save results for each component
        """
        results = {}

        try:
            self.logger.info(f"Saving output package to {base_path}")

            # Create base directory
            base_path.mkdir(parents=True, exist_ok=True)

            # Save main data
            if isinstance(package.data, pd.DataFrame):
                data_path = base_path / f"data.{data_format}"
                data_result = self.save_dataframe(package.data, data_path, data_format)
                results['data'] = data_result
            else:
                self.logger.warning("Main data is not a DataFrame, skipping data save")

            # Save metadata
            if include_metadata and package.metadata:
                metadata_path = base_path / "metadata.json"
                metadata_saved = self.save_metadata(package.metadata, metadata_path)
                results['metadata'] = OutputResult(success=metadata_saved, output_path=metadata_path)

            # Save quality report
            if include_quality_report and package.quality_report:
                quality_path = base_path / "quality_report.json"
                quality_saved = self.save_quality_report(package.quality_report, quality_path)
                results['quality_report'] = OutputResult(success=quality_saved, output_path=quality_path)

            # Save processing log
            if package.processing_log:
                log_path = base_path / "processing_log.json"
                log_saved = self.save_metadata({'processing_log': package.processing_log}, log_path)
                results['processing_log'] = OutputResult(success=log_saved, output_path=log_path)

            # Save pipeline summary
            if package.pipeline_summary:
                summary_path = base_path / "pipeline_summary.json"
                summary_saved = self.save_metadata(package.pipeline_summary, summary_path)
                results['pipeline_summary'] = OutputResult(success=summary_saved, output_path=summary_path)

            successful_saves = sum(1 for result in results.values() if result.success)
            self.logger.info(f"Output package saved: {successful_saves}/{len(results)} components successful")

            return results

        except Exception as e:
            self.logger.error(f"Failed to save output package: {e}")
            error_result = OutputResult(success=False, issues=[str(e)])
            return {'error': error_result}

    def format_output_data(self, data: Any, output_format: OutputFormat,
                          column_prefix: Optional[str] = None,
                          index_column: Optional[str] = None,
                          date_format: str = "%Y-%m-%d %H:%M:%S") -> Any:
        """
        Format data according to output specifications.

        Args:
            data: Data to format (usually DataFrame)
            output_format: Desired output format
            column_prefix: Prefix to add to column names
            index_column: Column to use as index
            date_format: Format for datetime columns

        Returns:
            Formatted data
        """
        try:
            if not isinstance(data, pd.DataFrame):
                return data

            df = data.copy()

            # Add column prefix if specified
            if column_prefix:
                df.columns = [f"{column_prefix}_{col}" for col in df.columns]

            # Set index column if specified
            if index_column and index_column in df.columns:
                df = df.set_index(index_column)

            # Format datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df[col] = df[col].dt.strftime(date_format)

            # Handle different output formats
            if output_format == OutputFormat.DATAFRAME:
                return df
            elif output_format == OutputFormat.CSV:
                return df
            elif output_format == OutputFormat.JSON:
                return df.to_dict('records')
            elif output_format == OutputFormat.PARQUET:
                return df
            elif output_format == OutputFormat.PICKLE:
                return df
            else:
                return df

        except Exception as e:
            self.logger.error(f"Failed to format output data: {e}")
            return data

    def generate_output_summary(self, package: OutputPackage) -> Dict[str, Any]:
        """
        Generate summary of output package.

        Args:
            package: OutputPackage to summarize

        Returns:
            Summary dictionary
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'package_components': [],
            'data_summary': {},
            'quality_summary': {},
            'processing_summary': {}
        }

        # Data summary
        if isinstance(package.data, pd.DataFrame):
            summary['data_summary'] = {
                'rows': len(package.data),
                'columns': len(package.data.columns),
                'memory_usage_mb': package.data.memory_usage(deep=True).sum() / (1024 * 1024),
                'column_types': package.data.dtypes.value_counts().to_dict(),
                'has_datetime_index': isinstance(package.data.index, pd.DatetimeIndex)
            }

        # Package components
        if package.metadata:
            summary['package_components'].append('metadata')
        if package.quality_report:
            summary['package_components'].append('quality_report')
        if package.processing_log:
            summary['package_components'].append('processing_log')
        if package.pipeline_summary:
            summary['package_components'].append('pipeline_summary')

        # Quality summary
        if package.quality_report:
            summary['quality_summary'] = {
                'overall_score': package.quality_report.get('overall_score'),
                'total_issues': len(package.quality_report.get('issues', [])),
                'recommendations_count': len(package.quality_report.get('recommendations', []))
            }

        # Processing summary
        if package.pipeline_summary:
            summary['processing_summary'] = {
                'pipeline_name': package.pipeline_summary.get('pipeline_name'),
                'total_stages': len(package.pipeline_summary.get('stages', [])),
                'total_time': package.pipeline_summary.get('total_processing_time')
            }

        return summary

    def create_output_filename(self, base_name: str, output_format: OutputFormat,
                             timestamp: bool = True) -> str:
        """
        Generate output filename with appropriate extension.

        Args:
            base_name: Base filename
            output_format: Output format
            timestamp: Whether to add timestamp

        Returns:
            Generated filename
        """
        filename = base_name

        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp_str}"

        # Add appropriate extension
        extensions = {
            OutputFormat.DATAFRAME: "pkl",  # DataFrame saved as pickle
            OutputFormat.CSV: "csv",
            OutputFormat.PARQUET: "parquet",
            OutputFormat.JSON: "json",
            OutputFormat.PICKLE: "pkl"
        }

        extension = extensions.get(output_format, "csv")
        return f"{filename}.{extension}"

    def validate_output_path(self, path: Path, overwrite: bool = False) -> List[str]:
        """
        Validate output path and return any issues.

        Args:
            path: Output path to validate
            overwrite: Whether overwriting existing files is allowed

        Returns:
            List of validation issues
        """
        issues = []

        try:
            # Check if parent directory exists or can be created
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create parent directory: {e}")

            # Check if file exists and overwrite is not allowed
            if path.exists() and not overwrite:
                issues.append(f"Output file already exists: {path}")

            # Check write permissions
            try:
                # Test write permission by creating a temporary file
                test_file = path.with_suffix('.tmp')
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                issues.append(f"No write permission for output path: {e}")

        except Exception as e:
            issues.append(f"Path validation failed: {e}")

        return issues

    def get_output_size_estimate(self, data: Any, output_format: OutputFormat) -> Dict[str, float]:
        """
        Estimate output file size for different formats.

        Args:
            data: Data to be saved
            output_format: Output format

        Returns:
            Size estimates in MB
        """
        if not isinstance(data, pd.DataFrame):
            return {'estimated_mb': 0.0}

        try:
            # Calculate memory usage
            memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)

            # Estimate compression ratios by format
            compression_ratios = {
                OutputFormat.DATAFRAME: 1.0,      # Pickle (no compression)
                OutputFormat.CSV: 0.8,            # CSV (text)
                OutputFormat.PARQUET: 0.3,        # Parquet (compressed)
                OutputFormat.JSON: 1.2,           # JSON (text, verbose)
                OutputFormat.PICKLE: 1.0          # Pickle (no compression)
            }

            ratio = compression_ratios.get(output_format, 1.0)
            estimated_mb = memory_mb * ratio

            return {
                'memory_mb': memory_mb,
                'estimated_mb': estimated_mb,
                'compression_ratio': ratio
            }

        except Exception as e:
            self.logger.warning(f"Failed to estimate output size: {e}")
            return {'estimated_mb': 0.0}