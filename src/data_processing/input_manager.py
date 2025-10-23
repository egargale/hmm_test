"""
Input management system for the unified data pipeline.

This module provides comprehensive input source management, supporting multiple
data sources and formats with automatic validation and preprocessing.
"""

from typing import Union, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from ..utils.logging_config import get_logger
from .pipeline_config import InputSourceType

logger = get_logger(__name__)


@dataclass
class InputData:
    """Container for input data with metadata."""
    data: Any
    source_type: InputSourceType
    source_path: Optional[Path] = None
    metadata: Dict[str, Any] = None
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class DataInputManager:
    """
    Manages multiple input sources and formats for the pipeline.

    Supports:
    - CSV files with automatic format detection
    - pandas DataFrames
    - Multiple encodings
    - Input validation and preprocessing
    - Metadata extraction
    """

    def __init__(self, default_encoding: str = "utf-8"):
        self.default_encoding = default_encoding
        self.logger = get_logger(f"{__name__}.DataInputManager")

    def load_from_csv(self, file_path: Union[str, Path], encoding: Optional[str] = None,
                     sample_size: Optional[int] = None, **kwargs) -> InputData:
        """
        Load data from CSV file with automatic preprocessing.

        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detect if None)
            sample_size: Number of rows to sample (None for all rows)
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            InputData object with loaded data and metadata
        """
        try:
            file_path = Path(file_path)
            encoding = encoding or self.default_encoding

            self.logger.info(f"Loading CSV file: {file_path}")

            # Validate file existence
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            # Detect encoding if not specified
            if encoding == "auto":
                encoding = self._detect_encoding(file_path)

            # Load data with options
            read_kwargs = {
                'encoding': encoding,
                'dtype': str,  # Keep as string for format detection
                **kwargs
            }

            if sample_size:
                read_kwargs['nrows'] = sample_size

            # Read CSV data
            data = pd.read_csv(file_path, **read_kwargs)

            # Extract metadata
            metadata = self._extract_file_metadata(file_path, data, encoding)

            self.logger.info(f"CSV loaded successfully: {len(data)} rows, {len(data.columns)} columns")

            return InputData(
                data=data,
                source_type=InputSourceType.CSV_FILE,
                source_path=file_path,
                metadata=metadata,
                encoding=encoding
            )

        except Exception as e:
            self.logger.error(f"Failed to load CSV file {file_path}: {e}")
            raise

    def load_from_dataframe(self, df: pd.DataFrame, source_name: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> InputData:
        """
        Load data from pandas DataFrame.

        Args:
            df: Input DataFrame
            source_name: Optional name for the data source
            metadata: Additional metadata

        Returns:
            InputData object with DataFrame and metadata
        """
        try:
            self.logger.info(f"Loading DataFrame: {len(df)} rows, {len(df.columns)} columns")

            # Validate DataFrame
            if df.empty:
                raise ValueError("Input DataFrame is empty")

            # Extract basic metadata
            df_metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'source_name': source_name or "dataframe"
            }

            # Check for datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                df_metadata['has_datetime_index'] = True
                df_metadata['date_range'] = {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat(),
                    'duration_days': (df.index.max() - df.index.min()).days
                }

            # Merge with provided metadata
            if metadata:
                df_metadata.update(metadata)

            return InputData(
                data=df.copy(),
                source_type=InputSourceType.DATAFRAME,
                metadata=df_metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to load DataFrame: {e}")
            raise

    def load_from_source(self, source: Union[str, Path, pd.DataFrame],
                        source_type: Optional[InputSourceType] = None,
                        **kwargs) -> InputData:
        """
        Load data from various sources with automatic type detection.

        Args:
            source: Data source (file path or DataFrame)
            source_type: Explicit source type (auto-detect if None)
            **kwargs: Additional arguments for specific source types

        Returns:
            InputData object with loaded data
        """
        # Auto-detect source type if not specified
        if source_type is None:
            source_type = self._detect_source_type(source)

        # Load based on source type
        if source_type == InputSourceType.CSV_FILE:
            return self.load_from_csv(source, **kwargs)
        elif source_type == InputSourceType.DATAFRAME:
            return self.load_from_dataframe(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def validate_input(self, input_data: InputData) -> ValidationResult:
        """
        Validate input data and return validation results.

        Args:
            input_data: InputData object to validate

        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}

        try:
            data = input_data.data

            # Basic data validation
            if data is None:
                issues.append("Input data is None")
                return ValidationResult(False, issues, warnings, recommendations, metadata)

            if isinstance(data, pd.DataFrame):
                # DataFrame validation
                if data.empty:
                    issues.append("Input DataFrame is empty")

                # Check for completely empty columns
                empty_cols = data.columns[data.isnull().all()].tolist()
                if empty_cols:
                    warnings.append(f"Empty columns detected: {empty_cols}")
                    recommendations.append("Consider removing empty columns")

                # Check for duplicate columns
                if data.columns.duplicated().any():
                    issues.append("Duplicate column names detected")
                    recommendations.append("Remove duplicate column names")

                # Check data size
                if len(data) < 10:
                    warnings.append("Very small dataset (less than 10 rows)")

                # Check for OHLCV columns
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                available_ohlcv = [col for col in ohlcv_cols if col in data.columns.str.lower()]
                metadata['ohlcv_coverage'] = len(available_ohlcv) / len(ohlcv_cols)

                if len(available_ohlcv) < 3:
                    recommendations.append("Consider adding more OHLCV columns for better analysis")

                # Metadata extraction
                metadata.update({
                    'data_shape': data.shape,
                    'column_count': len(data.columns),
                    'row_count': len(data),
                    'dtypes': data.dtypes.to_dict(),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
                })

            else:
                warnings.append(f"Data type not fully validated: {type(data)}")

            # Check encoding for file sources
            if input_data.source_type == InputSourceType.CSV_FILE:
                metadata['encoding'] = input_data.encoding

            is_valid = len(issues) == 0

            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata=metadata
            )

        except Exception as e:
            error_msg = f"Input validation failed: {e}"
            self.logger.error(error_msg)
            issues.append(error_msg)

            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                metadata=metadata
            )

    def get_input_metadata(self, input_data: InputData) -> Dict[str, Any]:
        """
        Get comprehensive metadata for input data.

        Args:
            input_data: InputData object

        Returns:
            Dictionary with comprehensive metadata
        """
        metadata = input_data.metadata.copy()
        metadata.update({
            'source_type': input_data.source_type.value,
            'encoding': input_data.encoding,
            'source_path': str(input_data.source_path) if input_data.source_path else None
        })

        # Add data-specific metadata
        if isinstance(input_data.data, pd.DataFrame):
            df = input_data.data

            # Data shape and size
            metadata['data_shape'] = df.shape
            metadata['total_cells'] = df.shape[0] * df.shape[1]

            # Missing data statistics
            missing_data = df.isnull().sum()
            metadata['missing_data'] = {
                'total_missing': missing_data.sum(),
                'missing_percentage': (missing_data.sum() / metadata['total_cells']) * 100,
                'columns_with_missing': missing_data[missing_data > 0].to_dict()
            }

            # Column statistics
            metadata['column_statistics'] = {
                'numeric_columns': len(df.select_dtypes(include=['number']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
                'object_columns': len(df.select_dtypes(include=['object']).columns),
                'all_columns': df.columns.tolist()
            }

            # Index information
            if isinstance(df.index, pd.DatetimeIndex):
                metadata['index_info'] = {
                    'type': 'datetime',
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat(),
                    'frequency': self._detect_frequency(df.index)
                }

        return metadata

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            import chardet

            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')

            self.logger.debug(f"Detected encoding: {encoding} (confidence: {result.get('confidence', 0):.2f})")
            return encoding

        except ImportError:
            self.logger.warning("chardet not available, using default encoding")
            return self.default_encoding
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using default encoding")
            return self.default_encoding

    def _extract_file_metadata(self, file_path: Path, data: pd.DataFrame, encoding: str) -> Dict[str, Any]:
        """Extract metadata from file and data."""
        try:
            import os

            # File metadata
            stat = file_path.stat()
            file_metadata = {
                'file_name': file_path.name,
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'encoding': encoding
            }

            # Data metadata
            data_metadata = {
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': data.columns.tolist(),
                'data_types': data.dtypes.to_dict()
            }

            # Combine metadata
            metadata = {**file_metadata, **data_metadata}

            # Check for datetime patterns in columns
            datetime_candidates = []
            for col in data.columns:
                if any(pattern in col.lower() for pattern in ['date', 'time', 'datetime']):
                    datetime_candidates.append(col)

            metadata['datetime_column_candidates'] = datetime_candidates

            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to extract file metadata: {e}")
            return {}

    def _detect_source_type(self, source: Any) -> InputSourceType:
        """Auto-detect source type."""
        if isinstance(source, pd.DataFrame):
            return InputSourceType.DATAFRAME
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.suffix.lower() == '.csv':
                return InputSourceType.CSV_FILE
            else:
                raise ValueError(f"Cannot determine source type for: {source}")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def _detect_frequency(self, datetime_index: pd.DatetimeIndex) -> Optional[str]:
        """Detect frequency of datetime index."""
        try:
            # Try pandas frequency detection
            inferred_freq = pd.infer_freq(datetime_index)
            if inferred_freq:
                return inferred_freq

            # Manual detection for common frequencies
            if len(datetime_index) > 1:
                time_diffs = datetime_index.to_series().diff().dropna()
                median_diff = time_diffs.median()

                if median_diff.total_seconds() <= 3600:  # <= 1 hour
                    return "intraday"
                elif median_diff.total_seconds() <= 86400:  # <= 1 day
                    return "daily"
                elif median_diff.total_seconds() <= 604800:  # <= 1 week
                    return "weekly"
                else:
                    return "monthly_or_longer"

            return None

        except Exception as e:
            self.logger.warning(f"Failed to detect frequency: {e}")
            return None

    def preview_data(self, input_data: InputData, n_rows: int = 5) -> Dict[str, Any]:
        """
        Get a preview of the input data.

        Args:
            input_data: InputData object
            n_rows: Number of rows to include in preview

        Returns:
            Dictionary with preview information
        """
        try:
            data = input_data.data

            if isinstance(data, pd.DataFrame):
                preview = {
                    'head': data.head(n_rows).to_dict('records'),
                    'columns': data.columns.tolist(),
                    'dtypes': data.dtypes.to_dict(),
                    'shape': data.shape,
                    'index_sample': list(data.head(n_rows).index)
                }

                # Add basic statistics for numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    preview['numeric_summary'] = data[numeric_cols].describe().to_dict()

                return preview
            else:
                return {'type': str(type(data)), 'preview': str(data)[:500]}

        except Exception as e:
            self.logger.error(f"Failed to generate data preview: {e}")
            return {'error': str(e)}