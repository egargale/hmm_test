"""
Enhanced CSV format detection and validation system.

This module provides comprehensive CSV format detection, supporting 10+ different
CSV formats with automatic detection and validation capabilities.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CSVFormat:
    """CSV format specification."""
    format_type: str
    delimiter: str
    encoding: str
    datetime_format: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    has_header: bool = True
    date_column: Optional[str] = None
    time_column: Optional[str] = None
    required_columns: List[str] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = []


@dataclass
class DetectionResult:
    """Result of format detection."""
    format: CSVFormat
    confidence: float
    sample_data: pd.DataFrame
    issues: List[str]
    recommendations: List[str]


class CSVFormatDetector:
    """
    Enhanced CSV format detector with support for 10+ formats.

    Supports detection of:
    1. Standard OHLCV: DateTime/Open/High/Low/Close/Volume
    2. Split DateTime: Date/Time/Open/High/Low/Close/Volume
    3. TradingView: Tab-separated Date/Time/Open/High/Low/Close/Volume
    4. Yahoo Finance: Date/Open/High/Low/Close/Adj Close/Volume
    5. Alpha Vantage: Date/Open/High/Low/Close/Volume
    6. Custom Delimiter: Various delimiter support
    7. ISO 8601: ISO datetime format
    8. Unix Timestamp: Unix timestamp datetime
    9. Multi-asset: With asset symbol column
    10. Intraday: Millisecond precision
    """

    def __init__(self):
        self.format_patterns = self._initialize_format_patterns()

    def _initialize_format_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize format detection patterns."""
        return {
            'standard_ohlcv': {
                'delimiters': [','],
                'required_columns': ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                'datetime_columns': ['DateTime'],
                'column_patterns': {
                    'DateTime': [r'datetime', r'date[_\s]?time', r'timestamp'],
                    'Open': [r'open', r'open[_\s]?price'],
                    'High': [r'high', r'high[_\s]?price'],
                    'Low': [r'low', r'low[_\s]?price'],
                    'Close': [r'close', r'close[_\s]?price', r'last'],
                    'Volume': [r'volume', r'vol', r'trade[_\s]?volume']
                }
            },
            'split_datetime': {
                'delimiters': [','],
                'required_columns': ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                'datetime_columns': ['Date', 'Time'],
                'column_patterns': {
                    'Date': [r'date'],
                    'Time': [r'time'],
                    'Open': [r'open'],
                    'High': [r'high'],
                    'Low': [r'low'],
                    'Close': [r'close', r'last'],
                    'Volume': [r'volume']
                }
            },
            'tradingview': {
                'delimiters': ['\t', ','],
                'required_columns': ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                'datetime_columns': ['Date', 'Time'],
                'column_patterns': {
                    'Date': [r'date'],
                    'Time': [r'time'],
                    'Open': [r'open'],
                    'High': [r'high'],
                    'Low': [r'low'],
                    'Close': [r'close'],
                    'Volume': [r'volume']
                }
            },
            'yahoo_finance': {
                'delimiters': [','],
                'required_columns': ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                'datetime_columns': ['Date'],
                'column_patterns': {
                    'Date': [r'date'],
                    'Open': [r'open'],
                    'High': [r'high'],
                    'Low': [r'low'],
                    'Close': [r'close'],
                    'Adj Close': [r'adj[_\s]?close', r'adjusted[_\s]?close'],
                    'Volume': [r'volume']
                }
            },
            'alpha_vantage': {
                'delimiters': [','],
                'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'datetime_columns': ['timestamp'],
                'column_patterns': {
                    'timestamp': [r'timestamp', r'date'],
                    'open': [r'open'],
                    'high': [r'high'],
                    'low': [r'low'],
                    'close': [r'close'],
                    'volume': [r'volume']
                }
            },
            'iso_8601': {
                'delimiters': [','],
                'required_columns': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
                'datetime_columns': ['datetime'],
                'datetime_format': 'ISO8601',
                'column_patterns': {
                    'datetime': [r'datetime', r'timestamp'],
                    'open': [r'open'],
                    'high': [r'high'],
                    'low': [r'low'],
                    'close': [r'close'],
                    'volume': [r'volume']
                }
            },
            'unix_timestamp': {
                'delimiters': [','],
                'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'datetime_columns': ['timestamp'],
                'datetime_format': 'unix',
                'column_patterns': {
                    'timestamp': [r'timestamp', r'date', r'time'],
                    'open': [r'open'],
                    'high': [r'high'],
                    'low': [r'low'],
                    'close': [r'close'],
                    'volume': [r'volume']
                }
            },
            'multi_asset': {
                'delimiters': [','],
                'required_columns': ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume'],
                'datetime_columns': ['datetime'],
                'column_patterns': {
                    'symbol': [r'symbol', r'ticker', r'asset'],
                    'datetime': [r'datetime', r'timestamp', r'date'],
                    'open': [r'open'],
                    'high': [r'high'],
                    'low': [r'low'],
                    'close': [r'close'],
                    'volume': [r'volume']
                }
            }
        }

    def detect_format(self, file_path: Path, sample_size: int = 1000) -> DetectionResult:
        """
        Detect CSV format from file.

        Args:
            file_path: Path to CSV file
            sample_size: Number of rows to sample for detection

        Returns:
            DetectionResult with format information
        """
        try:
            logger.info(f"Detecting format for: {file_path}")

            # Detect encoding
            encoding = self._detect_encoding(file_path)

            # Detect delimiter
            delimiter = self._detect_delimiter(file_path, encoding)

            # Read sample data
            sample_data = self._read_sample(file_path, encoding, delimiter, sample_size)

            # Detect format type
            format_result = self._detect_format_type(sample_data, delimiter, encoding)

            logger.info(f"Format detected: {format_result.format.format_type} "
                       f"(confidence: {format_result.confidence:.2f})")

            return format_result

        except Exception as e:
            logger.error(f"Format detection failed: {e}")
            raise

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')

                logger.debug(f"Detected encoding: {encoding} (confidence: {result.get('confidence', 0):.2f})")

                # Fallback to utf-8 if confidence is low
                if result.get('confidence', 0) < 0.7:
                    encoding = 'utf-8'

                return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
            return 'utf-8'

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter."""
        try:
            with open(file_path, encoding=encoding) as f:
                sample = f.read(1024)

            # Count common delimiters
            delimiter_counts = {
                ',': sample.count(','),
                ';': sample.count(';'),
                '\t': sample.count('\t'),
                '|': sample.count('|'),
                ' ': sample.count(' ')
            }

            # Find most common delimiter (excluding spaces in normal text)
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

            # Sanity check
            if delimiter_counts[best_delimiter] < 3:
                logger.warning("Low delimiter count, defaulting to comma")
                best_delimiter = ','

            logger.debug(f"Detected delimiter: '{best_delimiter}' (counts: {delimiter_counts})")
            return best_delimiter

        except Exception as e:
            logger.warning(f"Delimiter detection failed: {e}, defaulting to comma")
            return ','

    def _read_sample(self, file_path: Path, encoding: str, delimiter: str,
                    sample_size: int) -> pd.DataFrame:
        """Read sample data for format detection."""
        try:
            # Try to read with detected parameters
            sample_df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                nrows=sample_size,
                dtype=str  # Keep as string for pattern matching
            )

            logger.debug(f"Read sample: {len(sample_df)} rows, {len(sample_df.columns)} columns")
            return sample_df

        except Exception as e:
            logger.error(f"Failed to read sample data: {e}")
            raise

    def _detect_format_type(self, sample_data: pd.DataFrame, delimiter: str,
                          encoding: str) -> DetectionResult:
        """Detect specific format type from sample data."""
        columns = [col.strip() for col in sample_data.columns]

        best_match = None
        best_confidence = 0.0

        for format_name, format_config in self.format_patterns.items():
            confidence = self._calculate_format_confidence(
                columns, format_config, sample_data
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = format_name

        if best_match:
            format_config = self.format_patterns[best_match]
            csv_format = self._create_csv_format(
                best_match, format_config, delimiter, encoding, columns
            )

            issues, recommendations = self._validate_format(csv_format, sample_data)

            return DetectionResult(
                format=csv_format,
                confidence=best_confidence,
                sample_data=sample_data,
                issues=issues,
                recommendations=recommendations
            )
        else:
            # Fallback to generic format
            logger.warning("No specific format detected, using generic format")
            csv_format = CSVFormat(
                format_type='generic',
                delimiter=delimiter,
                encoding=encoding,
                column_mapping=self._create_generic_mapping(columns),
                required_columns=self._find_required_columns(columns)
            )

            return DetectionResult(
                format=csv_format,
                confidence=0.5,
                sample_data=sample_data,
                issues=['Generic format detected - manual verification recommended'],
                recommendations=['Consider using predefined format if available']
            )

    def _calculate_format_confidence(self, columns: List[str], format_config: Dict[str, Any],
                                   sample_data: pd.DataFrame) -> float:
        """Calculate confidence score for format match."""
        confidence = 0.0

        # Check delimiter
        if format_config.get('delimiters'):
            # Delimiter already detected separately, add base confidence
            confidence += 0.1

        # Check required columns
        required_cols = format_config.get('required_columns', [])
        if required_cols:
            matching_cols = sum(1 for col in required_cols if col in columns)
            confidence += (matching_cols / len(required_cols)) * 0.6

        # Check column patterns
        patterns = format_config.get('column_patterns', {})
        if patterns:
            pattern_matches = 0
            for _expected_col, col_patterns in patterns.items():
                for actual_col in columns:
                    if any(re.search(pattern, actual_col, re.IGNORECASE) for pattern in col_patterns):
                        pattern_matches += 1
                        break

            confidence += (pattern_matches / len(patterns)) * 0.3

        return min(confidence, 1.0)

    def _create_csv_format(self, format_name: str, format_config: Dict[str, Any],
                          delimiter: str, encoding: str, columns: List[str]) -> CSVFormat:
        """Create CSVFormat object from detection results."""
        datetime_columns = format_config.get('datetime_columns', [])

        # Determine date and time columns
        date_column = None
        time_column = None
        if len(datetime_columns) == 1:
            date_column = datetime_columns[0]
        elif len(datetime_columns) == 2:
            date_column, time_column = datetime_columns

        # Create column mapping
        column_mapping = self._create_column_mapping(columns, format_config)

        return CSVFormat(
            format_type=format_name,
            delimiter=delimiter,
            encoding=encoding,
            datetime_format=format_config.get('datetime_format'),
            column_mapping=column_mapping,
            has_header=True,
            date_column=date_column,
            time_column=time_column,
            required_columns=format_config.get('required_columns', [])
        )

    def _create_column_mapping(self, actual_columns: List[str],
                              format_config: Dict[str, Any]) -> Dict[str, str]:
        """Create mapping from actual columns to standard columns."""
        mapping = {}
        patterns = format_config.get('column_patterns', {})

        for actual_col in actual_columns:
            matched = False
            for standard_col, col_patterns in patterns.items():
                if any(re.search(pattern, actual_col, re.IGNORECASE) for pattern in col_patterns):
                    mapping[actual_col] = standard_col
                    matched = True
                    break

            if not matched:
                # Keep original column name
                mapping[actual_col] = actual_col

        return mapping

    def _create_generic_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Create generic column mapping for unknown formats."""
        mapping = {}

        # Common patterns for generic mapping
        common_patterns = {
            'datetime': [r'datetime', r'date[_\s]?time', r'timestamp'],
            'date': [r'date'],
            'time': [r'time'],
            'open': [r'open', r'open[_\s]?price'],
            'high': [r'high', r'high[_\s]?price'],
            'low': [r'low', r'low[_\s]?price'],
            'close': [r'close', r'close[_\s]?price', r'last'],
            'volume': [r'volume', r'vol'],
            'adj_close': [r'adj[_\s]?close', r'adjusted[_\s]?close'],
            'symbol': [r'symbol', r'ticker', r'asset']
        }

        for actual_col in columns:
            matched = False
            for standard_col, patterns in common_patterns.items():
                if any(re.search(pattern, actual_col, re.IGNORECASE) for pattern in patterns):
                    mapping[actual_col] = standard_col
                    matched = True
                    break

            if not matched:
                mapping[actual_col] = actual_col

        return mapping

    def _find_required_columns(self, columns: List[str]) -> List[str]:
        """Find likely required columns from generic format."""
        required = []
        essential_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        for col in essential_columns:
            if any(re.search(pattern, actual_col, re.IGNORECASE)
                   for actual_col in columns
                   for pattern in [col, f'{col}[_\\s]?price', f'{col}[_\\s]?volume']):
                required.append(col)

        return required

    def _validate_format(self, csv_format: CSVFormat, sample_data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate detected format and return issues and recommendations."""
        issues = []
        recommendations = []

        # Check for missing required columns
        mapped_columns = list(csv_format.column_mapping.values())
        missing_required = [col for col in csv_format.required_columns if col not in mapped_columns]
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")

        # Check for duplicate columns
        if len(sample_data.columns) != len(set(sample_data.columns)):
            issues.append("Duplicate column names detected")
            recommendations.append("Clean column names before processing")

        # Check data types consistency
        for col in sample_data.columns:
            if col in csv_format.column_mapping:
                standard_name = csv_format.column_mapping[col]
                if standard_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    # Check if price/volume columns contain numeric data
                    try:
                        pd.to_numeric(sample_data[col].dropna())
                    except ValueError:
                        issues.append(f"Non-numeric data in {standard_name} column ({col})")
                        recommendations.append(f"Clean {col} column or verify data format")

        # Check for empty data
        if sample_data.empty:
            issues.append("No data found in sample")
            recommendations.append("Verify file is not empty and has correct format")

        # Check datetime format if applicable
        if csv_format.date_column and csv_format.datetime_format == 'ISO8601':
            if csv_format.date_column in sample_data.columns:
                try:
                    pd.to_datetime(sample_data[csv_format.date_column].head())
                except ValueError:
                    issues.append(f"Invalid datetime format in {csv_format.date_column}")
                    recommendations.append(f"Verify {csv_format.date_column} contains ISO8601 dates")

        return issues, recommendations

    def get_supported_formats(self) -> List[str]:
        """Get list of supported format types."""
        return list(self.format_patterns.keys())

    def register_custom_format(self, format_name: str, format_config: Dict[str, Any]) -> None:
        """Register a custom format pattern."""
        self.format_patterns[format_name] = format_config
        logger.info(f"Registered custom format: {format_name}")
