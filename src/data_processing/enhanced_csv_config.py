"""
Enhanced CSV processing configuration system.

This module provides comprehensive configuration management for CSV processing,
including format profiles, validation rules, performance tuning, and error handling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """CSV processing modes."""

    STREAMING = "streaming"
    MEMORY = "memory"
    HYBRID = "hybrid"


class ErrorRecoveryMode(Enum):
    """Error recovery strategies."""

    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"
    REPAIR = "repair"


class OutlierDetectionMethod(Enum):
    """Outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class FormatProfile:
    """Predefined configuration for common data sources."""

    name: str
    delimiter: str
    encoding: str = "utf-8"
    datetime_format: Optional[str] = None
    column_mapping: Dict[str, str] = field(default_factory=dict)
    skip_rows: int = 0
    header_row: int = 0
    required_columns: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Custom validation rule definition."""

    name: str
    column: Optional[str] = None
    rule_type: str = "range"  # range, pattern, custom
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_level: str = "warning"  # info, warning, error, critical
    description: str = ""


@dataclass
class EnhancedCSVConfig:
    """
    Enhanced configuration for CSV processing with comprehensive options.

    This configuration supports all the enhanced features from Phase 2.1.3:
    - Advanced format detection and handling
    - Performance optimization settings
    - Validation and error handling
    - Feature engineering integration
    """

    # Format Detection
    auto_detect_format: bool = True
    supported_formats: List[str] = field(
        default_factory=lambda: [
            "standard_ohlcv",
            "split_datetime",
            "tradingview",
            "yahoo_finance",
            "alpha_vantage",
            "iso_8601",
            "unix_timestamp",
            "multi_asset",
        ]
    )
    fallback_format: str = "standard_ohlcv"
    confidence_threshold: float = 0.7

    # Performance Settings
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10000
    adaptive_chunking: bool = True
    memory_limit_mb: int = 1024
    enable_memory_mapping: bool = True
    downcast_dtypes: bool = True
    cache_results: bool = False
    cache_size_mb: int = 100

    # Validation Settings
    enable_validation: bool = True
    strict_mode: bool = False
    validate_ohlc_consistency: bool = True
    detect_outliers: bool = True
    outlier_detection_method: OutlierDetectionMethod = OutlierDetectionMethod.IQR
    outlier_threshold: float = 1.5
    validate_ranges: bool = True
    check_duplicates: bool = True
    analyze_missing_data: bool = True

    # Error Handling
    error_recovery_mode: ErrorRecoveryMode = ErrorRecoveryMode.SKIP
    max_retries: int = 3
    retry_delay: float = 1.0
    skip_error_rows: bool = True
    log_errors: bool = True
    create_error_report: bool = True

    # Data Processing
    normalize_timezones: bool = True
    target_timezone: str = "UTC"
    standardize_column_names: bool = True
    fill_missing_values: bool = False
    missing_value_strategy: str = (
        "interpolate"  # interpolate, forward_fill, backward_fill, mean, drop
    )
    remove_duplicates: bool = True
    sort_by_datetime: bool = True

    # Feature Engineering Integration (Phase 2.1.2)
    enable_feature_engineering: bool = True
    feature_config: Dict[str, Any] = field(default_factory=dict)
    apply_feature_selection: bool = False
    feature_quality_threshold: float = 0.5

    # Output Settings
    output_format: str = "dataframe"  # dataframe, csv, parquet, json
    include_metadata: bool = True
    include_quality_report: bool = True
    save_processing_log: bool = False

    # Custom Configuration
    custom_validation_rules: List[ValidationRule] = field(default_factory=list)
    format_profiles: Dict[str, FormatProfile] = field(default_factory=dict)
    custom_column_mappings: Dict[str, str] = field(default_factory=dict)

    # Logging and Monitoring
    log_level: str = "INFO"
    show_progress: bool = True
    performance_monitoring: bool = True
    memory_monitoring: bool = True

    def __post_init__(self):
        """Initialize default configurations."""
        self._setup_default_format_profiles()
        self._setup_default_validation_rules()

    def _setup_default_format_profiles(self) -> None:
        """Setup default format profiles for common data sources."""
        default_profiles = {
            "yahoo_finance": FormatProfile(
                name="yahoo_finance",
                delimiter=",",
                encoding="utf-8",
                column_mapping={
                    "Date": "datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                },
                required_columns=["Date", "Open", "High", "Low", "Close", "Volume"],
            ),
            "tradingview": FormatProfile(
                name="tradingview",
                delimiter="\t",
                encoding="utf-8",
                column_mapping={
                    "Date": "datetime",
                    "Time": "time",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                required_columns=[
                    "Date",
                    "Time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ],
            ),
            "alpha_vantage": FormatProfile(
                name="alpha_vantage",
                delimiter=",",
                encoding="utf-8",
                column_mapping={
                    "timestamp": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
                required_columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            ),
            "crypto_binance": FormatProfile(
                name="crypto_binance",
                delimiter=",",
                encoding="utf-8",
                column_mapping={
                    "Open time": "datetime",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Close time": "close_time",
                    "Quote asset volume": "quote_volume",
                },
                skip_rows=1,  # Skip header row
                required_columns=[
                    "Open time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ],
            ),
        }

        # Add default profiles if not already present
        for name, profile in default_profiles.items():
            if name not in self.format_profiles:
                self.format_profiles[name] = profile

    def _setup_default_validation_rules(self) -> None:
        """Setup default validation rules."""
        default_rules = [
            ValidationRule(
                name="price_non_negative",
                column=["open", "high", "low", "close"],
                rule_type="range",
                parameters={"min_value": 0},
                error_level="error",
                description="Prices should not be negative",
            ),
            ValidationRule(
                name="volume_non_negative",
                column="volume",
                rule_type="range",
                parameters={"min_value": 0},
                error_level="error",
                description="Volume should not be negative",
            ),
            ValidationRule(
                name="ohlc_relationships",
                rule_type="custom",
                parameters={"check_high": True, "check_low": True},
                error_level="error",
                description="High should be >= Open/Close, Low should be <= Open/Close",
            ),
            ValidationRule(
                name="datetime_continuity",
                rule_type="custom",
                parameters={"max_gap_minutes": 60},
                error_level="warning",
                description="Check for large gaps in datetime sequence",
            ),
        ]

        # Add default rules if not already present
        for rule in default_rules:
            if not any(r.name == rule.name for r in self.custom_validation_rules):
                self.custom_validation_rules.append(rule)

    def get_format_profile(self, name: str) -> Optional[FormatProfile]:
        """Get format profile by name."""
        return self.format_profiles.get(name)

    def add_format_profile(self, profile: FormatProfile) -> None:
        """Add a new format profile."""
        self.format_profiles[profile.name] = profile
        logger.info(f"Added format profile: {profile.name}")

    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a new validation rule."""
        self.custom_validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")

    def remove_validation_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        for i, rule in enumerate(self.custom_validation_rules):
            if rule.name == rule_name:
                del self.custom_validation_rules[i]
                logger.info(f"Removed validation rule: {rule_name}")
                return True
        return False

    def get_validation_rules_for_column(self, column: str) -> List[ValidationRule]:
        """Get validation rules applicable to a specific column."""
        applicable_rules = []

        for rule in self.custom_validation_rules:
            if rule.column is None:
                # Global rule
                applicable_rules.append(rule)
            elif isinstance(rule.column, str) and rule.column == column:
                applicable_rules.append(rule)
            elif isinstance(rule.column, list) and column in rule.column:
                applicable_rules.append(rule)

        return applicable_rules

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "auto_detect_format": self.auto_detect_format,
            "supported_formats": self.supported_formats,
            "processing_mode": self.processing_mode.value,
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_workers": self.max_workers,
            "chunk_size": self.chunk_size,
            "adaptive_chunking": self.adaptive_chunking,
            "memory_limit_mb": self.memory_limit_mb,
            "enable_validation": self.enable_validation,
            "strict_mode": self.strict_mode,
            "error_recovery_mode": self.error_recovery_mode.value,
            "enable_feature_engineering": self.enable_feature_engineering,
            "feature_config": self.feature_config,
            "validation_rules": [
                rule.__dict__ for rule in self.custom_validation_rules
            ],
            "format_profiles": {
                name: profile.__dict__ for name, profile in self.format_profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnhancedCSVConfig":
        """Create configuration from dictionary."""
        # Handle enum conversions
        if "processing_mode" in config_dict:
            config_dict["processing_mode"] = ProcessingMode(
                config_dict["processing_mode"]
            )

        if "error_recovery_mode" in config_dict:
            config_dict["error_recovery_mode"] = ErrorRecoveryMode(
                config_dict["error_recovery_mode"]
            )

        if "outlier_detection_method" in config_dict:
            config_dict["outlier_detection_method"] = OutlierDetectionMethod(
                config_dict["outlier_detection_method"]
            )

        # Handle validation rules
        if "validation_rules" in config_dict:
            rules = []
            for rule_dict in config_dict["validation_rules"]:
                rules.append(ValidationRule(**rule_dict))
            config_dict["custom_validation_rules"] = rules
            del config_dict["validation_rules"]

        # Handle format profiles
        if "format_profiles" in config_dict:
            profiles = {}
            for name, profile_dict in config_dict["format_profiles"].items():
                profiles[name] = FormatProfile(**profile_dict)
            config_dict["format_profiles"] = profiles

        return cls(**config_dict)

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate performance settings
        if self.chunk_size <= 0:
            issues.append("chunk_size must be positive")

        if self.memory_limit_mb <= 0:
            issues.append("memory_limit_mb must be positive")

        if self.max_workers is not None and self.max_workers <= 0:
            issues.append("max_workers must be positive")

        # Validate validation settings
        if self.outlier_threshold <= 0:
            issues.append("outlier_threshold must be positive")

        if 0 <= self.confidence_threshold > 1:
            issues.append("confidence_threshold must be between 0 and 1")

        # Validate feature engineering settings
        if self.feature_quality_threshold < 0 or self.feature_quality_threshold > 1:
            issues.append("feature_quality_threshold must be between 0 and 1")

        # Validate custom validation rules
        for rule in self.custom_validation_rules:
            if not rule.name:
                issues.append("Validation rule must have a name")
            if rule.error_level not in ["info", "warning", "error", "critical"]:
                issues.append(
                    f"Invalid error level in rule {rule.name}: {rule.error_level}"
                )

        return issues

    def get_performance_profile(self, file_size_mb: float) -> Dict[str, Any]:
        """Get performance recommendations based on file size."""
        profile = {
            "recommended_chunk_size": self.chunk_size,
            "recommended_workers": self.max_workers,
            "use_memory_mapping": self.enable_memory_mapping,
            "use_parallel_processing": self.enable_parallel_processing,
        }

        # Adjust recommendations based on file size
        if file_size_mb > 100:  # Large file
            profile["recommended_chunk_size"] = 50000
            profile["use_memory_mapping"] = True
            profile["use_parallel_processing"] = True
            if self.max_workers is None:
                profile["recommended_workers"] = min(4, self._get_cpu_count())

        elif file_size_mb > 10:  # Medium file
            profile["recommended_chunk_size"] = 25000
            profile["use_memory_mapping"] = True
            profile["use_parallel_processing"] = True

        else:  # Small file
            profile["recommended_chunk_size"] = min(len(self.chunk_size), 10000)
            profile["use_memory_mapping"] = False
            profile["use_parallel_processing"] = False

        return profile

    def _get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        try:
            import multiprocessing

            return multiprocessing.cpu_count()
        except Exception:
            return 2  # Conservative default

    def create_processing_summary(self) -> Dict[str, Any]:
        """Create a summary of current configuration."""
        return {
            "format_detection": {
                "auto_detect": self.auto_detect_format,
                "supported_formats": len(self.supported_formats),
                "confidence_threshold": self.confidence_threshold,
            },
            "performance": {
                "processing_mode": self.processing_mode.value,
                "parallel_processing": self.enable_parallel_processing,
                "chunk_size": self.chunk_size,
                "memory_limit_mb": self.memory_limit_mb,
                "memory_mapping": self.enable_memory_mapping,
            },
            "validation": {
                "enabled": self.enable_validation,
                "strict_mode": self.strict_mode,
                "validation_rules": len(self.custom_validation_rules),
                "outlier_detection": self.detect_outliers,
                "outlier_method": self.outlier_detection_method.value,
            },
            "features": {
                "feature_engineering": self.enable_feature_engineering,
                "feature_selection": self.apply_feature_selection,
                "feature_categories": list(self.feature_config.keys()),
            },
            "error_handling": {
                "recovery_mode": self.error_recovery_mode.value,
                "max_retries": self.max_retries,
                "skip_error_rows": self.skip_error_rows,
            },
        }


def create_default_config() -> EnhancedCSVConfig:
    """Create default enhanced CSV configuration."""
    return EnhancedCSVConfig()


def create_high_performance_config() -> EnhancedCSVConfig:
    """Create configuration optimized for performance."""
    config = EnhancedCSVConfig()
    config.processing_mode = ProcessingMode.STREAMING
    config.enable_parallel_processing = True
    config.max_workers = None  # Auto-detect
    config.chunk_size = 50000
    config.adaptive_chunking = True
    config.enable_memory_mapping = True
    config.downcast_dtypes = True
    config.cache_results = True
    config.cache_size_mb = 200

    # Reduce validation for speed
    config.enable_validation = True
    config.strict_mode = False
    config.detect_outliers = False  # Skip for performance

    return config


def create_high_quality_config() -> EnhancedCSVConfig:
    """Create configuration optimized for data quality."""
    config = EnhancedCSVConfig()
    config.enable_validation = True
    config.strict_mode = True
    config.validate_ohlc_consistency = True
    config.detect_outliers = True
    config.outlier_detection_method = OutlierDetectionMethod.IQR
    config.outlier_threshold = 1.5
    config.validate_ranges = True
    config.check_duplicates = True
    config.analyze_missing_data = True

    # Conservative error handling
    config.error_recovery_mode = ErrorRecoveryMode.RETRY
    config.max_retries = 5
    config.skip_error_rows = False

    # Enhanced feature engineering
    config.enable_feature_engineering = True
    config.apply_feature_selection = True
    config.feature_quality_threshold = 0.7

    return config


def create_streaming_config() -> EnhancedCSVConfig:
    """Create configuration optimized for streaming large files."""
    config = EnhancedCSVConfig()
    config.processing_mode = ProcessingMode.STREAMING
    config.chunk_size = 10000
    config.adaptive_chunking = True
    config.enable_memory_mapping = True
    config.cache_results = False  # Don't cache in streaming mode

    # Minimal validation for speed
    config.enable_validation = True
    config.strict_mode = False
    config.detect_outliers = False

    # Skip feature engineering in streaming (apply later if needed)
    config.enable_feature_engineering = False

    return config
