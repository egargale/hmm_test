"""
Unified pipeline configuration system.

This module provides comprehensive configuration management for the unified data pipeline,
integrating all configuration aspects from Phases 2.1.1-2.1.3 into a single, cohesive system.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from ..utils.logging_config import get_logger
from .enhanced_csv_config import EnhancedCSVConfig, create_high_performance_config
from .performance_optimizer import PerformanceConfig
from .data_validator import ValidationLevel

logger = get_logger(__name__)


class InputSourceType(Enum):
    """Supported input source types."""
    CSV_FILE = "csv_file"
    DATAFRAME = "dataframe"
    DATABASE = "database"
    API = "api"


class OutputFormat(Enum):
    """Supported output formats."""
    DATAFRAME = "dataframe"
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    PICKLE = "pickle"


class PipelineMode(Enum):
    """Pipeline execution modes."""
    STANDARD = "standard"          # Balanced performance and quality
    HIGH_PERFORMANCE = "high_performance"  # Optimized for speed
    HIGH_QUALITY = "high_quality"  # Optimized for data quality
    STREAMING = "streaming"        # For large datasets
    DEVELOPMENT = "development"    # For testing and debugging


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    enable_validation: bool = True
    strict_mode: bool = False
    validate_ohlc_consistency: bool = True
    detect_outliers: bool = True
    outlier_detection_method: str = 'iqr'
    outlier_threshold: float = 1.5
    validate_ranges: bool = True
    check_duplicates: bool = True
    analyze_missing_data: bool = True
    missing_data_strategy: str = "interpolate"  # interpolate, forward_fill, backward_fill, mean, drop
    quality_threshold: float = 0.5


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    enable_features: bool = True
    basic_indicators: bool = True
    enhanced_momentum: bool = True
    enhanced_volatility: bool = True
    enhanced_trend: bool = True
    enhanced_volume: bool = True
    time_features: bool = True
    feature_selection: bool = False
    feature_quality_threshold: float = 0.5

    # Specific indicator configurations
    indicator_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Set default indicator parameters."""
        if not self.indicator_params:
            self.indicator_params = {
                'basic_indicators': {
                    'enabled': True
                },
                'enhanced_momentum': {
                    'williams_r': {'length': 14},
                    'cci': {'length': 20},
                    'mfi': {'length': 14},
                    'mtm': {'period': 10},
                    'proc': {'period': 14}
                },
                'enhanced_volatility': {
                    'historical_volatility': {'window': 20},
                    'keltner_channels': {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0},
                    'donchian_channels': {'period': 20}
                },
                'enhanced_trend': {
                    'tma': {'period': 20},
                    'wma': {'period': 10},
                    'hma': {'period': 16},
                    'aroon': {'period': 14},
                    'dmi': {'period': 14}
                },
                'enhanced_volume': {
                    'adl': {'enabled': True},
                    'vpt': {'enabled': True},
                    'eom': {'period': 14},
                    'volume_roc': {'period': 10}
                },
                'time_features': {
                    'calendar_features': True,
                    'cyclical_features': True,
                    'intraday_features': True,
                    'weekend_effects': True
                }
            }


@dataclass
class OutputConfig:
    """Configuration for pipeline output."""
    format: OutputFormat = OutputFormat.DATAFRAME
    save_path: Optional[Path] = None
    include_metadata: bool = True
    include_quality_report: bool = True
    include_processing_log: bool = False
    overwrite_existing: bool = False
    compression: Optional[str] = None  # For supported formats

    # Output customization
    column_prefix: Optional[str] = None
    index_column: Optional[str] = None
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class PipelineConfig:
    """
    Unified configuration for the entire data pipeline.

    This configuration integrates all aspects of the data processing pipeline,
    including input handling, CSV processing, validation, feature engineering,
    performance optimization, and output formatting.
    """

    # Basic configuration
    name: str = "default_pipeline"
    mode: PipelineMode = PipelineMode.STANDARD
    description: Optional[str] = None

    # Input configuration
    input_source: Optional[Union[str, Path]] = None
    input_type: InputSourceType = InputSourceType.CSV_FILE
    encoding: str = "utf-8"

    # Stage configurations
    csv_config: EnhancedCSVConfig = field(default_factory=EnhancedCSVConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)

    # Pipeline execution settings
    enable_caching: bool = False
    cache_size_mb: int = 100
    parallel_stages: bool = False
    max_parallel_stages: int = 2

    # Logging and monitoring
    log_level: str = "INFO"
    enable_progress_tracking: bool = True
    enable_performance_monitoring: bool = True
    save_metrics: bool = False
    metrics_output_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization configuration adjustments."""
        # Adjust configurations based on pipeline mode
        self._apply_mode_specific_settings()

        # Validate configuration
        self._validate_configuration()

    def _apply_mode_specific_settings(self) -> None:
        """Apply mode-specific configuration adjustments."""
        if self.mode == PipelineMode.HIGH_PERFORMANCE:
            self.csv_config = create_high_performance_config()
            self.validation_config.strict_mode = False
            self.validation_config.detect_outliers = False
            self.feature_config.feature_selection = False

        elif self.mode == PipelineMode.HIGH_QUALITY:
            self.validation_config.strict_mode = True
            self.validation_config.outlier_threshold = 1.5
            self.feature_config.feature_selection = True
            self.feature_config.feature_quality_threshold = 0.7

        elif self.mode == PipelineMode.STREAMING:
            self.csv_config.processing_mode.value = "streaming"
            self.csv_config.cache_results = False
            self.validation_config.detect_outliers = False
            self.feature_config.enable_features = False

        elif self.mode == PipelineMode.DEVELOPMENT:
            self.enable_progress_tracking = True
            self.enable_performance_monitoring = True
            self.save_metrics = True

    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        issues = []

        # Validate input source
        if self.input_source is None and self.input_type == InputSourceType.CSV_FILE:
            issues.append("Input source must be specified for CSV file input")

        # Validate output configuration
        if self.output_config.save_path is None and self.output_config.format != OutputFormat.DATAFRAME:
            issues.append("Save path must be specified for file output formats")

        # Validate performance settings
        if self.performance_config.chunk_size <= 0:
            issues.append("Chunk size must be positive")

        # Validate feature configuration
        if self.feature_config.feature_quality_threshold < 0 or self.feature_config.feature_quality_threshold > 1:
            issues.append("Feature quality threshold must be between 0 and 1")

        if issues:
            logger.warning(f"Configuration validation issues: {issues}")

    @classmethod
    def from_mode(cls, mode: PipelineMode, **kwargs) -> 'PipelineConfig':
        """Create configuration from pipeline mode."""
        config = cls(mode=mode, **kwargs)
        return config

    @classmethod
    def create_high_performance(cls, **kwargs) -> 'PipelineConfig':
        """Create high-performance configuration."""
        return cls(mode=PipelineMode.HIGH_PERFORMANCE, **kwargs)

    @classmethod
    def create_high_quality(cls, **kwargs) -> 'PipelineConfig':
        """Create high-quality configuration."""
        return cls(mode=PipelineMode.HIGH_QUALITY, **kwargs)

    @classmethod
    def create_streaming(cls, **kwargs) -> 'PipelineConfig':
        """Create streaming configuration."""
        return cls(mode=PipelineMode.STREAMING, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'mode': self.mode.value,
            'description': self.description,
            'input_source': str(self.input_source) if self.input_source else None,
            'input_type': self.input_type.value,
            'encoding': self.encoding,
            'csv_config': self.csv_config.to_dict(),
            'validation_config': {
                'enable_validation': self.validation_config.enable_validation,
                'strict_mode': self.validation_config.strict_mode,
                'detect_outliers': self.validation_config.detect_outliers,
                'quality_threshold': self.validation_config.quality_threshold
            },
            'feature_config': {
                'enable_features': self.feature_config.enable_features,
                'feature_selection': self.feature_config.feature_selection,
                'feature_quality_threshold': self.feature_config.feature_quality_threshold,
                'indicator_params': self.feature_config.indicator_params
            },
            'performance_config': {
                'enable_parallel_processing': self.performance_config.enable_parallel_processing,
                'chunk_size': self.performance_config.chunk_size,
                'memory_limit_mb': self.performance_config.memory_limit_mb
            },
            'output_config': {
                'format': self.output_config.format.value,
                'save_path': str(self.output_config.save_path) if self.output_config.save_path else None,
                'include_metadata': self.output_config.include_metadata,
                'include_quality_report': self.output_config.include_quality_report
            },
            'enable_caching': self.enable_caching,
            'parallel_stages': self.parallel_stages,
            'log_level': self.log_level
        }

    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated pipeline config: {key} = {value}")
            else:
                logger.warning(f"Unknown pipeline configuration key: {key}")

        # Re-apply mode-specific settings and validation
        self._apply_mode_specific_settings()
        self._validate_configuration()

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'pipeline_name': self.name,
            'mode': self.mode.value,
            'input_type': self.input_type.value,
            'output_format': self.output_config.format.value,
            'validation_enabled': self.validation_config.enable_validation,
            'features_enabled': self.feature_config.enable_features,
            'parallel_processing': self.performance_config.enable_parallel_processing,
            'estimated_memory_mb': self.performance_config.memory_limit_mb,
            'chunk_size': self.performance_config.chunk_size
        }

    def get_feature_categories(self) -> List[str]:
        """Get enabled feature categories."""
        categories = []

        if self.feature_config.basic_indicators:
            categories.append('basic_indicators')
        if self.feature_config.enhanced_momentum:
            categories.append('enhanced_momentum')
        if self.feature_config.enhanced_volatility:
            categories.append('enhanced_volatility')
        if self.feature_config.enhanced_trend:
            categories.append('enhanced_trend')
        if self.feature_config.enhanced_volume:
            categories.append('enhanced_volume')
        if self.feature_config.time_features:
            categories.append('time_features')

        return categories

    def validate_input_compatibility(self, data_source: Any) -> List[str]:
        """Validate compatibility with input data source."""
        issues = []

        if self.input_type == InputSourceType.CSV_FILE:
            if not isinstance(data_source, (str, Path)):
                issues.append("CSV file input requires string or Path object")
            elif not Path(data_source).exists():
                issues.append(f"CSV file not found: {data_source}")

        elif self.input_type == InputSourceType.DATAFRAME:
            import pandas as pd
            if not isinstance(data_source, pd.DataFrame):
                issues.append("DataFrame input requires pandas DataFrame object")

        return issues

    def estimate_processing_resources(self, data_size_rows: int) -> Dict[str, Any]:
        """Estimate processing resources required for given data size."""
        # Memory estimation (rough calculation)
        bytes_per_row = 100  # Conservative estimate
        estimated_data_mb = (data_size_rows * bytes_per_row) / (1024 * 1024)

        # Processing time estimation based on performance benchmarks
        base_rows_per_second = 500000  # From Phase 2.1.3 benchmarks

        if self.mode == PipelineMode.HIGH_PERFORMANCE:
            rows_per_second = base_rows_per_second * 1.5
        elif self.mode == PipelineMode.HIGH_QUALITY:
            rows_per_second = base_rows_per_second * 0.5
        else:
            rows_per_second = base_rows_per_second

        estimated_time_seconds = data_size_rows / rows_per_second

        # Feature engineering overhead
        if self.feature_config.enable_features:
            estimated_time_seconds *= 1.5

        # Validation overhead
        if self.validation_config.enable_validation:
            estimated_time_seconds *= 1.2

        return {
            'estimated_memory_mb': estimated_data_mb * 2,  # 2x for processing overhead
            'estimated_time_seconds': estimated_time_seconds,
            'estimated_time_minutes': estimated_time_seconds / 60,
            'recommended_chunk_size': min(self.performance_config.chunk_size, data_size_rows // 10),
            'parallel_processing_recommended': data_size_rows > 50000
        }

    def create_pipeline_metadata(self) -> Dict[str, Any]:
        """Create pipeline metadata for tracking."""
        return {
            'pipeline_name': self.name,
            'mode': self.mode.value,
            'created_at': pd.Timestamp.now().isoformat(),
            'configuration_hash': hash(str(self.to_dict())),
            'feature_categories': self.get_feature_categories(),
            'validation_enabled': self.validation_config.enable_validation,
            'performance_optimization': self.performance_config.enable_parallel_processing,
            'output_format': self.output_config.format.value
        }


def create_config_from_template(template_name: str, **kwargs) -> PipelineConfig:
    """Create configuration from predefined template."""
    templates = {
        'default': PipelineConfig,
        'high_performance': PipelineConfig.create_high_performance,
        'high_quality': PipelineConfig.create_high_quality,
        'streaming': PipelineConfig.create_streaming
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

    config_creator = templates[template_name]
    return config_creator(**kwargs)


def load_config_from_file(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from file."""
    import json

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Convert string enums back to enum objects
        if 'mode' in config_dict:
            config_dict['mode'] = PipelineMode(config_dict['mode'])

        if 'input_type' in config_dict:
            config_dict['input_type'] = InputSourceType(config_dict['input_type'])

        # Reconstruct nested configurations
        # This is a simplified version - in practice, you'd want more robust reconstruction
        return PipelineConfig(**config_dict)

    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config_to_file(config: PipelineConfig, config_path: Path) -> bool:
    """Save pipeline configuration to file."""
    import json

    try:
        config_dict = config.to_dict()

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False