# Configuration Consolidation Plan

## Executive Summary

**Purpose**: Design unified configuration management system for enhanced src directory architecture
**Scope**: Consolidate configuration from main directory CLI arguments, src directory configs, and new requirements
**Target Modules**: 12 modules requiring configuration management
**Configuration Sources**: CLI arguments, config files (YAML/JSON), environment variables, defaults
**Complexity**: Medium (hierarchical configuration with validation and type safety)
**Priority**: High (Foundation for all subsequent migration phases)

This document outlines a comprehensive configuration management system that will consolidate all configuration sources across the HMM analysis system, providing centralized, type-safe, and extensible configuration management for the enhanced src directory architecture.

---

## Current Configuration State Analysis

### Main Directory Configuration Sources

**CLI Arguments Distribution**:
- **main.py**: 9 CLI arguments (n-states, max-iter, chunksize, etc.)
- **cli.py**: 14 CLI arguments (engine, n-restarts, generate-charts, etc.)
- **cli_simple.py**: 6 CLI arguments (n-states, test-size, random-seed)
- **cli_comprehensive.py**: 13 CLI arguments (engine, chunk-size, memory-monitor, etc.)

**Configuration Patterns Identified**:
1. **Direct CLI Arguments**: Command-line parameter parsing
2. **Hardcoded Defaults**: Fixed values in code
3. **Environment Variables**: Limited usage in comprehensive CLI
4. **No Configuration Files**: No external configuration support
5. **No Validation**: Limited input validation
6. **No Centralization**: Each script manages its own configuration

### Current Src Directory Configuration

**Existing Configuration Modules**:
```python
# Current configuration structure
src/utils/config.py           # Basic configuration utilities
src/utils/data_types.py       # Data type definitions
src/utils/logging_config.py   # Logging configuration
```

**Limitations of Current System**:
- **Fragmented**: Configuration scattered across multiple modules
- **Inconsistent**: Different patterns across modules
- **Limited**: No support for complex configuration scenarios
- **Static**: No runtime configuration updates
- **Uncentralized**: No single source of truth for configuration

---

## Enhanced Configuration Architecture

### Configuration Hierarchy

**Priority Order (highest to lowest)**:
1. **CLI Arguments**: Command-line parameters (highest priority)
2. **Environment Variables**: System environment variables
3. **Configuration Files**: YAML/JSON configuration files
4. **Default Values**: Built-in default configurations (lowest priority)

**Configuration Sources Mapping**:
```python
# Configuration source hierarchy
ConfigurationManager
├── CLI Arguments (argparse, click)
├── Environment Variables (os.environ)
├── Configuration Files (YAML, JSON)
└── Default Values (hardcoded defaults)
```

### Configuration Schema Structure

**Top-Level Configuration Categories**:
```python
# Configuration schema structure
{
    # Analysis Configuration
    "analysis": {
        "n_states": int,
        "covariance_type": str,
        "random_state": int,
        "test_size": float,
        "lookahead_days": int
    },

    # Processing Configuration
    "processing": {
        "engine": str,  # "streaming", "dask", "daft"
        "chunk_size": int,
        "memory_limit": str,
        "parallel_workers": int
    },

    # Feature Engineering Configuration
    "features": {
        "indicators": List[IndicatorConfig],
        "validation": FeatureValidationConfig,
        "preprocessing": PreprocessingConfig
    },

    # Model Configuration
    "models": {
        "hmm": HMMConfig,
        "lstm": LSTMConfig,
        "hybrid": HybridConfig
    },

    # Backtesting Configuration
    "backtesting": {
        "initial_capital": float,
        "commission": float,
        "slippage": float,
        "strategies": List[StrategyConfig]
    },

    # Visualization Configuration
    "visualization": {
        "chart_style": str,
        "color_scheme": str,
        "interactive": bool,
        "save_plots": bool
    },

    # Logging Configuration
    "logging": {
        "level": str,
        "format": str,
        "file_path": str,
        "rotation": LogRotationConfig
    },

    # Performance Configuration
    "performance": {
        "monitoring": bool,
        "memory_threshold": float,
        "profiling": bool,
        "benchmarks": bool
    }
}
```

---

## Configuration Management System Design

### 1. Core Configuration Manager

**Central Configuration Manager**:
```python
# src/configuration/manager.py
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import yaml
import json
from dataclasses import dataclass, field

@dataclass
class ConfigurationSources:
    """Configuration sources definition."""
    cli_args: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    config_files: List[Path] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)

class ConfigurationManager:
    """Centralized configuration manager."""

    def __init__(self):
        self.sources = ConfigurationSources()
        self._config: Dict[str, Any] = {}
        self._load_default_config()

    def load_configuration(self,
                          cli_args: Optional[Dict[str, Any]] = None,
                          config_file_paths: Optional[List[Path]] = None,
                          environment_prefix: str = "HMM_") -> Dict[str, Any]:
        """Load configuration from all sources."""

        # 1. Load defaults (lowest priority)
        self._config = self.sources.defaults.copy()

        # 2. Load configuration files
        if config_file_paths:
            self.sources.config_files = config_file_paths
            for config_file in config_file_paths:
                if config_file.exists():
                    file_config = self._load_config_file(config_file)
                    self._merge_config(self._config, file_config)

        # 3. Load environment variables
        env_config = self._load_environment_variables(environment_prefix)
        self.sources.environment = env_config
        self._merge_config(self._config, env_config)

        # 4. Load CLI arguments (highest priority)
        if cli_args:
            self.sources.cli_args = cli_args
            self._merge_config(self._config, cli_args)

        # 5. Validate configuration
        self._validate_configuration()

        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._get_nested_value(self._config, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._set_nested_value(self._config, key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._merge_config(self._config, updates)
```

### 2. Configuration Schema and Validation

**Pydantic-Based Configuration Models**:
```python
# src/configuration/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum

class CovarianceType(str, Enum):
    FULL = "full"
    DIAG = "diag"
    SPHERICAL = "spherical"
    TIED = "tied"

class ProcessingEngine(str, Enum):
    STREAMING = "streaming"
    DASK = "dask"
    DAFT = "daft"
    BATCH = "batch"
    REALTIME = "realtime"

class AnalysisConfig(BaseModel):
    """Analysis configuration schema."""
    n_states: int = Field(default=3, ge=2, le=10, description="Number of HMM states")
    covariance_type: CovarianceType = Field(default=CovarianceType.FULL, description="HMM covariance type")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test data proportion")
    lookahead_days: int = Field(default=1, ge=0, le=10, description="Lookahead bias prevention days")
    n_restarts: int = Field(default=10, ge=1, le=20, description="Number of HMM training restarts")
    max_iter: int = Field(default=100, ge=10, le=1000, description="Maximum training iterations")
    tolerance: float = Field(default=1e-3, ge=1e-6, le=1e-1, description="Convergence tolerance")

class ProcessingConfig(BaseModel):
    """Data processing configuration schema."""
    engine: ProcessingEngine = Field(default=ProcessingEngine.STREAMING, description="Processing engine")
    chunk_size: int = Field(default=100000, ge=1000, le=10000000, description="Chunk size for processing")
    memory_limit: str = Field(default="8GB", description="Memory usage limit")
    parallel_workers: Optional[int] = Field(default=None, ge=1, le=32, description="Number of parallel workers")
    data_type_optimization: bool = Field(default=True, description="Enable data type optimization")

    @validator('memory_limit')
    def validate_memory_limit(cls, v):
        if not v.endswith(('GB', 'MB', 'KB')):
            raise ValueError('Memory limit must end with GB, MB, or KB')
        return v

class IndicatorConfig(BaseModel):
    """Technical indicator configuration."""
    name: str = Field(..., description="Indicator name")
    enabled: bool = Field(default=True, description="Enable/disable indicator")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")

    class Config:
        extra = "forbid"  # Prevent additional fields

class FeatureConfig(BaseModel):
    """Feature engineering configuration."""
    indicators: List[IndicatorConfig] = Field(
        default_factory=lambda: [
            IndicatorConfig(name="atr", parameters={"window": 14}),
            IndicatorConfig(name="rsi", parameters={"window": 14}),
            IndicatorConfig(name="bollinger", parameters={"window": 20, "std_dev": 2}),
            IndicatorConfig(name="macd", parameters={"fast": 12, "slow": 26, "signal": 9}),
            IndicatorConfig(name="adx", parameters={"window": 14})
        ],
        description="Technical indicators to compute"
    )
    validation: bool = Field(default=True, description="Enable feature validation")
    preprocessing: bool = Field(default=True, description="Enable data preprocessing")
    nan_handling: str = Field(default="drop", regex="^(drop|fill|interpolate)$", description="NaN handling strategy")

class HMMConfig(BaseModel):
    """HMM model configuration."""
    algorithm: str = Field(default="gaussian", regex="^(gaussian|gmm|custom)$", description="HMM algorithm type")
    n_components: int = Field(default=3, ge=2, le=10, description="Number of hidden states")
    covariance_type: CovarianceType = Field(default=CovarianceType.FULL, description="Covariance matrix type")
    n_iter: int = Field(default=100, ge=10, le=1000, description="Maximum EM iterations")
    tol: float = Field(default=1e-3, ge=1e-6, le=1e-1, description="Convergence tolerance")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    verbose: bool = Field(default=False, description="Verbose training output")

class LSTMConfig(BaseModel):
    """LSTM model configuration."""
    sequence_length: int = Field(default=60, ge=10, le=500, description="Input sequence length")
    hidden_units: List[int] = Field(default=[64, 64], description="LSTM hidden layer sizes")
    dropout_rate: float = Field(default=0.5, ge=0.0, le=0.9, description="Dropout rate")
    learning_rate: float = Field(default=0.001, ge=1e-5, le=1e-1, description="Learning rate")
    epochs: int = Field(default=100, ge=1, le=1000, description="Training epochs")
    batch_size: int = Field(default=32, ge=1, le=512, description="Training batch size")
    optimizer: str = Field(default="adam", regex="^(adam|sgd|rmsprop)$", description="Optimizer")
    early_stopping: bool = Field(default=True, description="Enable early stopping")

class BacktestingConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = Field(default=100000.0, ge=1000.0, description="Initial capital")
    commission: float = Field(default=0.001, ge=0.0, le=0.1, description="Commission rate")
    slippage: float = Field(default=0.0001, ge=0.0, le=0.01, description="Slippage rate")
    lookahead_bias_prevention: bool = Field(default=True, description="Prevent lookahead bias")
    position_sizing: str = Field(default="fixed", regex="^(fixed|volatility|kelly)$", description="Position sizing method")
    risk_management: bool = Field(default=True, description="Enable risk management")

class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    chart_style: str = Field(default="seaborn", regex="^(matplotlib|seaborn|plotly)$", description="Chart style")
    color_scheme: str = Field(default="viridis", description="Color scheme")
    interactive: bool = Field(default=False, description="Enable interactive plots")
    save_plots: bool = Field(default=True, description="Save plots to files")
    plot_directory: str = Field(default="./plots", description="Plot output directory")
    figure_size: tuple = Field(default=(12, 8), description="Default figure size")

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$", description="Log level")
    format: str = Field(default="%(asctime)s [%(levelname)s] %(message)s", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup files")
    console_output: bool = Field(default=True, description="Enable console output")

class PerformanceConfig(BaseModel):
    """Performance monitoring configuration."""
    monitoring: bool = Field(default=False, description="Enable performance monitoring")
    memory_threshold: float = Field(default=0.8, ge=0.5, le=0.95, description="Memory usage threshold")
    profiling: bool = Field(default=False, description="Enable code profiling")
    benchmarks: bool = Field(default=False, description="Run performance benchmarks")
    metrics_file: Optional[str] = Field(default=None, description="Performance metrics output file")

class RootConfig(BaseModel):
    """Root configuration schema."""
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    class Config:
        extra = "allow"  # Allow additional configuration sections
```

### 3. Configuration Loaders

**Multi-Format Configuration Loading**:
```python
# src/configuration/loaders.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import os

class ConfigurationLoader:
    """Configuration file loader supporting multiple formats."""

    @staticmethod
    def load_yaml(file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML file {file_path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {file_path}")

    @staticmethod
    def load_json(file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON file {file_path}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {file_path}")

    @staticmethod
    def detect_format(file_path: Path) -> str:
        """Detect configuration file format."""
        suffix = file_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return 'yaml'
        elif suffix == '.json':
            return 'json'
        else:
            raise ConfigurationError(f"Unsupported configuration format: {suffix}")

    @classmethod
    def load_file(cls, file_path: Path) -> Dict[str, Any]:
        """Load configuration file with format detection."""
        format_type = cls.detect_format(file_path)

        if format_type == 'yaml':
            return cls.load_yaml(file_path)
        elif format_type == 'json':
            return cls.load_json(file_path)
        else:
            raise ConfigurationError(f"Unsupported format: {format_type}")

class EnvironmentLoader:
    """Environment variable loader with prefix support."""

    def __init__(self, prefix: str = "HMM_"):
        self.prefix = prefix

    def load_environment(self) -> Dict[str, Any]:
        """Load environment variables with prefix."""
        env_config = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.prefix):].lower()

                # Convert nested keys (HMM_ANALYSIS_N_STATES -> analysis.n_states)
                config_key = self._convert_env_key(config_key)

                # Try to parse as JSON, fallback to string
                try:
                    env_config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Handle common types
                    env_config[config_key] = self._parse_value(value)

        return env_config

    def _convert_env_key(self, env_key: str) -> str:
        """Convert environment key to configuration key."""
        # Replace underscores with dots for nested keys
        return env_key.replace('_', '.')

    def _parse_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Integer values
        try:
            return int(value)
        except ValueError:
            pass

        # Float values
        try:
            return float(value)
        except ValueError:
            pass

        # String values
        return value
```

### 4. Configuration Validation

**Comprehensive Validation Framework**:
```python
# src/configuration/validators.py
from pydantic import ValidationError
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ConfigurationValidator:
    """Configuration validation and error reporting."""

    def __init__(self, schema_class=RootConfig):
        self.schema_class = schema_class
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        try:
            # Use Pydantic for validation
            validated_config = self.schema_class(**config)
            self._perform_custom_validations(validated_config.dict())

            if self.errors:
                logger.error(f"Configuration validation failed: {self.errors}")
                return False

            if self.warnings:
                logger.warning(f"Configuration warnings: {self.warnings}")

            return True

        except ValidationError as e:
            self.errors.extend([str(error) for error in e.errors()])
            logger.error(f"Configuration validation errors: {self.errors}")
            return False

    def _perform_custom_validations(self, config: Dict[str, Any]) -> None:
        """Perform custom validation beyond Pydantic schema."""

        # Validate data file paths
        if 'data' in config and 'input_path' in config['data']:
            input_path = Path(config['data']['input_path'])
            if not input_path.exists():
                self.errors.append(f"Input data file not found: {input_path}")

        # Validate output directory permissions
        if 'output' in config and 'directory' in config['output']:
            output_dir = Path(config['output']['directory'])
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    self.errors.append(f"Cannot create output directory: {output_dir}")

        # Validate model configurations
        self._validate_model_configs(config)

        # Validate processing engine compatibility
        self._validate_engine_compatibility(config)

    def _validate_model_configs(self, config: Dict[str, Any]) -> None:
        """Validate model-specific configurations."""

        models = config.get('models', {})

        # Validate HMM configuration
        if 'hmm' in models:
            hmm_config = models['hmm']
            if hmm_config.get('n_components', 1) < 2:
                self.errors.append("HMM n_components must be at least 2")

        # Validate LSTM configuration
        if 'lstm' in models:
            lstm_config = models['lstm']
            if lstm_config.get('sequence_length', 1) < 10:
                self.warnings.append("LSTM sequence length should be at least 10")

            if lstm_config.get('hidden_units', []):
                hidden_units = lstm_config['hidden_units']
                if any(units <= 0 for units in hidden_units):
                    self.errors.append("LSTM hidden units must be positive")

    def _validate_engine_compatibility(self, config: Dict[str, Any]) -> None:
        """Validate processing engine compatibility with configuration."""

        processing = config.get('processing', {})
        engine = processing.get('engine')
        chunk_size = processing.get('chunk_size')

        # Validate engine-specific configurations
        if engine == 'dask' and chunk_size:
            if chunk_size < 1000:
                self.warnings.append("Dask engine may benefit from larger chunk sizes")

        if engine == 'daft' and chunk_size:
            if chunk_size < 10000:
                self.warnings.append("Daft engine works best with larger chunk sizes")

    def get_validation_report(self) -> str:
        """Get detailed validation report."""
        report = []

        if self.errors:
            report.append("Configuration Errors:")
            for error in self.errors:
                report.append(f"  ❌ {error}")

        if self.warnings:
            report.append("Configuration Warnings:")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")

        if not self.errors and not self.warnings:
            report.append("✅ Configuration validation passed")

        return "\n".join(report)
```

### 5. Default Configuration

**Comprehensive Default Configuration**:
```python
# src/configuration/defaults.py
from typing import Dict, Any

DEFAULT_CONFIGURATION = {
    "analysis": {
        "n_states": 3,
        "covariance_type": "full",
        "random_state": 42,
        "test_size": 0.2,
        "lookahead_days": 1,
        "n_restarts": 10,
        "max_iter": 100,
        "tolerance": 1e-3
    },

    "processing": {
        "engine": "streaming",
        "chunk_size": 100000,
        "memory_limit": "8GB",
        "parallel_workers": None,
        "data_type_optimization": True
    },

    "features": {
        "indicators": [
            {
                "name": "log_returns",
                "enabled": True,
                "parameters": {}
            },
            {
                "name": "atr",
                "enabled": True,
                "parameters": {"window": 14}
            },
            {
                "name": "rsi",
                "enabled": True,
                "parameters": {"window": 14}
            },
            {
                "name": "bollinger_bands",
                "enabled": True,
                "parameters": {"window": 20, "std_dev": 2}
            },
            {
                "name": "macd",
                "enabled": True,
                "parameters": {"fast": 12, "slow": 26, "signal": 9}
            },
            {
                "name": "adx",
                "enabled": True,
                "parameters": {"window": 14}
            },
            {
                "name": "stochastic",
                "enabled": True,
                "parameters": {"window": 14, "smooth_window": 3}
            },
            {
                "name": "sma_ratio",
                "enabled": True,
                "parameters": {"window": 5}
            },
            {
                "name": "volume_ratio",
                "enabled": True,
                "parameters": {"window": 3}
            }
        ],
        "validation": True,
        "preprocessing": True,
        "nan_handling": "drop"
    },

    "models": {
        "hmm": {
            "algorithm": "gaussian",
            "n_components": 3,
            "covariance_type": "full",
            "n_iter": 100,
            "tol": 1e-3,
            "random_state": 42,
            "verbose": False
        },
        "lstm": {
            "sequence_length": 60,
            "hidden_units": [64, 64],
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "optimizer": "adam",
            "early_stopping": True
        }
    },

    "backtesting": {
        "initial_capital": 100000.0,
        "commission": 0.001,
        "slippage": 0.0001,
        "lookahead_bias_prevention": True,
        "position_sizing": "fixed",
        "risk_management": True,
        "strategies": [
            {
                "name": "regime_based",
                "type": "hmm_states",
                "parameters": {
                    "long_states": [0],
                    "short_states": [2],
                    "neutral_states": [1]
                }
            }
        ]
    },

    "visualization": {
        "chart_style": "seaborn",
        "color_scheme": "viridis",
        "interactive": False,
        "save_plots": True,
        "plot_directory": "./plots",
        "figure_size": [12, 8],
        "dpi": 300
    },

    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(message)s",
        "file_path": None,
        "max_file_size": "10MB",
        "backup_count": 5,
        "console_output": True
    },

    "performance": {
        "monitoring": False,
        "memory_threshold": 0.8,
        "profiling": False,
        "benchmarks": False,
        "metrics_file": None
    }
}
```

---

## CLI Configuration Integration

### 1. Unified CLI Configuration

**CLI Argument Mapping**:
```python
# src/cli/config.py
import click
from pathlib import Path
from typing import Dict, Any, Optional
from ..configuration.manager import ConfigurationManager
from ..configuration.loaders import ConfigurationLoader

class CLIConfigurationManager:
    """CLI-specific configuration management."""

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config_file_path: Optional[Path] = None

    def setup_cli_options(self, ctx: click.Context,
                          config_file: Optional[Path] = None,
                          log_level: Optional[str] = None,
                          quiet: Optional[bool] = None,
                          verbose: Optional[bool] = None) -> Dict[str, Any]:
        """Setup CLI options and load configuration."""

        # Store CLI arguments
        cli_args = {}

        # Handle logging level
        if log_level:
            cli_args['logging.level'] = log_level.upper()

        # Handle quiet/verbose modes
        if quiet:
            cli_args['logging.level'] = 'ERROR'
        elif verbose:
            cli_args['logging.level'] = 'DEBUG'

        # Store configuration file path
        self.config_file_path = config_file

        # Load configuration
        config_file_paths = [config_file] if config_file else self._find_config_files()

        config = self.config_manager.load_configuration(
            cli_args=cli_args,
            config_file_paths=config_file_paths
        )

        # Store in context for use by commands
        ctx.ensure_object(dict)
        ctx.obj['config'] = config
        ctx.obj['config_manager'] = self.config_manager

        return config

    def _find_config_files(self) -> list[Path]:
        """Find configuration files in standard locations."""
        config_files = []

        # Standard configuration locations
        search_paths = [
            Path.cwd() / "hmm_config.yaml",
            Path.cwd() / "hmm_config.yml",
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path.cwd() / ".hmm" / "config.yaml",
            Path.home() / ".hmm" / "config.yaml"
        ]

        for path in search_paths:
            if path.exists():
                config_files.append(path)
                break  # Use first found

        return config_files

# CLI option decorators with configuration integration
@click.option('--config-file',
              type=click.Path(exists=True, path_type=Path),
              help='Configuration file path (YAML/JSON)')
@click.option('--log-level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              help='Logging level')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Suppress output except errors')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Enable verbose output')
def setup_configuration(ctx, config_file, log_level, quiet, verbose):
    """Setup configuration for CLI commands."""
    config_manager = CLIConfigurationManager()
    config = config_manager.setup_cli_options(ctx, config_file, log_level, quiet, verbose)
    return config
```

### 2. Command-Specific Configuration

**Analyze Command Configuration**:
```python
# src/cli/commands/analyze.py
import click
from ..config import setup_configuration

@click.command()
@click.option('--input-csv', '-i',
              type=click.Path(exists=True, path_type=Path),
              required=True,
              help='Input CSV file with futures data')
@click.option('--output-dir', '-o',
              type=click.Path(path_type=Path),
              default=Path('./output'),
              help='Output directory for results')
@click.option('--n-states', '-n',
              type=click.IntRange(min=2, max=10),
              help='Number of HMM states (overrides config file)')
@click.option('--engine',
              type=click.Choice(['streaming', 'dask', 'daft']),
              help='Processing engine (overrides config file)')
@click.option('--chunk-size',
              type=int,
              help='Chunk size for processing (overrides config file)')
@click.pass_context
def analyze(ctx, input_csv, output_dir, n_states, engine, chunk_size):
    """Run HMM analysis with enhanced configuration."""

    config = ctx.obj['config']

    # Override configuration with CLI arguments
    cli_overrides = {}

    if n_states is not None:
        cli_overrides['analysis.n_states'] = n_states

    if engine is not None:
        cli_overrides['processing.engine'] = engine

    if chunk_size is not None:
        cli_overrides['processing.chunk_size'] = chunk_size

    # Update configuration
    config_manager = ctx.obj['config_manager']
    config_manager.update(cli_overrides)

    # Use updated configuration
    updated_config = config_manager.get('')

    # Execute analysis with configuration
    execute_analysis(updated_config, input_csv, output_dir)
```

---

## Configuration File Examples

### 1. YAML Configuration File

**hmm_config.yaml**:
```yaml
# HMM Analysis Configuration
# This file contains all configuration parameters for the HMM analysis system

# Analysis Configuration
analysis:
  n_states: 3                    # Number of hidden states
  covariance_type: "full"          # Covariance matrix type
  random_state: 42                # Random seed for reproducibility
  test_size: 0.2                  # Test data proportion
  lookahead_days: 1               # Lookahead bias prevention
  n_restarts: 10                  # Number of training restarts
  max_iter: 100                   # Maximum EM iterations
  tolerance: 0.001                # Convergence tolerance

# Data Processing Configuration
processing:
  engine: "streaming"             # Processing engine
  chunk_size: 100000             # Chunk size for processing
  memory_limit: "8GB"             # Memory usage limit
  parallel_workers: null          # Number of parallel workers
  data_type_optimization: true  # Enable data type optimization

# Feature Engineering Configuration
features:
  indicators:
    - name: "log_returns"
      enabled: true
      parameters: {}

    - name: "atr"
      enabled: true
      parameters:
        window: 14

    - name: "rsi"
      enabled: true
      parameters:
        window: 14

    - name: "bollinger_bands"
      enabled: true
      parameters:
        window: 20
        std_dev: 2

    - name: "macd"
      enabled: true
      parameters:
        fast: 12
        slow: 26
        signal: 9

    - name: "adx"
      enabled: true
      parameters:
        window: 14

    - name: "stochastic"
      enabled: true
      parameters:
        window: 14
        smooth_window: 3

    - name: "sma_ratio"
      enabled: true
      parameters:
        window: 5

    - name: "volume_ratio"
      enabled: true
      parameters:
        window: 3

  validation: true                # Enable feature validation
  preprocessing: true             # Enable data preprocessing
  nan_handling: "drop"           # NaN handling strategy

# Model Configuration
models:
  hmm:
    algorithm: "gaussian"        # HMM algorithm type
    n_components: 3              # Number of hidden states
    covariance_type: "full"      # Covariance matrix type
    n_iter: 100                 # Maximum EM iterations
    tol: 0.001                  # Convergence tolerance
    random_state: 42            # Random seed for reproducibility
    verbose: false               # Verbose training output

  lstm:
    sequence_length: 60          # Input sequence length
    hidden_units: [64, 64]      # LSTM hidden layer sizes
    dropout_rate: 0.5           # Dropout rate
    learning_rate: 0.001         # Learning rate
    epochs: 100                 # Training epochs
    batch_size: 32              # Training batch size
    optimizer: "adam"            # Optimizer
    early_stopping: true         # Enable early stopping

# Backtesting Configuration
backtesting:
  initial_capital: 100000.0      # Initial capital
  commission: 0.001               # Commission rate
  slippage: 0.0001               # Slippage rate
  lookahead_bias_prevention: true # Prevent lookahead bias
  position_sizing: "fixed"        # Position sizing method
  risk_management: true           # Enable risk management

  strategies:
    - name: "regime_based"
      type: "hmm_states"
      parameters:
        long_states: [0]
        short_states: [2]
        neutral_states: [1]

# Visualization Configuration
visualization:
  chart_style: "seaborn"          # Chart style
  color_scheme: "viridis"        # Color scheme
  interactive: false              # Enable interactive plots
  save_plots: true               # Save plots to files
  plot_directory: "./plots"       # Plot output directory
  figure_size: [12, 8]           # Default figure size
  dpi: 300                      # Plot resolution

# Logging Configuration
logging:
  level: "INFO"                  # Log level
  format: "%(asctime)s [%(levelname)s] %(message)s"  # Log format
  file_path: null                # Log file path
  max_file_size: "10MB"           # Maximum log file size
  backup_count: 5                 # Number of backup files
  console_output: true            # Enable console output

# Performance Configuration
performance:
  monitoring: false               # Enable performance monitoring
  memory_threshold: 0.8           # Memory usage threshold
  profiling: false               # Enable code profiling
  benchmarks: false               # Run performance benchmarks
  metrics_file: null             # Performance metrics output file

# Output Configuration
output:
  directory: "./output"           # Output directory
  save_model: true               # Save trained models
  save_charts: true              # Save visualization charts
  save_dashboard: true           # Save interactive dashboard
  save_report: true              # Save analysis report
  create_subdirectories: true   # Create timestamped subdirectories
```

### 2. JSON Configuration File

**config.json**:
```json
{
  "analysis": {
    "n_states": 4,
    "covariance_type": "diag",
    "random_state": 123,
    "test_size": 0.25,
    "n_restarts": 15
  },
  "processing": {
    "engine": "dask",
    "chunk_size": 50000,
    "parallel_workers": 4
  },
  "features": {
    "indicators": [
      {
        "name": "atr",
        "enabled": true,
        "parameters": {"window": 20}
      },
      {
        "name": "rsi",
        "enabled": true,
        "parameters": {"window": 21}
      }
    ]
  },
  "backtesting": {
    "initial_capital": 50000.0,
    "commission": 0.002,
    "strategies": [
      {
        "name": "aggressive",
        "parameters": {"position_multiplier": 1.5}
      }
    ]
  }
}
```

---

## Migration Implementation Plan

### Phase 1: Core Configuration Framework (8 hours)

**Task 1.1: Create Configuration Manager (3 hours)**
- Implement `ConfigurationManager` class
- Add configuration loading from multiple sources
- Implement configuration hierarchy and merging logic
- Add basic validation framework

**Task 1.2: Implement Schema Validation (3 hours)**
- Create Pydantic configuration models
- Implement comprehensive validation rules
- Add custom validation logic
- Create error reporting system

**Task 1.3: Create Configuration Loaders (2 hours)**
- Implement YAML and JSON configuration loaders
- Add environment variable loading
- Implement format detection and parsing
- Add error handling for invalid configurations

### Phase 2: CLI Integration (6 hours)

**Task 2.1: Integrate CLI with Configuration (3 hours)**
- Create CLI configuration manager
- Implement CLI argument override logic
- Add configuration file discovery
- Update CLI commands to use configuration

**Task 2.2: Create Configuration Commands (3 hours)**
- Add configuration validation command
- Create configuration generation commands
- Implement configuration debugging tools
- Add configuration documentation commands

### Phase 3: Default Configuration (4 hours)

**Task 3.1: Create Default Configuration (2 hours)**
- Implement comprehensive default configuration
- Add module-specific default values
- Create configuration templates
- Add configuration validation for defaults

**Task 3.2: Configuration Examples (2 hours)**
- Create YAML configuration example
- Create JSON configuration example
- Add configuration documentation
- Create configuration validation examples

### Phase 4: Advanced Features (4 hours)

**Task 4.1: Dynamic Configuration Updates (2 hours)**
- Implement runtime configuration updates
- Add configuration change monitoring
- Create configuration reload functionality
- Add configuration change callbacks

**Task 4.2: Configuration Profile Support (2 hours)**
- Implement configuration profiles (dev, prod, test)
- Add profile switching capabilities
- Create profile-specific configurations
- Add profile validation and management

---

## Testing Strategy

### Unit Tests

**Configuration Manager Tests**:
```python
def test_configuration_hierarchy():
    # Test configuration source hierarchy
    config_manager = ConfigurationManager()

    # Test default configuration
    default_config = config_manager.get_configuration()
    assert default_config['analysis']['n_states'] == 3

    # Test CLI argument override
    cli_args = {'analysis.n_states': 5}
    config_manager.load_configuration(cli_args=cli_args)
    assert config_manager.get('analysis.n_states') == 5

def test_configuration_validation():
    # Test configuration validation
    validator = ConfigurationValidator()

    # Valid configuration
    valid_config = {
        'analysis': {'n_states': 3, 'covariance_type': 'full'},
        'processing': {'engine': 'streaming', 'chunk_size': 100000}
    }
    assert validator.validate(valid_config) == True

    # Invalid configuration
    invalid_config = {
        'analysis': {'n_states': 1},  # Too few states
        'processing': {'engine': 'invalid'}
    }
    assert validator.validate(invalid_config) == False
```

**Configuration Loader Tests**:
```python
def test_yaml_loader():
    # Test YAML configuration loading
    loader = ConfigurationLoader()

    # Valid YAML
    valid_yaml = {
        'analysis': {'n_states': 3},
        'processing': {'engine': 'streaming'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_yaml, f)
        config = loader.load_yaml(Path(f.name))
        assert config == valid_yaml

def test_environment_loader():
    # Test environment variable loading
    os.environ['HMM_ANALYSIS_N_STATES'] = '5'
    os.environ['HMM_PROCESSING_ENGINE'] = 'dask'

    loader = EnvironmentLoader('HMM_')
    env_config = loader.load_environment()

    assert env_config['analysis.n_states'] == 5
    assert env_config['processing.engine'] == 'dask'
```

### Integration Tests

**CLI Configuration Tests**:
```python
def test_cli_configuration_integration():
    # Test CLI with configuration file
    runner = CliRunner()

    # Create test configuration file
    config_content = {
        'analysis': {'n_states': 4},
        'processing': {'engine': 'dask'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_content, f)

        # Test CLI with config file
        result = runner.invoke(cli, [
            '--config-file', f.name,
            'analyze',
            '--input-csv', 'test_data.csv'
        ])

        assert result.exit_code == 0
        assert 'Configuration loaded' in result.output

def test_cli_argument_override():
    # Test CLI argument override
    runner = CliRunner()

    result = runner.invoke(cli, [
        'analyze',
        '--input-csv', 'test_data.csv',
        '--n-states', '5',
        '--engine', 'daft'
    ])

    assert result.exit_code == 0
    # CLI arguments should override defaults
```

### Performance Tests

**Configuration Loading Performance**:
```python
def test_configuration_loading_performance():
    # Test performance with large configuration
    large_config = generate_large_configuration()

    loader = ConfigurationLoader()

    start_time = time.time()
    loaded_config = loader.load_yaml(Path('large_config.yaml'))
    loading_time = time.time() - start_time

    assert loading_time < 1.0  # Should load within 1 second

def test_configuration_memory_usage():
    # Test memory usage of configuration system
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Measure memory before configuration loading
    memory_before = process.memory_info().rss

    # Load configuration
    config_manager = ConfigurationManager()
    config_manager.load_configuration()

    # Measure memory after configuration loading
    memory_after = process.memory_info().rss

    memory_increase = memory_after - memory_before
    assert memory_increase < 10 * 1024 * 1024  # Less than 10MB increase
```

---

## Success Criteria

### Functional Requirements

- [ ] Configuration loaded from all sources (CLI, env vars, files, defaults)
- [ ] Configuration hierarchy respected (CLI > env > file > defaults)
- [ ] All configuration types validated against schemas
- [ ] CLI arguments override configuration file values
- [ ] Environment variables override file values
- [ ] Default configuration values applied when no other source

### Technical Requirements

- [ ] Pydantic-based validation with clear error messages
- [ ] Support for YAML and JSON configuration files
- [ ] Environment variable loading with prefix support
- [ ] Runtime configuration updates supported
- [ ] Configuration profile support (dev/prod/test)
- [ ] Memory-efficient configuration loading

### Quality Requirements

- [ ] Comprehensive error handling and reporting
- [ ] Configuration documentation with examples
- [ ] Unit test coverage >90% for configuration system
- [ ] Integration tests for CLI configuration
- [ ] Performance benchmarks for loading time
- [ ] Memory usage optimization for large configurations

---

## Conclusion

The configuration consolidation plan provides a comprehensive, unified configuration management system that will:

1. **Centralize Configuration**: Single source of truth for all configuration
2. **Support Multiple Sources**: CLI arguments, environment variables, config files, defaults
3. **Provide Type Safety**: Pydantic-based validation with clear error messages
4. **Enable Flexibility**: Extensible schema for future configuration needs
5. **Improve User Experience**: Clear configuration examples and validation

This configuration system will serve as the foundation for the enhanced src directory architecture, enabling consistent, validated configuration across all modules and components while providing the flexibility needed for complex analysis scenarios.

---

*Configuration Consolidation Plan Completed: October 23, 2025*
*Next Step: Phase 1.2.4 - Design testing framework*