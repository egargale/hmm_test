"""
Configuration Management Module

Implements Pydantic-based configuration management for loading and validating
application settings from YAML/JSON configuration files.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class HMMConfig(BaseModel):
    """Configuration for Hidden Markov Model parameters."""

    n_states: int = Field(3, ge=1, le=10, description="Number of hidden states")
    covariance_type: str = Field(
        "diag", description="Covariance type ('full', 'diag', 'tied', 'spherical')"
    )
    max_iter: int = Field(100, ge=1, description="Maximum number of EM iterations")
    random_state: int = Field(42, ge=0, description="Random state for reproducibility")
    tol: float = Field(1e-3, ge=1e-6, description="Convergence threshold")
    num_restarts: int = Field(
        3, ge=1, description="Number of restarts with different random states"
    )

    @field_validator("covariance_type")
    @classmethod
    def validate_covariance_type(cls, v: str) -> str:
        """Validate covariance type is one of the allowed values."""
        allowed_types = ["full", "diag", "tied", "spherical"]
        if v not in allowed_types:
            raise ValueError(f"covariance_type must be one of {allowed_types}")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for data processing engine."""

    engine_type: str = Field(
        "streaming", description="Processing engine ('streaming', 'dask', 'daft')"
    )
    chunk_size: int = Field(
        10000, ge=1, description="Chunk size for streaming processing"
    )
    memory_limit_gb: float = Field(8.0, ge=0.5, description="Memory limit in GB")
    enable_validation: bool = Field(True, description="Enable data validation")
    downcast_floats: bool = Field(
        True, description="Downcast float64 to float32 for memory efficiency"
    )

    # Technical indicators configuration
    indicators: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "sma": {"length": 20},
            "ema": {"length": 20},
            "rsi": {"length": 14},
            "atr": {"length": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"length": 20, "std": 2},
            "roc": {"length": 10},
            "stochastic": {"k": 14, "d": 3},
            "adx": {"length": 14},
            "volume_sma": {"length": 20},
        },
        description="Technical indicators configuration",
    )

    @field_validator("engine_type")
    @classmethod
    def validate_engine_type(cls, v: str) -> str:
        """Validate engine type is one of the allowed values."""
        allowed_types = ["streaming", "dask", "daft", "auto"]
        if v not in allowed_types:
            raise ValueError(f"engine_type must be one of {allowed_types}")
        return v


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    initial_capital: float = Field(
        100000.0, gt=0, description="Initial capital for backtesting"
    )
    commission_per_trade: float = Field(0.0, ge=0, description="Commission per trade")
    slippage_bps: float = Field(0.0, ge=0, description="Slippage in basis points")
    position_size: float = Field(1.0, gt=0, description="Position size multiplier")
    lookahead_lag: int = Field(
        1, ge=0, description="Number of periods to lag for lookahead bias prevention"
    )

    # State to position mapping (e.g., {0: 1, 1: -1, 2: 0} for long/short/flat)
    state_map: Optional[Dict[int, int]] = Field(
        None,
        description="Mapping from HMM states to trading positions (1=long, -1=short, 0=flat)",
    )

    @field_validator("state_map")
    @classmethod
    def set_default_state_map(cls, v: Optional[Dict[int, int]]) -> Dict[int, int]:
        """Set default state mapping if not provided."""
        if v is None:
            return {0: 1, 1: -1, 2: 0}  # Default: state 0=long, 1=short, 2=flat
        return v

    @field_validator("state_map")
    @classmethod
    def validate_state_map(cls, v: Dict[int, int]) -> Dict[int, int]:
        """Validate state mapping values."""
        for state, position in v.items():
            if state < 0:
                raise ValueError(f"State ID {state} cannot be negative")
            if position not in [-1, 0, 1]:
                raise ValueError(
                    f"Position {position} for state {state} must be -1, 0, or 1"
                )
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Log message format",
    )
    date_format: str = Field("%Y-%m-%d %H:%M:%S", description="Date format for logs")
    file_path: Optional[str] = Field(None, description="Optional log file path")
    max_file_size: str = Field("10 MB", description="Maximum log file size")
    backup_count: int = Field(5, ge=0, description="Number of backup log files")
    enable_rotation: bool = Field(True, description="Enable log file rotation")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"level must be one of {allowed_levels}")
        return v.upper()


class Config(BaseModel):
    """Main configuration class combining all sub-configurations."""

    hmm: HMMConfig = Field(default_factory=HMMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields
        use_enum_values=True,
    )


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML config file {config_path}: {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(e.msg, e.doc, e.pos) from e

    if data is None:
        data = {}

    return Config(**data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.

    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration file

    Raises:
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.dict()

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    except Exception as e:
        raise RuntimeError(f"Error saving config file {config_path}: {e}") from e


def create_default_config(config_path: Union[str, Path]) -> Config:
    """
    Create a default configuration file.

    Args:
        config_path: Path where to create the default configuration file

    Returns:
        Config: Default configuration object
    """
    config = Config()
    save_config(config, config_path)
    return config
