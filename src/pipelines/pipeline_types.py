"""
Type definitions for pipeline components and configurations.

This module contains dataclasses and enums that define the structure
and configuration options for HMM processing pipelines.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class PipelineStage(Enum):
    """Pipeline processing stages"""

    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    RESULTS_SAVING = "results_saving"
    BACKTESTING = "backtesting"
    VISUALIZATION = "visualization"
    COMPLETED = "completed"


class PipelineStatus(Enum):
    """Pipeline execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingMode(Enum):
    """Data processing modes"""

    STREAMING = "streaming"  # Process large files in chunks
    MEMORY = "memory"  # Load entire dataset into memory
    HYBRID = "hybrid"  # Use streaming for large, memory for small datasets


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""

    # Technical indicators
    enable_log_returns: bool = True
    enable_atr: bool = True
    enable_roc: bool = True
    enable_rsi: bool = True
    enable_bollinger_bands: bool = True
    enable_adx: bool = True
    enable_stochastic: bool = True
    enable_sma_ratios: bool = True
    enable_price_position: bool = True
    enable_volume_features: bool = True

    # Indicator parameters
    atr_window: int = 3
    roc_window: int = 3
    rsi_window: int = 3
    bollinger_window: int = 3
    bollinger_std_dev: float = 2.0
    adx_window: int = 3
    stoch_window: int = 3
    stoch_smooth_window: int = 3
    sma_window: int = 5
    volume_window: int = 3

    # Custom features
    custom_features: List[str] = field(default_factory=list)
    feature_selection: Optional[List[str]] = None

    # Feature processing
    normalize_features: bool = True
    handle_missing: str = "drop"  # "drop", "fill", "interpolate"
    fill_method: str = "forward"  # "forward", "backward", "mean", "median"


@dataclass
class TrainingConfig:
    """Configuration for HMM model training"""

    # Model parameters
    n_states: int = 3
    covariance_type: str = "diag"  # "full", "diag", "tied", "spherical"
    n_iter: int = 100
    tol: float = 1e-6
    min_covar: float = 1e-3
    random_state: int = 42
    verbose: bool = True

    # Training options
    init_params: str = "stmc"  # "stmc", "stm", "smc", etc.
    implementation: str = "log"  # "log", "scaling"

    # Model selection
    auto_select_states: bool = False
    state_range: tuple = (2, 8)
    selection_metric: str = "bic"  # "aic", "bic", "hqc"

    # Validation
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping: bool = False
    patience: int = 10


@dataclass
class PersistenceConfig:
    """Configuration for model and results persistence"""

    # Model saving
    save_model: bool = True
    model_format: str = "pickle"  # "pickle", "joblib", "custom"
    model_path: Optional[Path] = None
    compress_model: bool = False

    # Results saving
    save_results: bool = True
    results_format: str = "csv"  # "csv", "parquet", "hdf5"
    results_path: Optional[Path] = None

    # Metadata
    save_metadata: bool = True
    metadata_format: str = "json"  # "json", "yaml"

    # Versioning
    version_models: bool = False
    max_versions: int = 5

    # Backup
    create_backup: bool = True
    backup_frequency: str = "daily"  # "daily", "weekly", "monthly"


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""

    # Chunk processing
    chunk_size: int = 100_000
    max_memory_mb: int = 1024
    processing_mode: ProcessingMode = ProcessingMode.HYBRID

    # Parallel processing
    n_workers: int = 1
    use_multiprocessing: bool = False

    # Progress tracking
    show_progress: bool = True
    progress_update_interval: int = 1000

    # Error handling
    skip_errors: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0

    # Memory optimization
    downcast_dtypes: bool = True
    memory_cleanup_interval: int = 10

    # Data validation
    validate_chunks: bool = True
    strict_validation: bool = False


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""

    # Strategy parameters
    strategy_type: str = "state_based"  # "state_based", "threshold", "custom"
    long_states: List[int] = field(default_factory=lambda: [0])
    short_states: List[int] = field(default_factory=lambda: [2])

    # Position sizing
    position_sizing: str = "fixed"  # "fixed", "volatility", "kelly", "custom"
    position_size: float = 1.0
    max_position_size: float = 1.0

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Transaction costs
    commission: float = 0.0
    slippage: float = 0.0
    financing_cost: float = 0.0

    # Metrics
    metrics: List[str] = field(
        default_factory=lambda: [
            "sharpe_ratio",
            "max_drawdown",
            "total_return",
            "win_rate",
            "profit_factor",
            "calmar_ratio",
        ]
    )

    # Benchmarking
    benchmark: Optional[str] = None  # "buy_and_hold", "random", custom path

    # Output
    save_trades: bool = True
    save_equity_curve: bool = True
    save_performance_report: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""

    # Component configurations
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    backtesting: Optional[BacktestConfig] = None

    # Pipeline settings
    name: str = "hmm_pipeline"
    description: str = ""
    seed: int = 42

    # I/O settings
    input_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    enable_caching: bool = True
    cache_dir: Optional[Path] = None

    # Logging and monitoring
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(message)s"
    enable_progress_bar: bool = True
    enable_timing: bool = True

    # Error handling
    continue_on_error: bool = False
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary"""
        # Handle nested dataclasses
        features = FeatureConfig(**config_dict.get("features", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        persistence = PersistenceConfig(**config_dict.get("persistence", {}))
        streaming = StreamingConfig(**config_dict.get("streaming", {}))

        backtesting = None
        if "backtesting" in config_dict and config_dict["backtesting"]:
            backtesting = BacktestConfig(**config_dict["backtesting"])

        return cls(
            features=features,
            training=training,
            persistence=persistence,
            streaming=streaming,
            backtesting=backtesting,
            **{
                k: v
                for k, v in config_dict.items()
                if k
                not in [
                    "features",
                    "training",
                    "persistence",
                    "streaming",
                    "backtesting",
                ]
            },
        )


@dataclass
class PipelineResult:
    """Results from pipeline execution"""

    # Execution metadata
    pipeline_name: str
    execution_time: float
    status: PipelineStatus
    stages_completed: List[PipelineStage]

    # Data results
    processed_data: Optional[Any] = None  # pd.DataFrame
    features: Optional[Any] = None  # np.ndarray
    states: Optional[np.ndarray] = None

    # Model results
    model: Optional[Any] = None
    model_metrics: Optional[Dict[str, float]] = None

    # Backtesting results
    backtest_results: Optional[Any] = None
    performance_metrics: Optional[Dict[str, float]] = None

    # File paths
    output_files: Dict[str, Path] = field(default_factory=dict)

    # Logs and metadata
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        """Check if pipeline completed successfully"""
        return self.status == PipelineStatus.COMPLETED and len(self.errors) == 0

    def get_output_path(self, output_type: str) -> Optional[Path]:
        """Get path for specific output type"""
        return self.output_files.get(output_type)

    def add_output_file(self, output_type: str, path: Path) -> None:
        """Add output file path"""
        self.output_files[output_type] = path

    def add_log(self, message: str) -> None:
        """Add log message"""
        self.logs.append(message)

    def add_error(self, error: str) -> None:
        """Add error message"""
        self.errors.append(error)


@dataclass
class ProcessingStats:
    """Statistics for data processing"""

    # Data stats
    total_rows: int = 0
    processed_rows: int = 0
    dropped_rows: int = 0
    memory_used_mb: float = 0.0

    # Time stats
    loading_time: float = 0.0
    feature_time: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0

    # Model stats
    convergence_iterations: int = 0
    log_likelihood: float = 0.0
    aic: Optional[float] = None
    bic: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        import dataclasses

        return dataclasses.asdict(self)
