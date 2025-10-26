"""
Unified HMM training and inference pipeline.

This module provides a high-level pipeline interface that combines
data processing, feature engineering, model training, and results
persistence into a unified workflow.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data_processing.feature_engineering import FeatureEngineer
from ..data_processing.streaming_processor import StreamingDataProcessor
from ..model_training.hmm_trainer import train_single_hmm_model, train_model, HMMTrainingResult


class HMMTrainer:
    """Compatibility wrapper for HMM training functions."""

    def __init__(self, config=None):
        self.config = config

    def train(self, features, n_components=3, **kwargs):
        """Train HMM model using the underlying train_single_hmm_model function."""
        return train_single_hmm_model(
            features=features,
            n_components=n_components,
            config=self.config,
            **kwargs
        )
from ..model_training.model_persistence import save_model, load_model, get_model_info


class ModelPersistence:
    """Compatibility wrapper for model persistence functions."""

    def __init__(self, config=None):
        self.config = config

    def save(self, model, scaler, filepath, **kwargs):
        """Save model and scaler using the underlying save_model function."""
        return save_model(
            model=model,
            scaler=scaler,
            filepath=filepath,
            **kwargs
        )

    def load(self, filepath, **kwargs):
        """Load model and scaler using the underlying load_model function."""
        return load_model(filepath=filepath, **kwargs)

    def get_info(self, filepath, **kwargs):
        """Get model info using the underlying get_model_info function."""
        return get_model_info(path=filepath, **kwargs)
from ..backtesting.strategy_engine import StrategyEngine
from ..backtesting.performance_analyzer import PerformanceAnalyzer
from ..utils.logging_config import setup_logger
from .pipeline_types import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    PipelineStatus,
    ProcessingStats
)


logger = setup_logger(__name__)


class PipelineError(Exception):
    """Base pipeline exception"""
    pass


class DataProcessingError(PipelineError):
    """Data processing specific errors"""
    pass


class ModelTrainingError(PipelineError):
    """Model training specific errors"""
    pass


class HMMPipeline:
    """
    Unified HMM training and inference pipeline.

    This class provides a complete workflow for processing financial data,
    training HMM models, and generating results including backtesting.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the HMM pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = ProcessingStats()
        self.result = PipelineResult(
            pipeline_name=config.name,
            execution_time=0.0,
            status=PipelineStatus.PENDING,
            stages_completed=[]
        )

        # Initialize components
        self.feature_engineer = None
        self.streaming_processor = None
        self.trainer = None
        self.persistence = None
        self.strategy_engine = None
        self.performance_analyzer = None

        # Internal state
        self._current_stage = None
        self._start_time = None

        logger.info(f"Initialized pipeline: {config.name}")

    def _initialize_components(self) -> None:
        """Initialize pipeline components"""
        try:
            self._update_stage(PipelineStage.INITIALIZATION)

            # Initialize feature engineer
            self.feature_engineer = FeatureEngineer(self.config.features)

            # Initialize streaming processor
            self.streaming_processor = StreamingDataProcessor(self.config.streaming)

            # Initialize trainer
            self.trainer = HMMTrainer(self.config.training)

            # Initialize persistence
            self.persistence = ModelPersistence(self.config.persistence)

            # Initialize backtesting components if enabled
            if self.config.backtesting:
                self.strategy_engine = StrategyEngine(self.config.backtesting)
                self.performance_analyzer = PerformanceAnalyzer()

            logger.info("All pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise PipelineError(f"Component initialization failed: {e}")

    def _update_stage(self, stage: PipelineStage) -> None:
        """Update current pipeline stage"""
        self._current_stage = stage
        self.result.stages_completed.append(stage)
        logger.info(f"Pipeline stage: {stage.value}")

    def _start_timing(self) -> None:
        """Start execution timing"""
        self._start_time = time.perf_counter()

    def _end_timing(self) -> float:
        """End execution timing and return elapsed time"""
        if self._start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        self.result.execution_time = elapsed
        return elapsed

    async def run(self, data_path: Path) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            data_path: Path to input data file

        Returns:
            Pipeline execution results
        """
        try:
            self._start_timing()
            self.result.status = PipelineStatus.RUNNING

            # Validate inputs
            self._validate_inputs(data_path)

            # Initialize components
            self._initialize_components()

            # Load and process data
            processed_data = await self._load_and_process_data(data_path)

            # Extract features
            features = self._extract_features(processed_data)

            # Train or load model
            model, states = await self._train_or_load_model(features)

            # Save results
            self._save_results(processed_data, states, model)

            # Run backtesting if enabled
            if self.config.backtesting:
                await self._run_backtesting(processed_data, states)

            # Generate reports
            self._generate_reports()

            # Complete pipeline
            self._complete_pipeline()

            return self.result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.result.status = PipelineStatus.FAILED
            self.result.add_error(str(e))
            raise

    def _validate_inputs(self, data_path: Path) -> None:
        """Validate pipeline inputs"""
        if not data_path.exists():
            raise FileNotFoundError(f"Input file not found: {data_path}")

        if not data_path.suffix.lower() == '.csv':
            raise ValueError(f"Expected CSV file, got: {data_path.suffix}")

        logger.info(f"Input validation passed: {data_path}")

    async def _load_and_process_data(self, data_path: Path) -> pd.DataFrame:
        """Load and process input data"""
        try:
            self._update_stage(PipelineStage.DATA_LOADING)
            start_time = time.perf_counter()

            # Process data using streaming processor
            processed_data = await self.streaming_processor.process_stream(
                data_path, self.feature_engineer
            )

            # Update statistics
            self.stats.loading_time = time.perf_counter() - start_time
            self.stats.total_rows = len(processed_data)

            logger.info(f"Data loaded and processed: {len(processed_data)} rows")
            return processed_data

        except Exception as e:
            raise DataProcessingError(f"Failed to load data: {e}")

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from processed data"""
        try:
            self._update_stage(PipelineStage.FEATURE_ENGINEERING)
            start_time = time.perf_counter()

            # Get feature columns
            feature_names = self.feature_engineer.get_feature_names()

            # Validate features exist
            missing_features = [f for f in feature_names if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Extract features
            features = data[feature_names].values

            # Handle missing values
            if np.any(np.isnan(features)):
                if self.config.features.handle_missing == "drop":
                    # Drop rows with NaN values
                    valid_mask = ~np.isnan(features).any(axis=1)
                    features = features[valid_mask]
                    logger.warning(f"Dropped {np.sum(~valid_mask)} rows with missing features")
                else:
                    raise ValueError("Feature matrix contains NaN values")

            # Update statistics
            self.stats.feature_time = time.perf_counter() - start_time
            self.stats.processed_rows = len(features)

            logger.info(f"Features extracted: {features.shape}")
            return features

        except Exception as e:
            raise DataProcessingError(f"Feature extraction failed: {e}")

    async def _train_or_load_model(self, features: np.ndarray) -> tuple:
        """Train new model or load existing model"""
        try:
            self._update_stage(PipelineStage.MODEL_TRAINING)
            start_time = time.perf_counter()

            # Check if we should load existing model
            if self.config.persistence.model_path and self.config.persistence.model_path.exists():
                logger.info("Loading existing model...")
                model_data = await self.persistence.load_model(self.config.persistence.model_path)
                model = model_data['model']
                scaler = model_data['scaler']

                # Apply scaling
                features_scaled = scaler.transform(features)

            else:
                logger.info("Training new model...")
                # Train new model
                model, scaler, training_metrics = await self.trainer.train(features)

                # Apply scaling
                features_scaled = scaler.transform(features)

                # Update statistics
                self.stats.training_time = time.perf_counter() - start_time
                self.stats.convergence_iterations = training_metrics.get('n_iter', 0)
                self.stats.log_likelihood = training_metrics.get('log_likelihood', 0.0)
                self.stats.aic = training_metrics.get('aic')
                self.stats.bic = training_metrics.get('bic')

                # Save model if configured
                if self.config.persistence.save_model:
                    await self._save_model(model, scaler)

            # Predict states
            self._update_stage(PipelineStage.MODEL_INFERENCE)
            start_time = time.perf_counter()

            states = model.predict(features_scaled)

            # Apply lookahead bias prevention if configured
            if hasattr(self.config, 'prevent_lookahead') and self.config.prevent_lookahead:
                states = np.roll(states, 1)
                states[0] = states[1]

            self.stats.inference_time = time.perf_counter() - start_time

            logger.info(f"Model training/inference completed. States: {np.unique(states)}")
            return model, states

        except Exception as e:
            raise ModelTrainingError(f"Model training/inference failed: {e}")

    async def _save_model(self, model, scaler) -> None:
        """Save trained model"""
        try:
            model_path = self.config.persistence.model_path
            if not model_path:
                # Generate default path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = Path(f"hmm_model_{timestamp}.pkl")

            await self.persistence.save_model(model, scaler, model_path)
            self.result.add_output_file("model", model_path)

            logger.info(f"Model saved to: {model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            # Don't fail pipeline for model save issues

    def _save_results(self, data: pd.DataFrame, states: np.ndarray, model) -> None:
        """Save pipeline results"""
        try:
            self._update_stage(PipelineStage.RESULTS_SAVING)

            # Add states to data
            result_data = data.copy()
            result_data['state'] = states

            # Save results
            if self.config.persistence.save_results:
                results_path = self.config.persistence.results_path
                if not results_path:
                    # Generate default path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = Path(f"hmm_results_{timestamp}.csv")

                result_data.to_csv(results_path, index=True)
                self.result.add_output_file("results", results_path)
                logger.info(f"Results saved to: {results_path}")

            # Store in result object
            self.result.processed_data = result_data
            self.result.states = states
            self.result.model = model

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Don't fail pipeline for result save issues

    async def _run_backtesting(self, data: pd.DataFrame, states: np.ndarray) -> None:
        """Run backtesting if enabled"""
        try:
            self._update_stage(PipelineStage.BACKTESTING)

            # Generate trading signals
            positions = self.strategy_engine.generate_positions(data, states)

            # Calculate returns
            returns = self.strategy_engine.calculate_returns(data, positions)

            # Analyze performance
            performance_metrics = self.performance_analyzer.analyze(returns)

            # Store results
            self.result.backtest_results = returns
            self.result.performance_metrics = performance_metrics

            # Save backtesting results
            if self.config.backtesting.save_trades:
                trades_path = Path(f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                returns.to_csv(trades_path)
                self.result.add_output_file("backtest_trades", trades_path)

            logger.info(f"Backtesting completed. Sharpe: {performance_metrics.get('sharpe_ratio', 'N/A')}")

        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            # Don't fail pipeline for backtesting issues

    def _generate_reports(self) -> None:
        """Generate pipeline reports"""
        try:
            self._update_stage(PipelineStage.VISUALIZATION)

            # Add execution summary
            self.result.metadata.update({
                'pipeline_config': self.config.to_dict(),
                'processing_stats': self.stats.to_dict(),
                'execution_timestamp': datetime.now().isoformat()
            })

            logger.info("Pipeline reports generated")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    def _complete_pipeline(self) -> None:
        """Complete pipeline execution"""
        self._update_stage(PipelineStage.COMPLETED)
        self.result.status = PipelineStatus.COMPLETED

        execution_time = self._end_timing()
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")

    @classmethod
    def from_config_file(cls, config_path: Path) -> 'HMMPipeline':
        """Create pipeline from configuration file"""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = PipelineConfig.from_dict(config_dict)
        return cls(config)

    @classmethod
    def from_args(cls, args) -> 'HMMPipeline':
        """Create pipeline from command line arguments (backward compatibility)"""
        # Convert legacy CLI args to new config format
        config = cls._convert_args_to_config(args)
        return cls(config)

    @staticmethod
    def _convert_args_to_config(args) -> PipelineConfig:
        """Convert legacy CLI arguments to PipelineConfig"""
        from .pipeline_types import (
            FeatureConfig, TrainingConfig, PersistenceConfig,
            StreamingConfig, BacktestConfig
        )

        # Create feature config (mimic main.py defaults)
        features = FeatureConfig()

        # Create training config
        training = TrainingConfig(
            n_states=args.n_states,
            n_iter=args.max_iter,
            random_state=42,
            verbose=True
        )

        # Create persistence config
        persistence = PersistenceConfig(
            model_path=Path(args.model_path) if args.model_path else None,
            save_model=bool(args.model_out),
            model_path_save=Path(args.model_out) if args.model_out else None
        )

        # Create streaming config
        streaming = StreamingConfig(
            chunk_size=args.chunksize,
            show_progress=True
        )

        # Create backtesting config if enabled
        backtesting = None
        if args.backtest:
            backtesting = BacktestConfig()

        # Create pipeline config
        config = PipelineConfig(
            features=features,
            training=training,
            persistence=persistence,
            streaming=streaming,
            backtesting=backtesting,
            input_path=Path(args.csv) if args.csv else None
        )

        # Add legacy attributes
        config.prevent_lookahead = args.prevent_lookahead if hasattr(args, 'prevent_lookahead') else False
        config.plot = args.plot if hasattr(args, 'plot') else False

        return config