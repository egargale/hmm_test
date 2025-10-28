"""
Pipeline module for unified HMM processing workflows.

This module provides high-level pipeline interfaces that combine
data processing, feature engineering, model training, and results
persistence into unified workflows.
"""

from .hmm_pipeline import HMMPipeline, PipelineConfig, PipelineResult
from .pipeline_types import (
    BacktestConfig,
    FeatureConfig,
    PersistenceConfig,
    PipelineStage,
    PipelineStatus,
    ProcessingMode,
    StreamingConfig,
    TrainingConfig,
)

__all__ = [
    "HMMPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "PipelineStatus",
    "ProcessingMode",
    "FeatureConfig",
    "TrainingConfig",
    "PersistenceConfig",
    "StreamingConfig",
    "BacktestConfig",
]
