"""
Model Training Module

This module provides Hidden Markov Model training and inference functionality for the HMM futures
analysis project, including feature scaling, convergence monitoring, numerical stability,
state inference, and lookahead bias prevention.
"""

from .hmm_trainer import (
    HMMTrainingResult,
    add_numerical_stability_epsilon,
    create_feature_scaler,
    evaluate_model,
    get_hmm_model_info,
    predict_states,
    train_model,
    train_single_hmm_model,
    validate_features_for_hmm,
    validate_hmm_config,
)
from .inference_engine import (
    InferenceResult,
    LaggedInferenceResult,
    analyze_state_stability,
    create_inference_dataframe,
    get_lagged_states,
    predict_states_comprehensive,
    predict_states_with_lag,
    validate_inference_inputs,
)
from .model_persistence import (
    ModelMetadata,
    copy_model,
    delete_model,
    generate_model_hash,
    get_library_versions,
    get_model_info,
    list_saved_models,
    load_model,
    save_model,
)

__all__ = [
    'HMMTrainingResult',
    'train_model',
    'predict_states',
    'evaluate_model',
    'get_hmm_model_info',
    'validate_features_for_hmm',
    'validate_hmm_config',
    'create_feature_scaler',
    'add_numerical_stability_epsilon',
    'train_single_hmm_model',
    'InferenceResult',
    'LaggedInferenceResult',
    'predict_states_comprehensive',
    'get_lagged_states',
    'predict_states_with_lag',
    'analyze_state_stability',
    'create_inference_dataframe',
    'validate_inference_inputs',
    'ModelMetadata',
    'save_model',
    'load_model',
    'list_saved_models',
    'delete_model',
    'copy_model',
    'get_model_info',
    'generate_model_hash',
    'get_library_versions'
]
