"""
Model Training Module

This module provides Hidden Markov Model training and inference functionality for the HMM futures
analysis project, including feature scaling, convergence monitoring, numerical stability,
state inference, and lookahead bias prevention.
"""

from .hmm_trainer import (
    HMMTrainingResult,
    train_model,
    predict_states,
    evaluate_model,
    get_hmm_model_info,
    validate_features_for_hmm,
    validate_hmm_config,
    create_feature_scaler,
    add_numerical_stability_epsilon,
    train_single_hmm_model
)

from .inference_engine import (
    InferenceResult,
    LaggedInferenceResult,
    predict_states_comprehensive,
    get_lagged_states,
    predict_states_with_lag,
    analyze_state_stability,
    create_inference_dataframe,
    validate_inference_inputs
)

from .model_persistence import (
    ModelMetadata,
    save_model,
    load_model,
    list_saved_models,
    delete_model,
    copy_model,
    get_model_info,
    generate_model_hash,
    get_library_versions
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