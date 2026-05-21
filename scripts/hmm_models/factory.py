"""
HMM Model Factory Module

Provides factory functions for creating different types of HMM models
with automatic configuration and hyperparameter management.
"""

from typing import Any, Dict, List, Optional

from utils import get_logger

from .base import BaseHMMModel
from .gaussian_hmm import GaussianHMMModel
from .gmm_hmm import GMMHMMModel

logger = get_logger(__name__)


class HMMModelFactory:
    """
    Factory class for creating and configuring HMM models.

    Provides a unified interface for creating different types of HMM models
    with automatic hyperparameter selection and validation.
    """

    # Registry of available model classes
    MODEL_REGISTRY = {
        "gaussian": GaussianHMMModel,
        "gmm": GMMHMMModel,
    }

    # Default hyperparameter configurations
    DEFAULT_CONFIGS = {
        "gaussian": {
            "n_components": 3,
            "covariance_type": "full",
            "random_state": 42,
            "n_iter": 100,
            "tol": 1e-6,
            "verbose": False,
        },
        "gmm": {
            "n_components": 3,
            "n_mix": 2,
            "covariance_type": "full",
            "random_state": 42,
            "n_iter": 100,
            "tol": 1e-6,
            "verbose": False,
        },
    }

    # Data size-based hyperparameter recommendations
    SIZE_BASED_CONFIGS = {
        "small": {  # < 1000 samples
            "gaussian": {"n_components": 2, "n_iter": 50},
            "gmm": {"n_components": 2, "n_mix": 2, "n_iter": 50},
        },
        "medium": {  # 1000-10000 samples
            "gaussian": {"n_components": 3, "n_iter": 100},
            "gmm": {"n_components": 3, "n_mix": 2, "n_iter": 100},
        },
        "large": {  # > 10000 samples
            "gaussian": {"n_components": 4, "n_iter": 200},
            "gmm": {"n_components": 4, "n_mix": 3, "n_iter": 200},
        },
    }

    @classmethod
    def create_model(
        self,
        model_type: str = "gaussian",
        config: Optional[Dict[str, Any]] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        **kwargs,
    ) -> BaseHMMModel:
        """
        Create an HMM model with specified configuration.

        Args:
            model_type: Type of HMM model ('gaussian', 'gmm')
            config: Custom configuration dictionary
            n_samples: Number of training samples (for auto-configuration)
            n_features: Number of features (for auto-configuration)
            **kwargs: Additional parameters for model initialization

        Returns:
            Configured HMM model instance
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {list(self.MODEL_REGISTRY.keys())}"
            )

        logger.info(f"Creating {model_type} HMM model")

        # Start with default configuration
        model_config = self.DEFAULT_CONFIGS[model_type].copy()

        # Apply size-based configuration if sample count is provided
        if n_samples is not None:
            size_category = self._categorize_data_size(n_samples)
            size_config = self.SIZE_BASED_CONFIGS[size_category].get(model_type, {})
            model_config.update(size_config)
            logger.info(f"Applied {size_category} data configuration: {size_config}")

        # Apply custom configuration
        if config:
            model_config.update(config)
            logger.info(f"Applied custom configuration: {config}")

        # Apply explicit kwargs (highest priority)
        if kwargs:
            model_config.update(kwargs)
            logger.info(f"Applied explicit parameters: {kwargs}")

        # Validate configuration
        model_config = self._validate_config(
            model_type, model_config, n_samples, n_features
        )

        # Create model instance
        model_class = self.MODEL_REGISTRY[model_type]
        model = model_class(**model_config)

        logger.info(f"Created {model_type} HMM model: {model}")
        return model

    @classmethod
    def create_model_ensemble(
        self,
        model_types: Optional[List[str]] = None,
        n_components_range: Optional[List[int]] = None,
        config_variations: Optional[Dict[str, List[Any]]] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
    ) -> List[BaseHMMModel]:
        """
        Create an ensemble of HMM models with different configurations.

        Args:
            model_types: List of model types to include
            n_components_range: Range of n_components to try
            config_variations: Dictionary of parameter variations
            n_samples: Number of training samples
            n_features: Number of features

        Returns:
            List of configured HMM models
        """
        if model_types is None:
            model_types = ["gaussian", "gmm"]

        if n_components_range is None:
            n_components_range = [2, 3, 4]

        if config_variations is None:
            config_variations = {
                "covariance_type": ["full", "diag"],
                "n_iter": [100, 200],
            }

        logger.info(f"Creating model ensemble with {len(model_types)} model types")
        logger.info(f"Components range: {n_components_range}")
        logger.info(f"Configuration variations: {config_variations}")

        ensemble = []

        for model_type in model_types:
            for n_components in n_components_range:
                # Base configuration
                base_config = {"n_components": n_components}

                # Generate all combinations of variations
                config_combinations = self._generate_config_combinations(
                    base_config, config_variations
                )

                for config in config_combinations:
                    try:
                        model = self.create_model(
                            model_type=model_type,
                            config=config,
                            n_samples=n_samples,
                            n_features=n_features,
                        )
                        ensemble.append(model)
                    except Exception as e:
                        logger.warning(
                            f"Failed to create {model_type} model with config {config}: {e}"
                        )

        logger.info(f"Created ensemble of {len(ensemble)} models")
        return ensemble

    @classmethod
    def auto_select_model_type(
        self,
        n_samples: int,
        n_features: int,
        data_complexity: Optional[str] = None,
        computational_budget: str = "medium",
    ) -> str:
        """
        Automatically select the best model type based on data characteristics.

        Args:
            n_samples: Number of training samples
            n_features: Number of features
            data_complexity: Expected complexity of data ('simple', 'moderate', 'complex')
            computational_budget: Computational budget ('low', 'medium', 'high')

        Returns:
            Recommended model type
        """
        # Default complexity based on data size
        if data_complexity is None:
            if n_samples < 1000:
                data_complexity = "simple"
            elif n_samples < 10000:
                data_complexity = "moderate"
            else:
                data_complexity = "complex"

        logger.info(
            f"Auto-selecting model for {n_samples} samples, {n_features} features"
        )
        logger.info(
            f"Data complexity: {data_complexity}, Budget: {computational_budget}"
        )

        # Decision matrix
        if computational_budget == "low":
            # Prefer simpler models for low budget
            if data_complexity in ["simple", "moderate"]:
                return "gaussian"
            else:
                return "gaussian"  # Still prefer gaussian due to speed

        elif computational_budget == "medium":
            # Balanced approach
            if data_complexity == "simple":
                return "gaussian"
            elif data_complexity == "moderate":
                return "gaussian"  # Default to gaussian unless specifically needed
            else:
                return "gmm"  # Complex data benefits from GMM

        else:  # high budget
            # Use most appropriate model regardless of complexity
            if data_complexity == "simple":
                return "gaussian"
            else:
                return "gmm"

    @classmethod
    def _categorize_data_size(self, n_samples: int) -> str:
        """Categorize data size for configuration purposes."""
        if n_samples < 1000:
            return "small"
        elif n_samples < 10000:
            return "medium"
        else:
            return "large"

    @classmethod
    def _validate_config(
        self,
        model_type: str,
        config: Dict[str, Any],
        n_samples: Optional[int],
        n_features: Optional[int],
    ) -> Dict[str, Any]:
        """Validate and adjust configuration parameters."""
        validated_config = config.copy()

        # Validate n_components against sample size
        if n_samples is not None and "n_components" in validated_config:
            max_components = min(
                n_samples // 10, 10
            )  # Rule of thumb: at least 10 samples per state
            if validated_config["n_components"] > max_components:
                logger.warning(
                    f"Reducing n_components from {validated_config['n_components']} to {max_components} "
                    f"due to limited sample size ({n_samples})"
                )
                validated_config["n_components"] = max_components

        # Validate n_mix for GMM models
        if model_type == "gmm" and "n_mix" in validated_config:
            if n_samples is not None:
                max_mix = min(n_samples // (validated_config["n_components"] * 10), 5)
                if validated_config["n_mix"] > max_mix:
                    logger.warning(
                        f"Reducing n_mix from {validated_config['n_mix']} to {max_mix} "
                        f"due to limited sample size"
                    )
                    validated_config["n_mix"] = max_mix

        # Validate covariance_type
        if "covariance_type" in validated_config:
            valid_types = ["full", "tied", "diag", "spherical"]
            if validated_config["covariance_type"] not in valid_types:
                logger.warning(
                    f"Invalid covariance_type {validated_config['covariance_type']}, using 'full'"
                )
                validated_config["covariance_type"] = "full"

        # Adjust n_iter based on sample size
        if "n_iter" in validated_config and n_samples is not None:
            if n_samples > 50000:
                # Reduce iterations for very large datasets
                validated_config["n_iter"] = min(validated_config["n_iter"], 100)
                logger.info(
                    f"Reduced n_iter to {validated_config['n_iter']} for large dataset"
                )

        return validated_config

    @classmethod
    def _generate_config_combinations(
        self, base_config: Dict[str, Any], variations: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of configuration variations."""
        import itertools

        # Create lists of values for each parameter
        param_lists = {}
        for param, values in variations.items():
            if isinstance(values, list):
                param_lists[param] = values
            else:
                param_lists[param] = [values]

        # Generate all combinations
        keys = list(param_lists.keys())
        values_lists = list(param_lists.values())
        combinations = list(itertools.product(*values_lists))

        # Create configuration dictionaries
        configs = []
        for combination in combinations:
            config = base_config.copy()
            for i, key in enumerate(keys):
                config[key] = combination[i]
            configs.append(config)

        return configs

    @classmethod
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available model types.

        Returns:
            Dictionary with model information
        """
        models_info = {}

        for model_type, model_class in self.MODEL_REGISTRY.items():
            models_info[model_type] = {
                "class": model_class.__name__,
                "description": model_class.__doc__.strip().split("\n")[0]
                if model_class.__doc__
                else "No description",
                "default_config": self.DEFAULT_CONFIGS[model_type],
                "capabilities": self._get_model_capabilities(model_type),
            }

        return models_info

    @classmethod
    def _get_model_capabilities(self, model_type: str) -> Dict[str, Any]:
        """Get capabilities for a specific model type."""
        capabilities = {
            "gaussian": {
                "emission_type": "Gaussian",
                "multi_modal": False,
                "computational_complexity": "Low",
                "best_for": [
                    "simple_regimes",
                    "unimodal_distributions",
                    "limited_data",
                ],
                "limitations": ["Cannot capture multi-modal distributions"],
            },
            "gmm": {
                "emission_type": "Gaussian Mixture",
                "multi_modal": True,
                "computational_complexity": "Medium to High",
                "best_for": [
                    "complex_regimes",
                    "multi_modal_distributions",
                    "market_microstructure",
                ],
                "limitations": [
                    "Higher computational cost",
                    "More parameters to estimate",
                ],
            },
        }

        return capabilities.get(model_type, {})

    @classmethod
    def create_recommended_config(
        self,
        model_type: str,
        n_samples: int,
        n_features: int,
        target_use_case: str = "general",
    ) -> Dict[str, Any]:
        """
        Create a recommended configuration for a specific use case.

        Args:
            model_type: Type of HMM model
            n_samples: Number of training samples
            n_features: Number of features
            target_use_case: Target use case ('general', 'trading', 'analysis', 'real_time')

        Returns:
            Recommended configuration dictionary
        """
        base_config = self.DEFAULT_CONFIGS[model_type].copy()

        # Apply size-based adjustments
        size_category = self._categorize_data_size(n_samples)
        size_config = self.SIZE_BASED_CONFIGS[size_category].get(model_type, {})
        base_config.update(size_config)

        # Use case specific adjustments
        if target_use_case == "trading":
            # Trading applications often need fewer states for interpretability
            base_config["n_components"] = min(base_config["n_components"], 3)
            if model_type == "gmm":
                base_config["n_mix"] = min(base_config["n_mix"], 2)

        elif target_use_case == "analysis":
            # Analysis can handle more complexity
            if size_category == "large":
                base_config["n_components"] = min(base_config["n_components"] + 1, 6)
                if model_type == "gmm":
                    base_config["n_mix"] = min(base_config["n_mix"] + 1, 4)

        elif target_use_case == "real_time":
            # Real-time applications need speed
            base_config["n_iter"] = min(base_config["n_iter"], 50)
            base_config["tol"] = max(base_config["tol"], 1e-4)

        return self._validate_config(model_type, base_config, n_samples, n_features)


# Convenience functions
def create_gaussian_hmm(**kwargs) -> GaussianHMMModel:
    """Create a Gaussian HMM model with default or custom parameters."""
    return HMMModelFactory.create_model("gaussian", **kwargs)


def create_gmm_hmm(**kwargs) -> GMMHMMModel:
    """Create a GMM HMM model with default or custom parameters."""
    return HMMModelFactory.create_model("gmm", **kwargs)


def auto_create_hmm(
    n_samples: int,
    n_features: int,
    data_complexity: Optional[str] = None,
    computational_budget: str = "medium",
    **kwargs,
) -> BaseHMMModel:
    """
    Automatically create the most appropriate HMM model.

    Args:
        n_samples: Number of training samples
        n_features: Number of features
        data_complexity: Expected data complexity
        computational_budget: Computational budget
        **kwargs: Additional parameters

    Returns:
        Configured HMM model
    """
    model_type = HMMModelFactory.auto_select_model_type(
        n_samples, n_features, data_complexity, computational_budget
    )

    return HMMModelFactory.create_model(
        model_type=model_type, n_samples=n_samples, n_features=n_features, **kwargs
    )
