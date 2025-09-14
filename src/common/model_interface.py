"""Abstract Base Classes for Model Interface

This module defines the common interfaces that all framework-specific model
implementations must follow, enabling the benchmarking and demo systems to
work with any model implementation uniformly.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelMetrics:
    """Standard metrics collected from model operations."""

    # Loading metrics
    model_load_time: float = 0.0
    tokenizer_load_time: float = 0.0

    # Inference metrics
    first_token_time: float = 0.0
    tokens_per_second: float = 0.0
    total_inference_time: float = 0.0
    tokens_generated: int = 0

    # Memory metrics
    peak_memory_gb: float = 0.0
    model_memory_gb: float = 0.0

    # Framework-specific metrics
    framework_specific: dict[str, Any] = None

    def __post_init__(self):
        if self.framework_specific is None:
            self.framework_specific = {}


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    seed: int | None = None

    # Framework-specific options
    framework_options: dict[str, Any] = None

    def __post_init__(self):
        if self.framework_options is None:
            self.framework_options = {}


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers across all frameworks."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text string
        """

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Size of the vocabulary
        """

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """Padding token ID."""


class BaseModel(ABC):
    """Abstract base class for all model implementations across frameworks."""

    def __init__(self, model_path: str | Path, **kwargs):
        """Initialize model.

        Args:
            model_path: Path to model weights/config
            **kwargs: Framework-specific options
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.framework_options = kwargs
        self._metrics = ModelMetrics()

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Name of the framework (e.g., 'tinygrad', 'pytorch', 'hybrid')."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Type of model (e.g., 'llama', 'gpt2')."""

    @property
    @abstractmethod
    def model_size(self) -> str:
        """Size identifier (e.g., '1B', '8B', '124M')."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from disk/weights.

        This should handle:
        - Loading model weights
        - Setting up device placement
        - Framework-specific optimizations
        - Updating load time metrics
        """

    @abstractmethod
    def load_tokenizer(self) -> BaseTokenizer:
        """Load and return the tokenizer.

        Returns:
            Tokenizer instance for this model
        """

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        """Generate text from prompt with streaming.

        Args:
            prompt: Input prompt text
            config: Generation configuration

        Yields:
            Generated tokens as strings

        Note:
            Should update metrics during generation
        """

    @abstractmethod
    def generate_single(self, prompt: str, config: GenerationConfig) -> str:
        """Generate complete text from prompt.

        Args:
            prompt: Input prompt text
            config: Generation configuration

        Returns:
            Complete generated text
        """

    @abstractmethod
    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage.

        Returns:
            Dictionary with memory usage in GB
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (memory, cache, etc.)."""

    def get_metrics(self) -> ModelMetrics:
        """Get current metrics.

        Returns:
            Current model metrics
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics to default values."""
        self._metrics = ModelMetrics()


class BaseLLaMAModel(BaseModel):
    """Abstract base class for LLaMA model implementations."""

    @property
    def model_type(self) -> str:
        return "llama"

    @abstractmethod
    def get_model_config(self) -> dict[str, Any]:
        """Get model architecture configuration.

        Returns:
            Dictionary with model config (dim, n_layers, n_heads, etc.)
        """


class BaseGPTModel(BaseModel):
    """Abstract base class for GPT model implementations."""

    @property
    def model_type(self) -> str:
        return "gpt2"

    @abstractmethod
    def get_model_config(self) -> dict[str, Any]:
        """Get model architecture configuration.

        Returns:
            Dictionary with model config (dim, n_layers, n_heads, etc.)
        """


class ModelFactory(ABC):
    """Abstract factory for creating framework-specific models."""

    @abstractmethod
    def create_llama_model(self, model_path: str | Path, model_size: str, **kwargs) -> BaseLLaMAModel:
        """Create LLaMA model for this framework.

        Args:
            model_path: Path to model weights
            model_size: Model size identifier
            **kwargs: Framework-specific options

        Returns:
            LLaMA model instance
        """

    @abstractmethod
    def create_gpt_model(self, model_path: str | Path, model_size: str, **kwargs) -> BaseGPTModel:
        """Create GPT model for this framework.

        Args:
            model_path: Path to model weights
            model_size: Model size identifier
            **kwargs: Framework-specific options

        Returns:
            GPT model instance
        """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Name of this framework."""

    @abstractmethod
    def list_supported_models(self) -> dict[str, list[str]]:
        """List supported models and sizes.

        Returns:
            Dictionary mapping model types to supported sizes
        """


# Registry for framework factories
_framework_registry: dict[str, ModelFactory] = {}


def register_framework(name: str, factory: ModelFactory) -> None:
    """Register a framework factory.

    Args:
        name: Framework name
        factory: Factory instance for this framework
    """
    _framework_registry[name] = factory


def get_framework_factory(name: str) -> ModelFactory:
    """Get factory for a framework.

    Args:
        name: Framework name

    Returns:
        Factory instance

    Raises:
        ValueError: If framework not registered
    """
    if name not in _framework_registry:
        raise ValueError(f"Framework '{name}' not registered. Available: {list(_framework_registry.keys())}")
    return _framework_registry[name]


def list_available_frameworks() -> list[str]:
    """List all registered frameworks.

    Returns:
        List of framework names
    """
    return list(_framework_registry.keys())


def create_model(framework: str, model_type: str, model_path: str | Path, model_size: str, **kwargs) -> BaseModel:
    """Create a model using the appropriate framework.

    Args:
        framework: Framework name
        model_type: Model type ('llama', 'gpt2')
        model_path: Path to model weights
        model_size: Model size identifier
        **kwargs: Framework-specific options

    Returns:
        Model instance

    Raises:
        ValueError: If framework or model type not supported
    """
    factory = get_framework_factory(framework)

    if model_type == "llama":
        return factory.create_llama_model(model_path, model_size, **kwargs)
    if model_type == "gpt2" or model_type == "gpt":
        return factory.create_gpt_model(model_path, model_size, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")


# Export main interfaces
__all__ = [
    "BaseGPTModel",
    "BaseLLaMAModel",
    "BaseModel",
    "BaseTokenizer",
    "GenerationConfig",
    "ModelFactory",
    "ModelMetrics",
    "create_model",
    "get_framework_factory",
    "list_available_frameworks",
    "register_framework",
]
