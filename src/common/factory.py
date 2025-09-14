"""TinyGrad Model Factory

Factory for creating TinyGrad-specific model implementations.
"""

from pathlib import Path

from common.model_interface import BaseGPTModel, BaseLLaMAModel, ModelFactory

from .gpt_model import TinyGradGPTModel
from .llama_model import TinyGradLLaMAModel


class TinyGradFactory(ModelFactory):
    """Factory for TinyGrad model implementations."""

    @property
    def framework_name(self) -> str:
        return "tinygrad"

    def create_llama_model(self, model_path: str | Path, model_size: str, **kwargs) -> BaseLLaMAModel:
        """Create TinyGrad LLaMA model.

        Args:
            model_path: Path to model weights
            model_size: Model size identifier ('1B', '8B', etc.)
            **kwargs: TinyGrad-specific options

        Returns:
            TinyGrad LLaMA model instance
        """
        return TinyGradLLaMAModel(model_path=model_path, model_size=model_size, **kwargs)

    def create_gpt_model(self, model_path: str | Path, model_size: str, **kwargs) -> BaseGPTModel:
        """Create TinyGrad GPT model.

        Args:
            model_path: Path to model weights
            model_size: Model size identifier ('124M', '355M', etc.)
            **kwargs: TinyGrad-specific options

        Returns:
            TinyGrad GPT model instance
        """
        return TinyGradGPTModel(model_path=model_path, model_size=model_size, **kwargs)

    def list_supported_models(self) -> dict[str, list[str]]:
        """List supported models and sizes for TinyGrad.

        Returns:
            Dictionary mapping model types to supported sizes
        """
        return {"llama": ["1B", "8B", "70B", "405B"], "gpt2": ["124M", "355M", "774M", "1558M"]}
