"""TinyGrad Framework Implementation

This package contains TinyGrad-specific implementations of models with
TinyGrad-optimized loading, memory management, and inference.
"""

# Auto-register this framework
from common.model_interface import register_framework

from .factory import TinyGradFactory
from .llama_model import TinyGradLLaMAModel
from .tokenizer import TinyGradTokenizer

register_framework("tinygrad", TinyGradFactory())

__all__ = ["TinyGradFactory", "TinyGradLLaMAModel", "TinyGradTokenizer"]
