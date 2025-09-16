"""
Separate backend implementations for different ML frameworks.

This module provides clean separation between TinyGrad and PyTorch implementations,
allowing each framework to be developed and tested independently.
"""

from .pytorch_backend import MODEL_CONFIGS as PYTORCH_CONFIGS, PyTorchLLaMA, PyTorchTokenizer
from .tinygrad_backend import (
    get_tinygrad_model,
    get_tinygrad_tokenizer,
    run_tinygrad_benchmark,
    run_tinygrad_inference,
)

__all__ = [
    "PYTORCH_CONFIGS",
    "PyTorchLLaMA",
    "PyTorchTokenizer",
    "get_tinygrad_model",
    "get_tinygrad_tokenizer",
    "run_tinygrad_benchmark",
    "run_tinygrad_inference",
]
