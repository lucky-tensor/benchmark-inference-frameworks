"""
Backend implementations for ML frameworks.

TinyGrad implementation has been moved to src/frameworks/tinygrad.
This module now only contains PyTorch backend.
"""

from .pytorch_backend import MODEL_CONFIGS as PYTORCH_CONFIGS, PyTorchLLaMA, PyTorchTokenizer

__all__ = [
    "PYTORCH_CONFIGS",
    "PyTorchLLaMA",
    "PyTorchTokenizer",
]
