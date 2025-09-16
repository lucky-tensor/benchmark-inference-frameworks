#!/usr/bin/env python3
"""
Framework executor factory for creating framework-specific executors.

This module provides a centralized factory function for creating executors
from their respective framework directories.
"""

from benchmark_classes import FrameworkExecutor


def create_executor(framework_name: str) -> FrameworkExecutor:
    """Create an executor for the given framework name."""

    if framework_name == "tinygrad":
        from frameworks.tinygrad.executor import TinyGradExecutor
        return TinyGradExecutor()

    if framework_name == "pytorch-unoptimized":
        from frameworks.pytorch.executor import PyTorchUnoptimizedExecutor
        return PyTorchUnoptimizedExecutor()

    if framework_name == "pytorch-inductor":
        from frameworks.pytorch.executor import PyTorchInductorExecutor
        return PyTorchInductorExecutor()

    if framework_name == "pytorch-eager":
        from frameworks.pytorch.executor import PyTorchEagerExecutor
        return PyTorchEagerExecutor()

    if framework_name == "pytorch-aot_eager":
        from frameworks.pytorch.executor import PyTorchAOTEagerExecutor
        return PyTorchAOTEagerExecutor()

    available = [
        "tinygrad",
        "pytorch-unoptimized",
        "pytorch-inductor",
        "pytorch-eager",
        "pytorch-aot_eager"
    ]
    raise ValueError(f"Unknown framework: {framework_name}. Available: {', '.join(available)}")
