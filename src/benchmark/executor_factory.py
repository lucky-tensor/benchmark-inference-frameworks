#!/usr/bin/env python3
"""
Framework executor factory.

Creates framework-specific executor instances with proper lazy loading.
"""

from .framework_executor import FrameworkExecutor


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


def get_available_frameworks() -> list[str]:
    """Get list of available framework names."""
    return [
        "tinygrad",
        "pytorch-unoptimized",
        "pytorch-inductor",
        "pytorch-eager",
        "pytorch-aot_eager"
    ]
