#!/usr/bin/env python3
"""
Abstract framework executor interface.

Defines the contract that all ML framework implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .benchmark_run import BenchRun


class FrameworkExecutor(ABC):
    """
    Abstract base class for framework-specific benchmark execution.

    Each framework (TinyGrad, PyTorch variants, etc.) should implement this interface.
    """

    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the framework name (e.g., 'pytorch-inductor')."""

    @abstractmethod
    def load_model(self, bench_run: "BenchRun") -> Any:
        """
        Load the model for the given benchmark configuration.

        Should update bench_run.time_log.model_loading.
        Returns the loaded model instance.
        """

    @abstractmethod
    def load_tokenizer(self, bench_run: "BenchRun") -> Any:
        """
        Load the tokenizer for the given benchmark configuration.

        Should update bench_run.time_log.tokenizer_loading.
        Returns the loaded tokenizer instance.
        """

    @abstractmethod
    def prepare_input(self, bench_run: "BenchRun") -> tuple[Any, int]:
        """
        Prepare the input for inference.

        Returns (input_data, start_position).
        """

    @abstractmethod
    def run_inference(self, bench_run: "BenchRun", input_data: Any, start_pos: int) -> Any:
        """
        Run a single inference step.

        Returns the next token or output.
        """

    @abstractmethod
    def get_model_info(self, bench_run: "BenchRun") -> dict[str, Any]:
        """
        Get information about the loaded model.

        Should return a dict with keys like 'total_parameters', 'model_memory_gb', etc.
        """

    @abstractmethod
    def cleanup(self, bench_run: "BenchRun") -> None:
        """
        Clean up resources after benchmark completion.

        Should update bench_run.time_log.framework_cleanup.
        """

    def compile_model(self, bench_run: "BenchRun") -> None:
        """
        Optional model compilation step.

        Default implementation does nothing. Override if framework supports compilation.
        Should update bench_run.time_log.model_compilation.
        """
