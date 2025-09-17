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

    def run_multi_token_generation(self, bench_run: "BenchRun", input_data: Any, start_pos: int, num_tokens: int = 5) -> dict[str, float]:
        """
        Run multi-token generation to measure time to first token.

        Returns dict with timing metrics:
        - first_token_ms: Time to generate first token
        - avg_token_ms: Average time per token across all tokens
        """
        import time

        first_token_time = None
        token_times = []

        for i in range(num_tokens):
            token_start = time.perf_counter()
            _ = self.run_inference(bench_run, input_data, start_pos + i)
            token_time = time.perf_counter() - token_start
            token_times.append(token_time)

            if first_token_time is None:
                first_token_time = token_time

        return {
            "first_token_ms": first_token_time * 1000 if first_token_time else 0,
            "avg_token_ms": (sum(token_times) / len(token_times)) * 1000 if token_times else 0,
            "all_token_times": [t * 1000 for t in token_times]
        }

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
