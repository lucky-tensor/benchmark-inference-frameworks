#!/usr/bin/env python3
"""
Benchmark run configuration and context.

Defines the BenchRun class that encapsulates all benchmark execution parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_results import BenchmarkResults
from .time_tracking import TimeLog


@dataclass
class BenchRun:
    """
    Configuration and execution context for a benchmark run.

    This class encapsulates all the information needed to run a benchmark
    and stores the results and timing information.
    """

    # Model identification
    model_id: str  # e.g., "llama3-1b", "gpt2-small"
    model_path: Path
    model_algo: str  # e.g., "llama", "gpt", "bert"

    # Framework specification
    framework_name: str  # e.g., "tinygrad", "pytorch-unoptimized", "pytorch-inductor"
    framework_version: str | None = None

    # Benchmark configuration
    iterations: int = 20
    warmup_iterations: int = 2
    device: str = "auto"  # "auto", "cuda", "cpu"
    precision: str = "fp32"  # "fp32", "fp16", "mixed"

    # Generation parameters (for language models)
    temperature: float = 0.95
    top_k: int = 0
    top_p: float = 0.0
    alpha_f: float = 0.0  # Frequency penalty
    alpha_p: float = 0.0  # Presence penalty

    # Framework-specific options
    framework_options: dict[str, Any] = field(default_factory=dict)

    # Results and timing (populated during execution)
    time_log: TimeLog = field(default_factory=TimeLog)
    results: BenchmarkResults | None = None
    model_instance: Any | None = None
    tokenizer_instance: Any | None = None

    # Execution state
    is_executed: bool = False
    execution_error: str | None = None

    def get_framework_type(self) -> str:
        """Get the base framework type (e.g., 'pytorch' from 'pytorch-inductor')."""
        return self.framework_name.split("-")[0]

    def get_framework_variant(self) -> str | None:
        """Get the framework variant (e.g., 'inductor' from 'pytorch-inductor')."""
        parts = self.framework_name.split("-", 1)
        return parts[1] if len(parts) > 1 else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "model_algo": self.model_algo,
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "device": self.device,
            "precision": self.precision,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "alpha_f": self.alpha_f,
            "alpha_p": self.alpha_p,
            "framework_options": self.framework_options,
            "is_executed": self.is_executed,
            "execution_error": self.execution_error,
        }
