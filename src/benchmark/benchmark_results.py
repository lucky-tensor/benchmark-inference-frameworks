#!/usr/bin/env python3
"""
Benchmark results storage and metrics.

Contains the structured output data from completed benchmarks.
"""

from dataclasses import dataclass, field


@dataclass
class BenchmarkResults:
    """
    Contains the results of a single benchmark iteration.
    """

    # Performance metrics
    average_latency_ms: float
    peak_throughput_tokens_per_sec: float
    steady_state_throughput_tokens_per_sec: float
    cold_start_latency_ms: float

    # Memory metrics
    model_memory_gb: float
    peak_memory_gb: float

    # Model info
    total_parameters: int
    loaded_parameters: int
    precision: str

    # Additional metrics
    warmup_improvement_factor: float = 1.0
    compilation_success: bool = True
    error_messages: list[str] = field(default_factory=list)
