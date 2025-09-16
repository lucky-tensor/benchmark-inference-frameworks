#!/usr/bin/env python3
"""
Time tracking utilities for benchmark measurement.

Provides structured timing tracking for all phases of benchmark execution.
"""

import time
from dataclasses import dataclass, field
from typing import Union


@dataclass
class TimeLog:
    """
    Tracks timing for each step of the benchmark process.

    All times are in seconds (float).
    """

    # Core timing steps
    model_loading: Union[float, None] = None
    model_compilation: Union[float, None] = None
    tokenizer_loading: Union[float, None] = None
    memory_clearing: Union[float, None] = None
    cold_start: Union[float, None] = None

    # Inference timing arrays
    iteration_times: list[float] = field(default_factory=list)
    warmup_times: list[float] = field(default_factory=list)
    steady_state_times: list[float] = field(default_factory=list)

    # Cleanup timing
    framework_cleanup: Union[float, None] = None
    memory_cleanup: Union[float, None] = None

    def start_timer(self) -> float:
        """Start a timer and return the start time."""
        return time.time()

    def end_timer(self, start_time: float) -> float:
        """End a timer and return elapsed time in seconds."""
        return time.time() - start_time

    def log_step(self, step_name: str, duration: float) -> None:
        """Log a timed step."""
        if hasattr(self, step_name):
            setattr(self, step_name, duration)
        else:
            print(f"Warning: Unknown step '{step_name}' in TimeLog")

    def add_iteration_time(self, duration: float, is_warmup: bool = False) -> None:
        """Add an iteration time to the appropriate array."""
        self.iteration_times.append(duration)
        if is_warmup:
            self.warmup_times.append(duration)
        else:
            self.steady_state_times.append(duration)

    def get_average_steady_state_time(self) -> float:
        """Get average steady-state inference time."""
        if not self.steady_state_times:
            return 0.0
        return sum(self.steady_state_times) / len(self.steady_state_times)

    def get_total_time(self) -> float:
        """Get total benchmark time."""
        total = 0.0
        for attr_name in [
            "model_loading",
            "model_compilation",
            "tokenizer_loading",
            "memory_clearing",
            "cold_start",
            "framework_cleanup",
            "memory_cleanup",
        ]:
            value = getattr(self, attr_name)
            if value is not None:
                total += value
        return total + sum(self.iteration_times)
