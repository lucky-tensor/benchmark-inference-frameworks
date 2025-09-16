#!/usr/bin/env python3
"""
Benchmark class architecture for ML framework comparison.

This module provides a structured approach to benchmarking different ML frameworks
with consistent configuration, timing, and result tracking.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimeLog:
    """
    Tracks timing for each step of the benchmark process.

    All times are in seconds (float).
    """

    # Core timing steps
    model_loading: float | None = None
    model_compilation: float | None = None
    tokenizer_loading: float | None = None
    memory_clearing: float | None = None
    cold_start: float | None = None

    # Inference timing arrays
    iteration_times: list[float] = field(default_factory=list)
    warmup_times: list[float] = field(default_factory=list)
    steady_state_times: list[float] = field(default_factory=list)

    # Cleanup timing
    framework_cleanup: float | None = None
    memory_cleanup: float | None = None

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


class FrameworkExecutor(ABC):
    """
    Abstract base class for framework-specific benchmark execution.

    Each framework (TinyGrad, PyTorch variants, etc.) should implement this interface.
    """

    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the framework name (e.g., 'pytorch-inductor')."""

    @abstractmethod
    def load_model(self, bench_run: BenchRun) -> Any:
        """
        Load the model for the given benchmark configuration.

        Should update bench_run.time_log.model_loading.
        Returns the loaded model instance.
        """

    @abstractmethod
    def load_tokenizer(self, bench_run: BenchRun) -> Any:
        """
        Load the tokenizer for the given benchmark configuration.

        Should update bench_run.time_log.tokenizer_loading.
        Returns the loaded tokenizer instance.
        """

    @abstractmethod
    def prepare_input(self, bench_run: BenchRun) -> tuple[Any, int]:
        """
        Prepare the input for inference.

        Returns (input_data, start_position).
        """

    @abstractmethod
    def run_inference(self, bench_run: BenchRun, input_data: Any, start_pos: int) -> Any:
        """
        Run a single inference step.

        Returns the next token or output.
        """

    @abstractmethod
    def get_model_info(self, bench_run: BenchRun) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Should return a dict with keys like 'total_parameters', 'model_memory_gb', etc.
        """

    @abstractmethod
    def cleanup(self, bench_run: BenchRun) -> None:
        """
        Clean up resources after benchmark completion.

        Should update bench_run.time_log.framework_cleanup.
        """

    def compile_model(self, bench_run: BenchRun) -> None:
        """
        Optional model compilation step.

        Default implementation does nothing. Override if framework supports compilation.
        Should update bench_run.time_log.model_compilation.
        """


class BenchmarkSuite:
    """
    Manages a collection of benchmark runs and provides execution coordination.
    """

    def __init__(self):
        self.bench_runs: list[BenchRun] = []
        self.executors: dict[str, FrameworkExecutor] = {}

    def register_executor(self, framework_name: str, executor: FrameworkExecutor) -> None:
        """Register a framework executor."""
        self.executors[framework_name] = executor

    def add_benchmark(self, bench_run: BenchRun) -> None:
        """Add a benchmark run to the suite."""
        self.bench_runs.append(bench_run)

    def create_benchmark(
        self, model_id: str, model_path: str | Path, model_algo: str, framework_name: str, **kwargs
    ) -> BenchRun:
        """Create and add a benchmark run with the given parameters."""
        bench_run = BenchRun(
            model_id=model_id,
            model_path=Path(model_path),
            model_algo=model_algo,
            framework_name=framework_name,
            **kwargs,
        )
        self.add_benchmark(bench_run)
        return bench_run

    def execute_benchmark(self, bench_run: BenchRun) -> BenchmarkResults:
        """Execute a single benchmark run."""
        framework_type = bench_run.get_framework_type()

        if bench_run.framework_name not in self.executors:
            raise ValueError(f"No executor registered for framework: {bench_run.framework_name}")

        executor = self.executors[bench_run.framework_name]

        try:
            print(f"\n🚀 Running {bench_run.framework_name} Benchmark")
            print("=" * 60)
            print(f"🔧 FRAMEWORK VERIFICATION: Using {bench_run.framework_name} executor")
            print(f"🔧 EXECUTOR CLASS: {executor.__class__.__name__} from {executor.__class__.__module__}")
            print(f"Model: {bench_run.model_id}")
            print(f"Algorithm: {bench_run.model_algo}")
            print(f"Iterations: {bench_run.iterations}")

            # Load model
            print("Loading model...")
            start_time = bench_run.time_log.start_timer()
            bench_run.model_instance = executor.load_model(bench_run)
            bench_run.time_log.model_loading = bench_run.time_log.end_timer(start_time)

            # Compile model (if supported)
            if hasattr(executor, "compile_model") and callable(executor.compile_model):
                start_time = bench_run.time_log.start_timer()
                executor.compile_model(bench_run)
                bench_run.time_log.model_compilation = bench_run.time_log.end_timer(start_time)

            # Load tokenizer
            print("Loading tokenizer...")
            start_time = bench_run.time_log.start_timer()
            bench_run.tokenizer_instance = executor.load_tokenizer(bench_run)
            bench_run.time_log.tokenizer_loading = bench_run.time_log.end_timer(start_time)

            # Get model info
            model_info = executor.get_model_info(bench_run)
            print(f"Model: {model_info.get('total_parameters', 0):,} parameters")
            print(f"Memory: {model_info.get('model_memory_gb', 0):.2f} GB")

            # Prepare input
            input_data, start_pos = executor.prepare_input(bench_run)

            # Cold start measurement
            print("\n🥶 Cold Start Measurement")
            print("=" * 40)
            start_time = bench_run.time_log.start_timer()
            cold_start_result = executor.run_inference(bench_run, input_data, start_pos)
            cold_start_time = bench_run.time_log.end_timer(start_time)
            bench_run.time_log.cold_start = cold_start_time

            cold_start_throughput = 1.0 / cold_start_time if cold_start_time > 0 else 0
            print(f"❄️  Cold start: {cold_start_time * 1000:.2f}ms, {cold_start_throughput:.1f} tok/s")

            # Steady-state benchmark
            print(f"\n🔥 Steady-State Benchmark ({bench_run.iterations} iterations)")
            print("=" * 50)

            iteration_times = []
            for i in range(bench_run.iterations):
                start_time = bench_run.time_log.start_timer()
                _ = executor.run_inference(bench_run, input_data, start_pos + i + 1)
                iteration_time = bench_run.time_log.end_timer(start_time)

                is_warmup = i < bench_run.warmup_iterations
                bench_run.time_log.add_iteration_time(iteration_time, is_warmup)
                iteration_times.append(iteration_time)

                throughput = 1.0 / iteration_time if iteration_time > 0 else 0
                status_symbol = "🌡️" if is_warmup else "⚡"
                warmup_text = " (warmup)" if is_warmup else ""
                print(
                    f"{status_symbol} Iteration {i + 1:2d}: {iteration_time * 1000:6.2f}ms, {throughput:6.1f} tok/s{warmup_text}"
                )

            # Calculate results
            steady_state_times = bench_run.time_log.steady_state_times
            if not steady_state_times:
                steady_state_times = iteration_times  # Fallback if no warmup

            avg_latency = sum(steady_state_times) / len(steady_state_times)
            peak_throughput = max(1.0 / t for t in steady_state_times) if steady_state_times else 0
            steady_state_throughput = 1.0 / avg_latency if avg_latency > 0 else 0
            warmup_improvement = steady_state_throughput / cold_start_throughput if cold_start_throughput > 0 else 1.0

            # Create benchmark results
            bench_run.results = BenchmarkResults(
                average_latency_ms=avg_latency * 1000,
                peak_throughput_tokens_per_sec=peak_throughput,
                steady_state_throughput_tokens_per_sec=steady_state_throughput,
                cold_start_latency_ms=cold_start_time * 1000,
                model_memory_gb=model_info.get("model_memory_gb", 0),
                peak_memory_gb=model_info.get("peak_memory_gb", model_info.get("model_memory_gb", 0)),
                total_parameters=model_info.get("total_parameters", 0),
                loaded_parameters=model_info.get("loaded_parameters", model_info.get("total_parameters", 0)),
                precision=model_info.get("precision", bench_run.precision),
                warmup_improvement_factor=warmup_improvement,
                compilation_success=True,
            )

            # Cleanup
            start_time = bench_run.time_log.start_timer()
            executor.cleanup(bench_run)
            bench_run.time_log.framework_cleanup = bench_run.time_log.end_timer(start_time)

            bench_run.is_executed = True

            # Print summary
            print(f"\n🏆 {bench_run.framework_name} Benchmark Results")
            print("=" * 60)
            print(f"Average throughput:    {steady_state_throughput:.1f} tokens/second")
            print(f"Peak throughput:       {peak_throughput:.1f} tokens/second")
            print(f"Steady-state avg:      {steady_state_throughput:.1f} tokens/second")
            print(f"Warmup improvement:   {warmup_improvement:.1f}x faster than cold start")

            return bench_run.results

        except Exception as e:
            bench_run.execution_error = str(e)
            bench_run.is_executed = False
            print(f"❌ {bench_run.framework_name} benchmark failed: {e}")
            raise

    def execute_all(self) -> list[BenchmarkResults]:
        """Execute all benchmark runs in the suite."""
        results = []
        for bench_run in self.bench_runs:
            if not bench_run.is_executed:
                try:
                    result = self.execute_benchmark(bench_run)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to execute {bench_run.framework_name}: {e}")
                    continue
        return results

    def get_executed_runs(self) -> list[BenchRun]:
        """Get all successfully executed benchmark runs."""
        return [run for run in self.bench_runs if run.is_executed and run.results is not None]

    def compare_results(self) -> None:
        """Print a comparison of all executed benchmark results."""
        executed_runs = self.get_executed_runs()
        if len(executed_runs) < 2:
            print("Need at least 2 executed benchmarks for comparison")
            return

        print("\n📊 BENCHMARK COMPARISON")
        print("=" * 80)
        print(f"{'Framework':<20} {'Model':<12} {'Throughput (tok/s)':<18} {'Cold Start (ms)':<15}")
        print("-" * 80)

        for run in executed_runs:
            if run.results:
                print(
                    f"{run.framework_name:<20} {run.model_id:<12} "
                    f"{run.results.steady_state_throughput_tokens_per_sec:<18.1f} "
                    f"{run.results.cold_start_latency_ms:<15.1f}"
                )
