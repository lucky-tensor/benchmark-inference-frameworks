#!/usr/bin/env python3
"""
Benchmark suite orchestration and execution.

Manages collections of benchmark runs and coordinates their execution.
"""

from pathlib import Path

from .benchmark_results import BenchmarkResults
from .benchmark_run import BenchRun
from .framework_executor import FrameworkExecutor


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

            # Time to first token and cold start measurement
            print("\n🥶 Cold Start & TTFT Measurement")
            print("=" * 40)

            # Run multi-token generation to capture TTFT
            ttft_metrics = executor.run_multi_token_generation(bench_run, input_data, start_pos, num_tokens=5)
            time_to_first_token_ms = ttft_metrics["first_token_ms"]
            cold_start_time = time_to_first_token_ms / 1000  # Use TTFT as cold start for compatibility
            bench_run.time_log.cold_start = cold_start_time

            cold_start_throughput = 1.0 / cold_start_time if cold_start_time > 0 else 0
            print(f"❄️  Cold start: {cold_start_time * 1000:.2f}ms, {cold_start_throughput:.1f} tok/s")
            print(f"⚡ Time to first token: {time_to_first_token_ms:.2f}ms")

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
                    f"{status_symbol} Iteration {i + 1:2d}: "
                    f"{iteration_time * 1000:6.2f}ms, {throughput:6.1f} tok/s{warmup_text}"
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
                time_to_first_token_ms=time_to_first_token_ms,
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
            print(f"Time to first token:   {time_to_first_token_ms:.2f}ms")
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
        print("=" * 95)
        print(f"{'Framework':<20} {'Model':<12} {'Throughput (tok/s)':<18} {'TTFT (ms)':<12} {'Cold Start (ms)':<15}")
        print("-" * 95)

        for run in executed_runs:
            if run.results:
                print(
                    f"{run.framework_name:<20} {run.model_id:<12} "
                    f"{run.results.steady_state_throughput_tokens_per_sec:<18.1f} "
                    f"{run.results.time_to_first_token_ms:<12.1f} "
                    f"{run.results.cold_start_latency_ms:<15.1f}"
                )
