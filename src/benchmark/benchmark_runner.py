#!/usr/bin/env python3
"""
Benchmark runner and orchestration logic.

Handles benchmark execution, cleanup, and result reporting.
"""

import gc
import sys
import time

from .benchmark_suite import BenchmarkSuite
from .executor_factory import create_executor


def run_benchmarks(args) -> None:
    """
    Execute benchmarks based on parsed command-line arguments.

    Args:
        args: Parsed command-line arguments from argparse
    """
    # Validate model path
    if not args.model_path.exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Create benchmark suite
    suite = BenchmarkSuite()

    # Register executors for requested frameworks
    for framework_name in args.framework:
        try:
            executor = create_executor(framework_name)
            suite.register_executor(framework_name, executor)
            print(f"✅ Registered executor for {framework_name}")
        except Exception as e:
            print(f"❌ Failed to register {framework_name}: {e}")
            sys.exit(1)

    # Create benchmark runs
    for framework_name in args.framework:
        suite.create_benchmark(
            model_id=args.model_id,
            model_path=args.model_path,
            model_algo=args.model_algo,
            framework_name=framework_name,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            device=args.device,
            precision=args.precision,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            alpha_f=args.alpha_f,
            alpha_p=args.alpha_p,
        )
        print(f"📋 Created benchmark for {framework_name}")

    print(f"\n🚀 Starting benchmark suite with {len(suite.bench_runs)} configurations...")

    # Execute all benchmarks
    try:
        results = suite.execute_all()
        print(f"\n✅ Completed {len(results)} successful benchmarks")

        # Show comparison if multiple frameworks
        if len(results) > 1:
            suite.compare_results()

        # Final cleanup
        perform_final_cleanup()

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def perform_final_cleanup() -> None:
    """Perform final system and framework-specific cleanup."""
    print("\n🔄 Final system cleanup...")

    # System garbage collection
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)

    # Framework-specific cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

    print("✅ Final cleanup completed")
