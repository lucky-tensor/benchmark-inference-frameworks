#!/usr/bin/env python3
"""
Next-generation benchmark system using class-based architecture.

This script provides a clean, extensible benchmarking system for comparing
different ML frameworks with proper separation of concerns and consistent
timing/result tracking.
"""

import argparse
import sys
from pathlib import Path

from benchmark_classes import BenchmarkSuite
from framework_factory import create_executor


def main():
    parser = argparse.ArgumentParser(
        description="ML Framework Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single framework benchmark
  python benchmark_v2.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct --framework tinygrad

  # Compare multiple PyTorch variants
  python benchmark_v2.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct \\
    --framework pytorch-unoptimized pytorch-inductor pytorch-eager

  # Full comparison with custom parameters
  python benchmark_v2.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct \\
    --framework tinygrad pytorch-inductor --iterations 10 --temperature 0.8

Available frameworks:
  - tinygrad                : TinyGrad with JIT compilation
  - pytorch-unoptimized     : PyTorch without compilation
  - pytorch-inductor        : PyTorch with Inductor backend (fastest)
  - pytorch-eager           : PyTorch with Eager compilation
  - pytorch-aot_eager       : PyTorch with AOT Eager compilation
""",
    )

    # Model configuration
    parser.add_argument("--model-id", required=True, help="Model identifier (e.g., 'llama3-1b', 'gpt2-small')")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model files")
    parser.add_argument("--model-algo", default="llama", help="Model algorithm type (default: llama)")

    # Framework selection
    parser.add_argument(
        "--framework",
        nargs="+",
        required=True,
        choices=["tinygrad", "pytorch-unoptimized", "pytorch-inductor", "pytorch-eager", "pytorch-aot_eager"],
        help="Framework(s) to benchmark",
    )

    # Benchmark parameters
    parser.add_argument("--iterations", type=int, default=20, help="Number of inference iterations (default: 20)")
    parser.add_argument("--warmup-iterations", type=int, default=2, help="Number of warmup iterations (default: 2)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.95, help="Sampling temperature (default: 0.95)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (default: 0, disabled)")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling (default: 0.0, disabled)")
    parser.add_argument("--alpha-f", type=float, default=0.0, help="Frequency penalty (default: 0.0)")
    parser.add_argument("--alpha-p", type=float, default=0.0, help="Presence penalty (default: 0.0)")

    # Device and precision
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16", "mixed"], help="Model precision (default: fp32)"
    )

    args = parser.parse_args()

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
        bench_run = suite.create_benchmark(
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

        # Additional post-benchmark cleanup
        print("\n🔄 Final system cleanup...")
        import gc
        import time

        for i in range(3):
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

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
