#!/usr/bin/env python3
"""
Command-line interface parser for benchmark suite.

Defines and parses all command-line arguments for benchmark configuration.
"""

import argparse
from pathlib import Path

from .executor_factory import get_available_frameworks


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Framework Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single framework benchmark
  python main.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct --framework tinygrad

  # Compare multiple PyTorch variants
  python main.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct \\
    --framework pytorch-unoptimized pytorch-inductor pytorch-eager

  # Full comparison with custom parameters
  python main.py --model-id llama3-1b --model-path ~/models/llama3-1b-instruct \\
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
        choices=get_available_frameworks(),
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

    return parser
