#!/usr/bin/env python3
"""
ML Framework Benchmark Suite - Main Entry Point

A clean, extensible benchmarking system for comparing different ML frameworks
with proper separation of concerns and consistent timing/result tracking.

This is the main entry point that orchestrates the entire benchmarking process
using the modular benchmark system.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import create_parser, run_benchmarks


def main():
    """Main entry point for the benchmark suite."""
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Run the benchmarks
    run_benchmarks(args)


if __name__ == "__main__":
    main()
