#!/usr/bin/env python3
"""
Main entry point for the tinygrad LLaMA implementation.
Provides simplified command-line interface for running models.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llama.llama3 import main as llama_main


def main():
    """Main entry point with simplified arguments"""
    parser = argparse.ArgumentParser(
        description="Run LLaMA models with tinygrad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py --model llama3-1b
  uv run main.py --model llama3-8b --quantize int8
  uv run main.py --model llama3-70b --api --port 8080
        """
    )

    # Model selection (simplified)
    parser.add_argument(
        "--model",
        choices=["llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b"],
        default="llama3-1b",
        help="Model to run (default: llama3-1b)"
    )

    # Core options
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
    parser.add_argument("--download", action="store_true", help="Force download of model")

    # Interface options
    parser.add_argument("--api", action="store_true", help="Run web API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
    parser.add_argument("--port", type=int, default=7776, help="Web server port")

    # Generation options
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Debug/benchmark options
    parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--profile", action="store_true", help="Output profile data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Map simplified model names to size codes
    model_mapping = {
        "llama3-1b": "1B",
        "llama3-8b": "8B",
        "llama3-70b": "70B",
        "llama3-405b": "405B"
    }

    # Convert to the format expected by the original llama3.py
    sys.argv = ["llama3.py"]
    sys.argv.extend(["--size", model_mapping[args.model]])

    if args.quantize:
        sys.argv.extend(["--quantize", args.quantize])
    if args.shard > 1:
        sys.argv.extend(["--shard", str(args.shard)])
    if args.download:
        sys.argv.append("--download_model")
    if args.api:
        sys.argv.append("--api")
    if args.host != "0.0.0.0":
        sys.argv.extend(["--host", args.host])
    if args.port != 7776:
        sys.argv.extend(["--port", str(args.port)])
    if args.temperature != 0.85:
        sys.argv.extend(["--temperature", str(args.temperature)])
    if args.seed:
        sys.argv.extend(["--seed", str(args.seed)])
    if args.benchmark:
        sys.argv.append("--benchmark")
    if args.timing:
        sys.argv.append("--timing")
    if args.profile:
        sys.argv.append("--profile")
    if args.debug:
        sys.argv.append("--debug")

    # Call the original main function
    llama_main()


if __name__ == "__main__":
    main()