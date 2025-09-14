#!/usr/bin/env python3
"""
Main entry point for the TinyGrad LLaMA Inference Engine Benchmark Suite.

This application provides comprehensive benchmarking and interactive demo capabilities
for different ML inference frameworks including TinyGrad, PyTorch, and hybrid implementations.
"""

import argparse
import sys
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import frameworks to auto-register them
import frameworks.tinygrad

from tinygrad import Tensor

from common.model_interface import GenerationConfig, create_model, list_available_frameworks


def main():
    """Main entry point with unified interface for all frameworks and modes."""
    parser = argparse.ArgumentParser(
        description="TinyGrad LLaMA Inference Engine Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive Q&A mode (default)
  uv run python src/main.py
  uv run python src/main.py --framework tinygrad --model llama3-8b

  # Single inference with prompt
  uv run python src/main.py --prompt "What is AI?"
  uv run python src/main.py --framework tinygrad --prompt "Explain quantum computing"

  # Benchmarking mode
  uv run python src/main.py --benchmark
  uv run python src/main.py --framework tinygrad --benchmark --iterations 10

  # Framework comparison (when multiple frameworks are available)
  uv run python src/main.py --benchmark --framework tinygrad pytorch
        """,
    )

    # Framework selection
    available_frameworks = list_available_frameworks()
    parser.add_argument(
        "--framework",
        choices=available_frameworks,
        nargs="+",
        default=["tinygrad"],
        help=f"Framework(s) to use (default: tinygrad). Available: {', '.join(available_frameworks)}",
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b", "gpt2-124m", "gpt2-355m"],
        default="llama3-1b",
        help="Model to run (default: llama3-1b)",
    )

    # Core options
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
    parser.add_argument("--download", action="store_true", help="Force download of model")
    parser.add_argument("--model-path", type=Path, help="Custom path to model files")

    # Generation options
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature for generation")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--prompt", type=str, help="Run single inference with this prompt and exit")
    mode_group.add_argument("--benchmark", action="store_true", help="Run benchmark mode")

    # Benchmark options
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")

    # Debug/profiling options
    parser.add_argument("--timing", action="store_true", help="Print detailed timing per token")
    parser.add_argument("--profile", action="store_true", help="Output profiling data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        Tensor.manual_seed(args.seed)
    if args.benchmark:
        Tensor.manual_seed(42)  # Consistent seed for benchmarks
    print(f"🌱 Seed: {Tensor._seed}")

    # Parse model info
    model_type, model_size = parse_model_info(args.model)

    # Create generation config
    gen_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        seed=args.seed,
        framework_options={"timing": args.timing, "profile": args.profile},
    )

    # Run for each framework
    if len(args.framework) > 1:
        # Multi-framework comparison
        run_framework_comparison(args, model_type, model_size, gen_config)
    else:
        # Single framework
        run_single_framework(args.framework[0], args, model_type, model_size, gen_config)


def parse_model_info(model_name: str) -> tuple[str, str]:
    """Parse model name into type and size."""
    if model_name.startswith("llama3"):
        size = model_name.split("-")[1].upper()
        return "llama", size
    if model_name.startswith("gpt2"):
        size = model_name.split("-")[1].upper()
        return "gpt2", size
    raise ValueError(f"Unknown model format: {model_name}")


def run_single_framework(
    framework: str, args: any, model_type: str, model_size: str, gen_config: GenerationConfig
) -> None:
    """Run inference for a single framework."""
    print(f"🚀 Initializing {framework.upper()} framework")

    try:
        # Create model using factory pattern
        model = create_model(
            framework=framework,
            model_type=model_type,
            model_path=args.model_path,
            model_size=model_size,
            quantize=args.quantize,
            shard=args.shard,
            device="auto",
        )

        # Load model and tokenizer
        print("📦 Loading model...")
        start_time = time.time()
        model.load_model()
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")

        tokenizer = model.load_tokenizer()

        # Route to appropriate mode
        if args.benchmark:
            print("📊 Running benchmark mode")
            results = run_benchmark(model, gen_config, args.iterations)
            print_benchmark_results(results)
        elif args.prompt:
            print("💬 Single inference mode")
            print(f"Prompt: {args.prompt}\n")
            model.generate_single(args.prompt, gen_config)

            # Print metrics
            metrics = model.get_metrics()
            print("\n📊 Metrics:")
            print(f"  Tokens generated: {metrics.tokens_generated}")
            print(f"  First token time: {metrics.first_token_time * 1000:.1f}ms")
            print(f"  Tokens/second: {metrics.tokens_per_second:.1f}")
        else:
            print("🎮 Interactive Q&A mode")
            run_interactive_session(model, gen_config)

    except Exception as e:
        print(f"❌ Error running {framework}: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        if "model" in locals():
            model.cleanup()


def run_framework_comparison(args: any, model_type: str, model_size: str, gen_config: GenerationConfig) -> None:
    """Run comparison across multiple frameworks."""
    print("⚖️  Framework Comparison Mode")
    print(f"Frameworks: {', '.join(args.framework)}")

    if args.benchmark:
        # Benchmark comparison
        all_results = []
        for framework in args.framework:
            print(f"\n🔄 Testing {framework.upper()}...")
            try:
                model = create_model(
                    framework=framework,
                    model_type=model_type,
                    model_path=args.model_path,
                    model_size=model_size,
                    quantize=args.quantize,
                    shard=args.shard,
                    device="auto",
                )

                model.load_model()
                model.load_tokenizer()

                results = run_benchmark(model, gen_config, args.iterations)
                results["framework"] = framework
                all_results.append(results)

                model.cleanup()
            except Exception as e:
                print(f"❌ {framework} failed: {e}")
                continue

        # Print comparison table
        print_comparison_results(all_results)

    else:
        print("❌ Framework comparison only supported in benchmark mode")
        print("Use --benchmark flag for multi-framework comparison")
        sys.exit(1)


def run_interactive_session(model, gen_config: GenerationConfig) -> None:
    """Run interactive Q&A session."""
    print("Interactive Q&A mode. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            print("🤖 Assistant: ", end="", flush=True)

            # Generate response with streaming
            for token in model.generate(user_input, gen_config):
                print(token, end="", flush=True)

            print()  # New line after response

            # Print metrics
            metrics = model.get_metrics()
            if metrics.tokens_generated > 0:
                print(
                    f"📊 {metrics.tokens_generated} tokens, "
                    f"{metrics.tokens_per_second:.1f} tok/s, "
                    f"first: {metrics.first_token_time * 1000:.1f}ms\n"
                )

            model.reset_metrics()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def run_benchmark(model, gen_config: GenerationConfig, num_iterations: int = 5) -> dict:
    """Run benchmark with multiple prompts and iterations."""
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing briefly.",
        "Write a short poem about technology.",
    ]

    print(f"🏁 Running benchmark ({num_iterations} iterations)")

    results = {
        "iterations": num_iterations,
        "prompts": len(prompts),
        "generation_times": [],
        "tokens_per_second": [],
        "first_token_times": [],
    }

    for i in range(num_iterations):
        print(f"   Iteration {i + 1}/{num_iterations}")

        for prompt in prompts:
            # Use shorter config for benchmarking
            bench_config = GenerationConfig(
                max_length=50,  # Shorter for speed
                temperature=gen_config.temperature,
                seed=gen_config.seed,
            )

            model.reset_metrics()
            model.generate_single(prompt, bench_config)
            metrics = model.get_metrics()

            if metrics.tokens_generated > 0:
                results["generation_times"].append(metrics.total_inference_time)
                results["tokens_per_second"].append(metrics.tokens_per_second)
                if metrics.first_token_time:
                    results["first_token_times"].append(metrics.first_token_time)

    # Calculate averages
    if results["generation_times"]:
        results["avg_generation_time"] = sum(results["generation_times"]) / len(results["generation_times"])
        results["avg_tokens_per_second"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])

    if results["first_token_times"]:
        results["avg_first_token_time"] = sum(results["first_token_times"]) / len(results["first_token_times"])

    return results


def print_benchmark_results(results: dict) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)

    print("Configuration:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Prompts: {results['prompts']}")

    if "avg_generation_time" in results:
        print("\nPerformance Metrics:")
        print(f"  Average generation time: {results['avg_generation_time']:.3f}s")
        print(f"  Average tokens/second: {results['avg_tokens_per_second']:.1f}")

    if "avg_first_token_time" in results:
        print(f"  Average first token time: {results['avg_first_token_time'] * 1000:.1f}ms")

    print("=" * 60)


def print_comparison_results(all_results: list) -> None:
    """Print formatted comparison results."""
    if not all_results:
        print("❌ No successful benchmark results to compare")
        return

    print("\n" + "=" * 80)
    print("⚖️  FRAMEWORK COMPARISON RESULTS")
    print("=" * 80)

    # Print header
    print(f"{'Framework':<12} {'Avg Time (s)':<12} {'Tokens/sec':<12} {'First Token (ms)':<15}")
    print("-" * 80)

    # Print results for each framework
    for results in all_results:
        framework = results["framework"].upper()
        avg_time = results.get("avg_generation_time", 0.0)
        tokens_sec = results.get("avg_tokens_per_second", 0.0)
        first_token = results.get("avg_first_token_time", 0.0) * 1000

        print(f"{framework:<12} {avg_time:<12.3f} {tokens_sec:<12.1f} {first_token:<15.1f}")

    print("=" * 80)

    # Find best performing framework
    if len(all_results) > 1:
        best_throughput = max(all_results, key=lambda x: x.get("avg_tokens_per_second", 0))
        fastest_first_token = min(all_results, key=lambda x: x.get("avg_first_token_time", float("inf")))

        print("\n🏆 Performance Leaders:")
        print(
            f"  Highest throughput: {best_throughput['framework'].upper()} "
            f"({best_throughput.get('avg_tokens_per_second', 0):.1f} tok/s)"
        )
        print(
            f"  Fastest first token: {fastest_first_token['framework'].upper()} "
            f"({fastest_first_token.get('avg_first_token_time', 0) * 1000:.1f}ms)"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
