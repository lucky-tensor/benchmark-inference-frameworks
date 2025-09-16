#!/usr/bin/env python3
"""
Pure TinyGrad backend for LLaMA model inference.
This module provides a standalone TinyGrad implementation separate from PyTorch.
"""

import argparse
import sys
import time
from pathlib import Path


def get_tinygrad_model(model_size: str, model_path: Path | None = None, **kwargs):
    """Load TinyGrad model with specified configuration."""
    from tinygrad import Device

    # Add path for llama modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llama.model_config import build_transformer, resolve_model_path

    # Handle directory paths containing GGUF files
    if model_path and model_path.is_dir():
        # Look for GGUF files in the directory
        gguf_files = list(model_path.glob("*.gguf"))
        resolved_path = gguf_files[0] if gguf_files else resolve_model_path(model_path, model_size, False)
    else:
        # Resolve model path normally
        resolved_path = resolve_model_path(model_path, model_size, False)

    # Get device
    shard = kwargs.get("shard", 1)
    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(shard)) if shard > 1 else Device.DEFAULT

    # Build model
    return build_transformer(resolved_path, model_size=model_size, quantize=kwargs.get("quantize"), device=device)


def get_tinygrad_tokenizer(tokenizer_path: Path):
    """Load TinyGrad-compatible tokenizer."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.tokenizer import Tokenizer

    return Tokenizer(str(tokenizer_path))


def prepare_tinygrad_input(model, tokenizer):
    """Prepare input for TinyGrad model inference."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.generation import encode_message, encode_role, prefill

    # Prepare input tokens (same as TinyGrad benchmark)
    toks = [tokenizer.bos_id, *encode_message("user", "Hello.", tokenizer), *encode_role("assistant", tokenizer)]

    # Prefill the model
    start_pos = prefill(model, toks[:-1])

    return toks[-1], start_pos


def run_tinygrad_inference(model, input_data: int, start_pos: int):
    """Run single inference step with TinyGrad."""
    from tinygrad import GlobalCounters, Tensor

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.generation import ALPHA_F, ALPHA_P, TEMPERATURE, TOP_K, TOP_P

    GlobalCounters.reset()

    device = model.tok_embeddings.weight.device if hasattr(model.tok_embeddings.weight, "device") else "cuda"

    tok = model(Tensor([[input_data]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)

    return tok.item()


def get_tinygrad_model_info(model):
    """Get TinyGrad model information."""
    from tinygrad.nn.state import get_parameters

    params = get_parameters(model)
    total_params = sum(x.numel() for x in params)
    param_bytes = sum(x.uop.size * x.dtype.itemsize for x in params)

    return {
        "total_parameters": total_params,
        "loaded_parameters": total_params,
        "model_memory_gb": param_bytes / (1024**3),
        "precision": "FP32",  # TinyGrad uses FP32 precision
    }


def get_tinygrad_device_info():
    """Get TinyGrad device information."""
    from tinygrad import Device

    return str(Device.DEFAULT)


def run_tinygrad_benchmark(model_size: str = "1B", model_path: Path | None = None, iterations: int = 20, **kwargs):
    """Run standalone TinyGrad benchmark."""
    print("🚀 Running TinyGrad Benchmark")
    print("=" * 50)

    # Load model and tokenizer
    print("Loading TinyGrad model...")
    start_time = time.time()
    model = get_tinygrad_model(model_size, model_path, **kwargs)
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f}s")

    # Determine tokenizer path
    if model_path and model_path.is_dir():
        tokenizer_path = model_path / "tokenizer.model"
    elif model_path:
        tokenizer_path = model_path.parent / "tokenizer.model"
    else:
        # Default path
        tokenizer_path = Path.home() / "models" / f"llama3-{model_size.lower()}-instruct" / "tokenizer.model"

    print("Loading tokenizer...")
    tokenizer = get_tinygrad_tokenizer(tokenizer_path)

    # Get model info
    model_info = get_tinygrad_model_info(model)
    device = get_tinygrad_device_info()

    print(f"Model: {model_info['total_parameters']:,} parameters")
    print(f"Device: {device}")
    print(f"Memory: {model_info['model_memory_gb']:.2f} GB")

    # Prepare input
    input_data, start_pos = prepare_tinygrad_input(model, tokenizer)

    # Run cold start measurement
    print("\n🥶 Cold Start Measurement (includes JIT compilation)")
    print("=" * 50)

    # Time the cold start (first inference)
    cold_start_time = time.perf_counter()

    try:
        _cold_start_token = run_tinygrad_inference(model, input_data, start_pos)
        cold_start_end = time.perf_counter()
        cold_start_duration = cold_start_end - cold_start_time

        print(f"❄️  Cold start: {cold_start_duration * 1000:6.2f}ms, {1.0 / cold_start_duration:6.1f} tok/s")

    except Exception as e:
        print(f"❌ Cold start failed: {e}")
        raise

    # Run steady-state benchmark (excluding cold start)
    print(f"\n🔥 Steady-State Benchmark ({iterations} iterations)")
    print("=" * 50)
    token_times = []
    warmup_iterations = min(3, iterations // 4)  # Use first 25% or 3 iterations as warmup

    for i in range(iterations):
        # Time the inference
        start_time = time.perf_counter()

        try:
            run_tinygrad_inference(model, input_data, start_pos + i + 1)  # +1 because cold start used start_pos
            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            token_times.append(iteration_time)

            tokens_per_second = 1.0 / iteration_time
            status_symbol = "🌡️" if i < warmup_iterations else "⚡"
            warmup_text = " (warmup)" if i < warmup_iterations else ""
            print(
                f"{status_symbol} Iteration {i + 1:2d}: {iteration_time * 1000:6.2f}ms, {tokens_per_second:6.1f} tok/s{warmup_text}"
            )

        except Exception as e:
            print(f"❌ Error in iteration {i + 1}: {e}")
            break

    # Calculate results
    if not token_times:
        raise RuntimeError("No successful iterations completed")

    # Overall steady-state metrics
    avg_token_time = sum(token_times) / len(token_times)
    total_time = sum(token_times)
    min_token_time = min(token_times)
    max_token_time = max(token_times)

    avg_tokens_per_second = 1.0 / avg_token_time
    peak_tokens_per_second = 1.0 / min_token_time

    # Calculate steady-state metrics (excluding warmup)
    warmup_iterations = min(3, len(token_times) // 4)
    steady_state_times = token_times[warmup_iterations:] if len(token_times) > warmup_iterations else token_times

    if steady_state_times:
        steady_state_avg_time = sum(steady_state_times) / len(steady_state_times)
        steady_state_avg_tokens_per_second = 1.0 / steady_state_avg_time
        first_steady_token_time = steady_state_times[0] if steady_state_times else avg_token_time
    else:
        steady_state_avg_tokens_per_second = avg_tokens_per_second
        first_steady_token_time = avg_token_time

    # Print results
    print("\n🏆 TinyGrad Benchmark Results")
    print("=" * 50)
    print(f"Model: LLaMA {model_size} ({model_info['total_parameters']:,} parameters)")
    print(f"Device: {device}")
    print(f"Iterations: {len(token_times)}")

    print("\n📥 Model Loading:")
    print(f"  Model load time:     {model_load_time:6.2f}s")

    print("\n❄️  Cold Start Metrics:")
    print(f"  Cold start latency:  {cold_start_duration * 1000:6.2f}ms (includes JIT compilation)")
    print(f"  Cold start throughput: {1.0 / cold_start_duration:6.1f} tokens/second")

    print("\n⏱️  Steady-State Performance Metrics:")
    print(f"  Average latency:     {avg_token_time * 1000:6.2f}ms per token")
    print(f"  First token latency:  {first_steady_token_time * 1000:6.2f}ms")
    print(f"  Min latency:         {min_token_time * 1000:6.2f}ms per token")
    print(f"  Max latency:         {max_token_time * 1000:6.2f}ms per token")

    print("\n🚀 Throughput Metrics:")
    print(f"  Average throughput:  {avg_tokens_per_second:6.1f} tokens/second")
    print(f"  Peak throughput:     {peak_tokens_per_second:6.1f} tokens/second")
    print(f"  Steady-state avg:    {steady_state_avg_tokens_per_second:6.1f} tokens/second")

    # Show performance improvement from cold start to steady state
    improvement_factor = steady_state_avg_tokens_per_second / (1.0 / cold_start_duration)
    print(f"  Warmup improvement:  {improvement_factor:6.1f}x faster than cold start")

    print("\n💾 Memory Metrics:")
    print(f"  Model memory:        {model_info['model_memory_gb']:6.2f} GB")
    print(f"  Peak memory:         {model_info['model_memory_gb']:6.2f} GB (estimate)")
    print(f"  Precision:           {model_info['precision']}")


def main():
    """Standalone TinyGrad backend for benchmarking."""
    parser = argparse.ArgumentParser(description="Pure TinyGrad LLaMA Backend")
    parser.add_argument("--size", choices=["1B", "8B", "70B", "405B"], default="1B", help="Model size")
    parser.add_argument("--model", type=Path, help="Path to model directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Number of device shards")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")

    args = parser.parse_args()

    if args.benchmark:
        run_tinygrad_benchmark(
            model_size=args.size,
            model_path=args.model,
            iterations=args.iterations,
            quantize=args.quantize,
            shard=args.shard,
        )
    else:
        # Interactive mode
        print("TinyGrad LLaMA Model loading...")
        model = get_tinygrad_model(args.size, args.model, quantize=args.quantize, shard=args.shard)

        # Load tokenizer
        tokenizer_path = (
            args.model / "tokenizer.model" if args.model else Path.home() / "models/llama3-1b-instruct/tokenizer.model"
        )
        tokenizer = get_tinygrad_tokenizer(tokenizer_path)

        print("TinyGrad LLaMA Model loaded. Type 'quit' to exit.")

        while True:
            user_input = input("User: ")
            if user_input.lower() == "quit":
                break

            # Simple generation using TinyGrad's built-in chat interface
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from common.chat_interface import interact_with_model

                # This would need to be adapted for the specific chat interface
                print("Assistant: (TinyGrad interactive mode - full implementation would use chat_interface.py)")

            except ImportError:
                print("Assistant: Interactive chat not available in standalone mode")


if __name__ == "__main__":
    main()
