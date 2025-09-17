#!/usr/bin/env python3
"""
Pure TinyGrad backend for LLaMA model inference.
This module provides a standalone TinyGrad implementation separate from PyTorch.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Set optimal TinyGrad environment variables at module level for maximum performance
os.environ.setdefault("JIT", "1")          # Enable JIT compilation
os.environ.setdefault("CUDA", "1")         # Enable CUDA if available
os.environ.setdefault("FASTMATH", "1")     # Enable fast math optimizations
os.environ.setdefault("OPT", "2")          # Maximum optimization level
os.environ.setdefault("CLCACHE", "1")      # Enable kernel cache
os.environ.setdefault("CUDACACHE", "1")    # Enable CUDA kernel cache

# TinyGrad-specific optimizations for better caching
os.environ.setdefault("TINYGRAD_JIT", "1")       # Enable TinyGrad JIT
os.environ.setdefault("TINYGRAD_FUSION", "1")    # Enable kernel fusion

# CUDA driver cache settings for better persistence
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")  # 4GB cache size (vs 256MB default)
os.environ.setdefault("CUDA_CACHE_DISABLE", "0")           # Ensure caching is enabled
os.environ.setdefault("CUDA_FORCE_PTX_JIT", "0")          # Use cached binaries when available

# Set cache path for consistency (optional - uses default location)
cuda_cache_path = os.path.expanduser("~/.nv/ComputeCache")
os.environ.setdefault("CUDA_CACHE_PATH", cuda_cache_path)

# BEAM optimization configuration (4x speed improvement but very long compile times)
# Can be disabled by setting TINYGRAD_BEAM=0 environment variable
USE_BEAM = os.getenv("TINYGRAD_BEAM", "1") != "0"
if USE_BEAM:
    print("🚀 BEAM optimization enabled - expect long compile time but 4x speed improvement")


def get_tinygrad_model(model_size: str, model_path: Path | None = None, **kwargs):
    """Load TinyGrad model with specified configuration."""
    from tinygrad import Context, Device

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

    # Build model with BEAM optimization if enabled
    if USE_BEAM:
        with Context(BEAM=1):
            return build_transformer(resolved_path, model_size=model_size, quantize=kwargs.get("quantize"), device=device)
    else:
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
    from tinygrad import Context, GlobalCounters, Tensor

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.generation import ALPHA_F, ALPHA_P, TEMPERATURE, TOP_K, TOP_P

    # Reset counters once per inference call (matches main branch performance pattern)
    GlobalCounters.reset()

    device = model.tok_embeddings.weight.device if hasattr(model.tok_embeddings.weight, "device") else "cuda"

    # Run inference with BEAM optimization if enabled
    if USE_BEAM:
        with Context(BEAM=1):
            tok = model(Tensor([[input_data]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
    else:
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


def warm_tinygrad_cache(model, tokenizer, verbose: bool = False):
    """Warm up TinyGrad JIT cache by running a few inference steps."""
    if verbose:
        print("🔥 Warming TinyGrad JIT cache...")

    # Run a few quick inference steps to populate cache
    for i in range(3):
        try:
            # Use a simple single token to warm cache
            input_data, start_pos = prepare_tinygrad_input(model, tokenizer)
            _ = run_tinygrad_inference(model, input_data, start_pos + i)
            if verbose and i == 0:
                print("   Cache warming completed")
            break  # Exit after first successful run
        except Exception as e:
            if verbose:
                print(f"   Cache warming step {i+1} failed: {e}")
            continue


def direct_benchmark_test(model_path: Path, iterations: int = 5):
    """Direct benchmark test bypassing the executor framework."""
    print("🚀 Direct TinyGrad Performance Test (bypassing executor framework)")
    print("=" * 65)

    # Load model directly
    print("Loading model...")
    model_load_start = time.perf_counter()
    model = get_tinygrad_model("1B", model_path)
    model_load_time = time.perf_counter() - model_load_start
    print(f"✅ Model loaded in {model_load_time:.2f}s")

    # Load tokenizer directly
    tokenizer = get_tinygrad_tokenizer(model_path / "tokenizer.model")
    print("✅ Tokenizer loaded")

    # Prepare input directly
    input_data, start_pos = prepare_tinygrad_input(model, tokenizer)

    # Cold start test
    print("\n❄️  Cold start test...")
    cold_start = time.perf_counter()
    run_tinygrad_inference(model, input_data, start_pos)
    cold_time = time.perf_counter() - cold_start
    print(f"Cold start: {cold_time * 1000:.2f}ms ({1/cold_time:.1f} tok/s)")

    # Steady state tests
    print(f"\n⚡ Steady-state test ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        run_tinygrad_inference(model, input_data, start_pos + i + 1)
        end = time.perf_counter()
        iteration_time = end - start
        times.append(iteration_time)
        print(f"  Iteration {i+1}: {iteration_time*1000:.2f}ms ({1/iteration_time:.1f} tok/s)")

    # Results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    print("\n🏆 Direct Performance Results:")
    print(f"  Average: {avg_time*1000:.2f}ms ({1/avg_time:.1f} tok/s)")
    print(f"  Peak:    {min_time*1000:.2f}ms ({1/min_time:.1f} tok/s)")
    print("  vs Main Branch Target: 82+ tok/s")


def main():
    """Simple TinyGrad backend testing utility."""
    parser = argparse.ArgumentParser(description="TinyGrad LLaMA Backend Functions")
    parser.add_argument("--size", choices=["1B", "8B", "70B", "405B"], default="1B", help="Model size")
    parser.add_argument("--model", type=Path, help="Path to model directory")
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Number of device shards")
    parser.add_argument("--direct-benchmark", action="store_true", help="Run direct performance test")

    args = parser.parse_args()

    if args.direct_benchmark:
        if args.model:
            direct_benchmark_test(args.model)
        else:
            direct_benchmark_test(Path.home() / "models/llama3-1b-instruct")
        return

    print("TinyGrad LLaMA Backend Functions")
    print("=" * 40)
    print("This module provides backend functions for TinyGrad inference.")
    print("For benchmarking, use: uv run src/main.py --framework tinygrad")
    print("")
    print("Available functions:")
    print("  - get_tinygrad_model()")
    print("  - get_tinygrad_tokenizer()")
    print("  - prepare_tinygrad_input()")
    print("  - run_tinygrad_inference()")
    print("  - get_tinygrad_model_info()")
    print("  - get_tinygrad_device_info()")
    print("")
    print("Example usage in main benchmark suite:")
    print("  uv run src/main.py --model-id llama3-1b-instruct \\")
    print("    --model-path ~/models/llama3-1b-instruct \\")
    print("    --framework tinygrad --iterations 10")
    print("")
    print("Direct performance test:")
    print("  uv run src/frameworks/tinygrad/backends/tinygrad_backend.py --direct-benchmark")


if __name__ == "__main__":
    main()
