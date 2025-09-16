#!/usr/bin/env python3
"""
Unified benchmarking system for comparing different ML frameworks.

This script provides standardized benchmarking across TinyGrad, PyTorch, and
potentially other frameworks with shared reporting and analysis.
"""

import argparse
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "llama"))  # For extra module imports


@dataclass
class BenchmarkResult:
    """Standardized benchmark results across frameworks."""

    framework: str
    model_size: str
    iterations: int

    # Timing metrics (in seconds)
    model_load_time: float  # Time to load model and tokenizer
    total_time: float
    cold_start_time: float  # Time for first inference (including JIT compilation)
    first_token_time: float  # Time for first token after cold start
    avg_token_time: float
    min_token_time: float
    max_token_time: float
    token_times: list[float]  # Excludes cold start time

    # Throughput metrics (excluding cold start)
    avg_tokens_per_second: float
    peak_tokens_per_second: float
    steady_state_avg_tokens_per_second: float  # Average after warmup

    # Memory metrics (in GB)
    peak_memory_gb: float
    model_memory_gb: float

    # Model info
    total_parameters: int
    loaded_parameters: int

    # Additional metadata
    device: str
    precision: str
    quantization: str | None = None


class FrameworkBackend(ABC):
    """Abstract base class for framework-specific implementations."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the framework name."""

    @abstractmethod
    def load_model(self, model_size: str, model_path: Path | None = None, **kwargs) -> Any:
        """Load and return the model."""

    @abstractmethod
    def load_tokenizer(self, tokenizer_path: Path) -> Any:
        """Load and return the tokenizer."""

    @abstractmethod
    def prepare_input(self, model: Any, tokenizer: Any) -> tuple[Any, int]:
        """Prepare benchmark input. Returns (input, start_position)."""

    @abstractmethod
    def run_inference(self, model: Any, input_data: Any, start_pos: int) -> Any:
        """Run single inference step. Returns next token."""

    @abstractmethod
    def get_model_info(self, model: Any) -> dict[str, Any]:
        """Get model information (parameters, memory, etc.)."""

    @abstractmethod
    def get_device_info(self) -> str:
        """Get device information."""

    @abstractmethod
    def cleanup(self, model: Any, tokenizer: Any) -> None:
        """Cleanup resources."""


class TinyGradBackend(FrameworkBackend):
    """TinyGrad framework implementation using separated backend."""

    def __init__(self):
        pass

    def get_name(self) -> str:
        return "TinyGrad"

    def load_model(self, model_size: str, model_path: Path | None = None, **kwargs) -> Any:
        from backends.tinygrad_backend import get_tinygrad_model
        return get_tinygrad_model(model_size, model_path, **kwargs)

    def load_tokenizer(self, tokenizer_path: Path) -> Any:
        from backends.tinygrad_backend import get_tinygrad_tokenizer
        return get_tinygrad_tokenizer(tokenizer_path)

    def prepare_input(self, model: Any, tokenizer: Any) -> tuple[Any, int]:
        from backends.tinygrad_backend import prepare_tinygrad_input
        return prepare_tinygrad_input(model, tokenizer)

    def run_inference(self, model: Any, input_data: Any, start_pos: int) -> Any:
        from backends.tinygrad_backend import run_tinygrad_inference
        return run_tinygrad_inference(model, input_data, start_pos)

    def get_model_info(self, model: Any) -> dict[str, Any]:
        from backends.tinygrad_backend import get_tinygrad_model_info
        return get_tinygrad_model_info(model)

    def get_device_info(self) -> str:
        from backends.tinygrad_backend import get_tinygrad_device_info
        return get_tinygrad_device_info()

    def cleanup(self, model: Any, tokenizer: Any) -> None:
        # TinyGrad cleanup if needed
        pass


class PyTorchBackend(FrameworkBackend):
    """PyTorch framework implementation using separated backend."""

    def __init__(self):
        self._prefill_tokens = None
        self._prefilled = False

    def get_name(self) -> str:
        return "PyTorch"

    def load_model(self, model_size: str, model_path: Path | None = None, **kwargs) -> Any:
        import torch

        from backends.pytorch_backend import MODEL_CONFIGS, PyTorchLLaMA, load_pytorch_weights_from_tinygrad

        # Create model
        config = MODEL_CONFIGS[model_size]
        model = PyTorchLLaMA(**config)

        # Apply same optimizations as TinyGrad for fair comparison
        use_half = kwargs.get("use_half", True)  # Match TinyGrad mixed precision
        use_compile = kwargs.get("use_compile", False)  # Disabled by default due to CUDA graph issues

        if use_half:
            model = model.half()  # Use float16 like TinyGrad mixed precision

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if use_compile and hasattr(torch, 'compile'):
            print("Applying torch.compile for fair comparison with TinyGrad JIT...")
            # Clear CUDA cache before compilation to avoid resource conflicts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                # Try more conservative compilation settings
                model = torch.compile(model, mode="reduce-overhead", dynamic=False, fullgraph=False)
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), falling back to eager mode")
                use_compile = False

        model.eval()

        # Load weights if provided
        if model_path and model_path.is_dir():
            gguf_files = list(model_path.glob("*.gguf"))
            if gguf_files:
                weight_path = gguf_files[0]
                load_pytorch_weights_from_tinygrad(model, weight_path)

        return model

    def load_tokenizer(self, tokenizer_path: Path) -> Any:
        from backends.pytorch_backend import PyTorchTokenizer
        return PyTorchTokenizer(str(tokenizer_path))

    def prepare_input(self, model: Any, tokenizer: Any) -> tuple[Any, int]:
        # Prepare input (matching PyTorch backend)
        def encode_message(role: str, content: str) -> list[int]:
            if role == "user":
                return [
                    tokenizer.special_tokens["<|start_header_id|>"],
                    *tokenizer.encode("user"),
                    tokenizer.special_tokens["<|end_header_id|>"],
                    *tokenizer.encode("\n\n" + content.strip()),
                    tokenizer.special_tokens["<|eot_id|>"],
                ]
            return []

        def encode_role(role: str) -> list[int]:
            return [
                tokenizer.special_tokens["<|start_header_id|>"],
                *tokenizer.encode(role),
                tokenizer.special_tokens["<|end_header_id|>"],
                *tokenizer.encode("\n\n"),
            ]

        toks = [tokenizer.bos_id, *encode_message("user", "Hello."), *encode_role("assistant")]

        # For PyTorch, we need to do prefill manually
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store for later use
        self._prefill_tokens = torch.tensor([toks[:-1]], device=device, dtype=torch.long)

        return toks[-1], len(toks) - 1

    def run_inference(self, model: Any, input_data: Any, start_pos: int) -> Any:
        import torch

        device = next(model.parameters()).device

        # First iteration needs prefill
        if not self._prefilled:
            # Ensure CUDA is properly synchronized before first call
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                # Use normal temperature for prefill, not NaN
                _ = model(self._prefill_tokens, 0, temperature=0.85)

            # Synchronize after first compilation/execution
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self._prefilled = True

        # Run inference
        with torch.no_grad():
            input_tensor = torch.tensor([[input_data]], device=device, dtype=torch.long)
            tok = model(input_tensor, start_pos, temperature=0.85)

            if hasattr(tok, "item"):
                return tok.item()
            return int(tok[0])

    def get_model_info(self, model: Any) -> dict[str, Any]:
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        model_memory_gb = param_memory / (1024**3)

        return {
            "total_parameters": total_params,
            "loaded_parameters": total_params,
            "model_memory_gb": model_memory_gb,
            "precision": str(next(model.parameters()).dtype),
        }

    def get_device_info(self) -> str:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def cleanup(self, model: Any, tokenizer: Any) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(
    backend: FrameworkBackend, model_size: str = "1B", model_path: Path | None = None, iterations: int = 20, **kwargs
) -> BenchmarkResult:
    """Run standardized benchmark on specified backend."""

    print(f"🚀 Running {backend.get_name()} Benchmark")
    print("=" * 50)

    # Load model and tokenizer
    print("Loading model...")
    start_time = time.time()
    model = backend.load_model(model_size, model_path, **kwargs)
    model_only_time = time.time() - start_time
    print(f"Model loaded in {model_only_time:.2f}s")

    # Determine tokenizer path
    if model_path and model_path.is_dir():
        tokenizer_path = model_path / "tokenizer.model"
    elif model_path:
        tokenizer_path = model_path.parent / "tokenizer.model"
    else:
        # Default path
        tokenizer_path = Path.home() / "models" / f"llama3-{model_size.lower()}-instruct" / "tokenizer.model"

    print("Loading tokenizer...")
    tokenizer_start_time = time.time()
    tokenizer = backend.load_tokenizer(tokenizer_path)
    tokenizer_load_time = time.time() - tokenizer_start_time

    # Total model loading time (model + tokenizer)
    model_load_time = model_only_time + tokenizer_load_time

    # Get model info
    model_info = backend.get_model_info(model)
    device = backend.get_device_info()

    print(f"Model: {model_info['total_parameters']:,} parameters")
    print(f"Device: {device}")
    print(f"Memory: {model_info['model_memory_gb']:.2f} GB")

    # Prepare input
    input_data, start_pos = backend.prepare_input(model, tokenizer)

    # Run cold start measurement
    print("\n🥶 Cold Start Measurement (includes JIT compilation)")
    print("=" * 50)

    # Memory tracking before cold start
    if backend.get_name() == "PyTorch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # Time the cold start (first inference)
    cold_start_time = time.perf_counter()

    try:
        cold_start_token = backend.run_inference(model, input_data, start_pos)

        if backend.get_name() == "PyTorch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        cold_start_end = time.perf_counter()
        cold_start_duration = cold_start_end - cold_start_time

        print(f"❄️  Cold start: {cold_start_duration * 1000:6.2f}ms, {1.0 / cold_start_duration:6.1f} tok/s")

    except Exception as e:
        print(f"❌ Cold start failed: {e}")
        raise

    # Track memory usage after cold start
    peak_memory_gb = 0.0
    if backend.get_name() == "PyTorch":
        import torch

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        # For TinyGrad, use model memory as estimate
        peak_memory_gb = model_info["model_memory_gb"]

    # Run steady-state benchmark (excluding cold start)
    print(f"\n🔥 Steady-State Benchmark ({iterations} iterations)")
    print("=" * 50)
    token_times = []
    warmup_iterations = min(3, iterations // 4)  # Use first 25% or 3 iterations as warmup

    for i in range(iterations):
        # Memory tracking (framework-specific)
        if backend.get_name() == "PyTorch":
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Time the inference
        start_time = time.perf_counter()

        try:
            backend.run_inference(model, input_data, start_pos + i + 1)  # +1 because cold start used start_pos

            if backend.get_name() == "PyTorch":
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            token_times.append(iteration_time)

            # Track memory (update peak if higher)
            if backend.get_name() == "PyTorch":
                import torch

                if torch.cuda.is_available():
                    current_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_memory_gb = max(peak_memory_gb, current_memory)

            tokens_per_second = 1.0 / iteration_time
            status_symbol = "🌡️" if i < warmup_iterations else "⚡"
            warmup_text = " (warmup)" if i < warmup_iterations else ""
            print(
                f"{status_symbol} Iteration {i + 1:2d}: {iteration_time * 1000:6.2f}ms, {tokens_per_second:6.1f} tok/s{warmup_text}"
            )

        except Exception as e:
            print(f"❌ Error in iteration {i + 1}: {e}")
            break

    # Cleanup
    backend.cleanup(model, tokenizer)

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

    return BenchmarkResult(
        framework=backend.get_name(),
        model_size=model_size,
        iterations=len(token_times),
        model_load_time=model_load_time,
        total_time=total_time,
        cold_start_time=cold_start_duration,
        first_token_time=first_steady_token_time,  # First token after cold start
        avg_token_time=avg_token_time,
        min_token_time=min_token_time,
        max_token_time=max_token_time,
        token_times=token_times,
        avg_tokens_per_second=avg_tokens_per_second,
        peak_tokens_per_second=peak_tokens_per_second,
        steady_state_avg_tokens_per_second=steady_state_avg_tokens_per_second,
        peak_memory_gb=peak_memory_gb,
        model_memory_gb=model_info["model_memory_gb"],
        total_parameters=model_info["total_parameters"],
        loaded_parameters=model_info["loaded_parameters"],
        device=device,
        precision=model_info["precision"],
        quantization=kwargs.get("quantize"),
    )


def print_benchmark_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print(f"\n🏆 {result.framework} Benchmark Results")
    print("=" * 50)
    print(f"Model: LLaMA {result.model_size} ({result.total_parameters:,} parameters)")
    print(f"Device: {result.device}")
    print(f"Iterations: {result.iterations}")

    if result.quantization:
        print(f"Quantization: {result.quantization}")

    print("\n📥 Model Loading:")
    print(f"  Model load time:     {result.model_load_time:6.2f}s (model + tokenizer)")

    print("\n❄️  Cold Start Metrics:")
    print(f"  Cold start latency:  {result.cold_start_time * 1000:6.2f}ms (includes JIT compilation)")
    print(f"  Cold start throughput: {1.0 / result.cold_start_time:6.1f} tokens/second")

    print("\n⏱️  Steady-State Performance Metrics:")
    print(f"  Average latency:     {result.avg_token_time * 1000:6.2f}ms per token")
    print(f"  First token latency:  {result.first_token_time * 1000:6.2f}ms")
    print(f"  Min latency:         {result.min_token_time * 1000:6.2f}ms per token")
    print(f"  Max latency:         {result.max_token_time * 1000:6.2f}ms per token")

    print("\n🚀 Throughput Metrics:")
    print(f"  Average throughput:  {result.avg_tokens_per_second:6.1f} tokens/second")
    print(f"  Peak throughput:     {result.peak_tokens_per_second:6.1f} tokens/second")
    print(f"  Steady-state avg:    {result.steady_state_avg_tokens_per_second:6.1f} tokens/second")

    # Show performance improvement from cold start to steady state
    improvement_factor = result.steady_state_avg_tokens_per_second / (1.0 / result.cold_start_time)
    print(f"  Warmup improvement:  {improvement_factor:6.1f}x faster than cold start")

    print("\n💾 Memory Metrics:")
    print(f"  Model memory:        {result.model_memory_gb:6.2f} GB")
    print(f"  Peak memory:         {result.peak_memory_gb:6.2f} GB")
    print(f"  Precision:           {result.precision}")


def compare_results(results: list[BenchmarkResult]) -> None:
    """Compare multiple benchmark results."""
    if len(results) < 2:
        return

    print("\n📊 Framework Comparison")
    print("=" * 80)

    # Create comparison table
    frameworks = [r.framework for r in results]

    print(f"{'Metric':<25} {' | '.join(f'{fw:>15}' for fw in frameworks)}")
    print("-" * (25 + len(frameworks) * 18))

    # Model loading metrics
    load_times = [r.model_load_time for r in results]
    print(f"{'Model Load (s)':<25} {' | '.join(f'{lt:>15.2f}' for lt in load_times)}")

    print()  # Separator

    # Cold start metrics
    cold_start_times = [r.cold_start_time * 1000 for r in results]
    cold_start_throughputs = [1.0 / r.cold_start_time for r in results]

    print(f"{'Cold Start (ms)':<25} {' | '.join(f'{cs:>15.2f}' for cs in cold_start_times)}")
    print(f"{'Cold Start (tok/s)':<25} {' | '.join(f'{cst:>15.1f}' for cst in cold_start_throughputs)}")

    print()  # Separator

    # Steady-state metrics
    latencies = [r.avg_token_time * 1000 for r in results]
    throughputs = [r.avg_tokens_per_second for r in results]
    steady_throughputs = [r.steady_state_avg_tokens_per_second for r in results]
    memories = [r.peak_memory_gb for r in results]

    print(f"{'Avg Latency (ms)':<25} {' | '.join(f'{lat:>15.2f}' for lat in latencies)}")
    print(f"{'Avg Throughput (tok/s)':<25} {' | '.join(f'{thr:>15.1f}' for thr in throughputs)}")
    print(f"{'Steady-State (tok/s)':<25} {' | '.join(f'{st:>15.1f}' for st in steady_throughputs)}")
    print(f"{'Peak Memory (GB)':<25} {' | '.join(f'{mem:>15.2f}' for mem in memories)}")

    # Calculate relative performance
    if len(results) == 2:
        r1, r2 = results

        # Calculate various performance ratios
        load_time_ratio = r2.model_load_time / r1.model_load_time
        cold_start_ratio = r2.cold_start_time / r1.cold_start_time
        throughput_ratio = r1.avg_tokens_per_second / r2.avg_tokens_per_second
        steady_state_ratio = r1.steady_state_avg_tokens_per_second / r2.steady_state_avg_tokens_per_second
        latency_ratio = r2.avg_token_time / r1.avg_token_time
        memory_ratio = r2.peak_memory_gb / r1.peak_memory_gb

        print(f"\n🏁 Performance Comparison ({r1.framework} vs {r2.framework}):")

        # Model loading comparison
        if load_time_ratio > 1:
            print(f"  📥 {r1.framework} loads {load_time_ratio:.1f}x faster")
        else:
            print(f"  📥 {r2.framework} loads {1 / load_time_ratio:.1f}x faster")

        # Cold start comparison
        if cold_start_ratio > 1:
            print(f"  ❄️  {r1.framework} has {cold_start_ratio:.1f}x faster cold start")
        else:
            print(f"  ❄️  {r2.framework} has {1 / cold_start_ratio:.1f}x faster cold start")

        # Steady-state comparison
        if steady_state_ratio > 1:
            print(f"  🔥 {r1.framework} is {steady_state_ratio:.1f}x faster in steady-state throughput")
        else:
            print(f"  🔥 {r2.framework} is {1 / steady_state_ratio:.1f}x faster in steady-state throughput")

        if throughput_ratio > 1:
            print(f"  ⚡ {r1.framework} is {throughput_ratio:.1f}x faster in average throughput")
        else:
            print(f"  ⚡ {r2.framework} is {1 / throughput_ratio:.1f}x faster in average throughput")

        if latency_ratio > 1:
            print(f"  ⏱️  {r1.framework} has {latency_ratio:.1f}x lower latency")
        else:
            print(f"  ⏱️  {r2.framework} has {1 / latency_ratio:.1f}x lower latency")

        if memory_ratio > 1:
            print(f"  💾 {r1.framework} uses {memory_ratio:.1f}x less memory")
        else:
            print(f"  💾 {r2.framework} uses {1 / memory_ratio:.1f}x less memory")


def detect_model_type(model_path: Path) -> str:
    """Detect model type from model path and files."""
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Check for LLaMA indicators
    if model_path.is_dir():
        # Look for LLaMA-specific files
        files = list(model_path.rglob("*"))
        file_names = [f.name.lower() for f in files]

        # LLaMA indicators
        llama_indicators = ["llama", "tokenizer.model", "consolidated", "model.safetensors", ".gguf", "config.json"]

        # GPT indicators
        gpt_indicators = ["gpt", "vocab.bpe", "encoder.json", "pytorch_model"]

        llama_score = sum(1 for indicator in llama_indicators if any(indicator in name for name in file_names))
        gpt_score = sum(1 for indicator in gpt_indicators if any(indicator in name for name in file_names))

        if llama_score > gpt_score:
            return "llama"
        if gpt_score > 0:
            return "gpt"
    else:
        # Single file - check filename and extension
        filename = model_path.name.lower()
        if "llama" in filename or filename.endswith(".gguf"):
            return "llama"
        if "gpt" in filename:
            return "gpt"

    # Default to llama if unclear (most common in this codebase)
    return "llama"


def validate_model_type(model_path: Path, expected_type: str) -> None:
    """Validate that model type matches the expected type."""
    detected_type = detect_model_type(model_path)

    if detected_type != expected_type:
        raise ValueError(
            f"Model type mismatch: Expected '{expected_type}' but detected '{detected_type}' "
            f"from model path '{model_path}'. Please check your --model-type argument."
        )


def detect_model_size(model_path: Path) -> str:
    """Auto-detect model size from model path."""
    path_str = str(model_path).lower()

    # Look for size indicators in path
    if "1b" in path_str or "1.2b" in path_str:
        return "1B"
    if "8b" in path_str:
        return "8B"
    if "70b" in path_str:
        return "70B"
    if "405b" in path_str:
        return "405B"

    # Default to 1B if unclear
    print(f"⚠️  Could not detect model size from path '{model_path}', defaulting to 1B")
    return "1B"


def main():
    parser = argparse.ArgumentParser(
        description="Unified ML Framework Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run benchmark.py --framework tinygrad --model-type llama --model-path ~/models/llama3-1b-instruct/
  uv run benchmark.py --framework pytorch --model-type llama --model-path ~/models/llama3-1b-instruct/
  uv run benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/  # Compares all frameworks
        """,
    )

    # Framework selection
    parser.add_argument(
        "--framework",
        choices=["tinygrad", "pytorch"],
        nargs="*",
        help="Framework(s) to benchmark. If not specified, compares all frameworks.",
    )

    # Model options
    parser.add_argument("--model-type", choices=["llama", "gpt"], required=True, help="Model type to load")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model directory or file")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")

    # Model configuration
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Number of device shards")

    # Fairness options for PyTorch
    parser.add_argument("--fair-comparison", action="store_true", default=True,
                        help="Enable fair comparison mode (PyTorch uses same precision and optimizations as TinyGrad)")
    parser.add_argument("--pytorch-compile", action="store_true",
                        help="Enable torch.compile in PyTorch (default: disabled due to CUDA issues)")
    parser.add_argument("--pytorch-no-half", action="store_true",
                        help="Disable half precision in PyTorch (default: enabled for fair comparison)")

    args = parser.parse_args()

    # Validate model type matches the model path
    try:
        validate_model_type(args.model_path, args.model_type)
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return 1

    # Check if model type is supported
    if args.model_type != "llama":
        print(f"❌ Model type '{args.model_type}' is not currently supported. Only 'llama' is implemented.")
        return 1

    # Auto-detect model size from path if not explicitly provided
    model_size = detect_model_size(args.model_path)

    # Select backends to run
    backends = []

    # If no frameworks specified, default to all frameworks
    if not args.framework:
        frameworks = ["tinygrad", "pytorch"]
        print("No frameworks specified, benchmarking all frameworks for comparison")
    else:
        frameworks = args.framework

    if "tinygrad" in frameworks:
        backends.append(TinyGradBackend())
    if "pytorch" in frameworks:
        backends.append(PyTorchBackend())

    # Run benchmarks
    results = []
    for backend in backends:
        try:
            # Configure fairness options for PyTorch
            fairness_kwargs = {}
            if backend.get_name() == "PyTorch" and args.fair_comparison:
                fairness_kwargs.update({
                    "use_half": not args.pytorch_no_half,
                    "use_compile": args.pytorch_compile,
                })

            result = run_benchmark(
                backend=backend,
                model_size=model_size,
                model_path=args.model_path,
                iterations=args.iterations,
                quantize=args.quantize,
                shard=args.shard,
                **fairness_kwargs,
            )
            results.append(result)
            print_benchmark_results(result)

        except Exception as e:
            print(f"❌ {backend.get_name()} benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Show comparison if multiple frameworks were benchmarked
    if len(results) > 1:
        compare_results(results)

    return 0


if __name__ == "__main__":
    main()
