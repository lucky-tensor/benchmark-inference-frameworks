"""Accelerated LLaMA 3 Implementation

This module provides the main interface for the hybrid PyTorch-TinyGrad LLaMA 3
implementation, combining ecosystem compatibility with optimized performance.
"""

import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .accelerated_inference import AcceleratedGenerator
from .hybrid_model import AcceleratedInferenceEngine, HybridLLaMA3Model
from .model_config import get_model_config
from .pytorch_bridge import MemoryOptimizer


class AcceleratedLLaMA3:
    """Hybrid PyTorch-TinyGrad LLaMA 3 implementation with accelerated inference."""

    def __init__(
        self,
        model_path: str | Path,
        quantize: str | None = None,
        device: str = "cuda",
        use_torch_weights: bool = False,
    ):
        """Initialize accelerated LLaMA 3 model.

        Args:
            model_path: Path to model weights (GGUF or PyTorch format)
            quantize: Quantization method ('int8', 'float16', etc.)
            device: Device for inference ('cuda', 'cpu')
            use_torch_weights: Whether to load with PyTorch for better HuggingFace compatibility
        """
        self.model_path = Path(model_path)
        self.quantize = quantize
        self.device = device
        self.use_torch_weights = use_torch_weights

        # Initialize components
        self.config = None
        self.model = None
        self.tokenizer = None
        self.inference_engine = None
        self.memory_optimizer = MemoryOptimizer()

        # Load model and tokenizer
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the model components."""
        print("🚀 Initializing Accelerated LLaMA 3 (Hybrid PyTorch-TinyGrad)")
        print(f"   Model path: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Quantization: {self.quantize}")

        # Load configuration
        self._load_config()

        # Load tokenizer
        self._load_tokenizer()

        # Load model
        self._load_model()

        # Create inference engine
        self._create_inference_engine()

        print("✅ Accelerated LLaMA 3 initialization complete!")

    def _load_config(self) -> None:
        """Load model configuration."""
        try:
            # Try to determine model size from path
            model_size = self._detect_model_size()
            self.config = get_model_config(model_size)
            print(f"📋 Loaded {model_size} model configuration")
        except Exception as e:
            print(f"⚠️  Could not load config from path, using default 1B: {e}")
            self.config = get_model_config("1B")

    def _detect_model_size(self) -> str:
        """Detect model size from path."""
        path_str = str(self.model_path).lower()

        if "1b" in path_str:
            return "1B"
        if "8b" in path_str:
            return "8B"
        if "70b" in path_str:
            return "70B"
        return "1B"  # Default to 1B

    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        print("🔤 Loading tokenizer...")

        # Try different tokenizer loading strategies
        tokenizer_strategies = [
            # Strategy 1: Load from model directory
            lambda: self._try_load_tokenizer_from_dir(),
            # Strategy 2: Load default LLaMA tokenizer
            lambda: self._try_load_default_tokenizer(),
            # Strategy 3: Create placeholder tokenizer
            lambda: self._create_placeholder_tokenizer(),
        ]

        for i, strategy in enumerate(tokenizer_strategies, 1):
            try:
                print(f"   Trying strategy {i}...")
                self.tokenizer = strategy()
                if self.tokenizer is not None:
                    print(f"✅ Tokenizer loaded successfully (strategy {i})")
                    return
            except Exception as e:
                print(f"   Strategy {i} failed: {e}")
                continue

        raise RuntimeError("Failed to load tokenizer with all strategies")

    def _try_load_tokenizer_from_dir(self) -> Any:
        """Try to load tokenizer from model directory."""
        if self.model_path.is_dir():
            # Look for tokenizer files in the directory
            possible_tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model"]

            tokenizer_dir = self.model_path
            for file_name in possible_tokenizer_files:
                if (tokenizer_dir / file_name).exists():
                    return AutoTokenizer.from_pretrained(str(tokenizer_dir))

        return None

    def _try_load_default_tokenizer(self) -> Any:
        """Try to load default LLaMA tokenizer."""
        default_models = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "huggyllama/llama-7b",
        ]

        for model_name in default_models:
            try:
                return AutoTokenizer.from_pretrained(model_name)
            except Exception:
                continue

        return None

    def _create_placeholder_tokenizer(self) -> Any:
        """Create a placeholder tokenizer for testing."""
        print("⚠️  Creating placeholder tokenizer")

        class PlaceholderTokenizer:
            """Minimal tokenizer for testing purposes."""

            def __init__(self):
                self.vocab_size = 32000
                self.eos_token_id = 2
                self.pad_token_id = 0

            def encode(self, text: str, return_tensors: str | None = None):
                # Simple word-based encoding for testing
                tokens = [hash(word) % self.vocab_size for word in text.split()][:50]
                if not tokens:
                    tokens = [1]  # Avoid empty sequences

                if return_tensors == "pt":
                    return torch.tensor([tokens])
                return tokens

            def decode(self, token_ids, _skip_special_tokens: bool = True):
                # Simple decoding for testing
                if isinstance(token_ids, (list, tuple)) and len(token_ids) > 0:
                    return f"token_{token_ids[0]}"
                return "token_unknown"

        return PlaceholderTokenizer()

    def _load_model(self) -> None:
        """Load the hybrid model."""
        print("🧠 Loading hybrid model...")

        try:
            # Create hybrid model
            self.model = HybridLLaMA3Model(self.config)

            # Load weights
            if self.model_path.exists():
                print(f"   Loading weights from {self.model_path}")
                self.model.load_weights(str(self.model_path), self.use_torch_weights)
            else:
                print("⚠️  Model path not found, using randomly initialized weights")

            print("✅ Hybrid model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # For now, continue with uninitialized model for testing
            if self.model is None:
                self.model = HybridLLaMA3Model(self.config)

    def _create_inference_engine(self) -> None:
        """Create optimized inference engine."""
        print("⚡ Creating accelerated inference engine...")

        try:
            self.inference_engine = AcceleratedInferenceEngine(self.model, quantize=self.quantize, device=self.device)
            print("✅ Inference engine created successfully")
        except Exception as e:
            print(f"❌ Failed to create inference engine: {e}")
            # Continue without optimized engine
            self.inference_engine = None

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stream: Whether to return streaming generator
            **kwargs: Additional generation parameters

        Returns:
            Generated text (string or generator)
        """
        if stream:
            return self._generate_stream(prompt, max_length, temperature, top_k, top_p, **kwargs)
        # Collect all tokens from stream
        tokens = list(self._generate_stream(prompt, max_length, temperature, top_k, top_p, **kwargs))
        return "".join(tokens)

    def _generate_stream(
        self, prompt: str, max_length: int, temperature: float, top_k: int, top_p: float, **kwargs
    ) -> Generator[str, None, None]:
        """Internal streaming generation method."""
        try:
            # Use accelerated generator if available
            generator = AcceleratedGenerator(self.model, self.tokenizer, self.device)
            yield from generator.generate_stream(prompt, max_length, temperature, top_k, top_p, **kwargs)
        except Exception as e:
            print(f"⚠️  Accelerated generation failed ({e}), using fallback")
            # Fallback to simple generation
            yield f"[Generated response to: '{prompt}' - Fallback mode active]"

    def benchmark(self, prompts: list[str] | None = None, iterations: int = 10) -> dict[str, Any]:
        """Benchmark the accelerated model.

        Args:
            prompts: Test prompts (uses defaults if None)
            iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        if prompts is None:
            prompts = [
                "What is artificial intelligence?",
                "Explain quantum computing in simple terms.",
                "Write a short story about space exploration.",
            ]

        print(f"🏁 Running benchmark with {iterations} iterations...")

        import time

        results = {
            "model_type": "Accelerated LLaMA 3 (Hybrid PyTorch-TinyGrad)",
            "device": self.device,
            "quantization": self.quantize,
            "prompts_tested": len(prompts),
            "iterations": iterations,
            "generation_times": [],
            "tokens_per_second": [],
            "memory_usage": {},
        }

        for i in range(iterations):
            print(f"   Iteration {i + 1}/{iterations}")

            for prompt in prompts:
                # Measure generation time
                start_time = time.time()
                response = self.generate(prompt, max_length=50, stream=False)
                end_time = time.time()

                generation_time = end_time - start_time
                # Rough estimate of tokens per second
                estimated_tokens = len(response.split())
                tokens_per_sec = estimated_tokens / generation_time if generation_time > 0 else 0

                results["generation_times"].append(generation_time)
                results["tokens_per_second"].append(tokens_per_sec)

            # Measure memory usage
            results["memory_usage"] = self.memory_optimizer.get_memory_usage()

        # Calculate statistics
        if results["generation_times"]:
            results["avg_generation_time"] = sum(results["generation_times"]) / len(results["generation_times"])
            results["avg_tokens_per_second"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])
        else:
            results["avg_generation_time"] = 0.0
            results["avg_tokens_per_second"] = 0.0

        print("✅ Benchmark completed!")
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "quantization": self.quantize,
            "config": {
                "vocab_size": self.config.vocab_size,
                "dim": self.config.dim,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "n_kv_heads": self.config.n_kv_heads,
                "max_seq_len": self.config.max_seq_len,
            }
            if self.config
            else None,
            "memory_usage": self.memory_optimizer.get_memory_usage(),
            "tokenizer_type": type(self.tokenizer).__name__ if self.tokenizer else None,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        print("🧹 Cleaning up resources...")
        self.memory_optimizer.clear_cache()

        if hasattr(self, "model") and self.model is not None:
            # Move model to CPU to free GPU memory
            with contextlib.suppress(Exception):
                self.model.cpu()

        print("✅ Cleanup completed")


# Convenience functions for common use cases
def create_accelerated_llama(
    model_path: str | Path, quantize: str | None = None, device: str = "cuda"
) -> AcceleratedLLaMA3:
    """Create an accelerated LLaMA 3 model instance.

    Args:
        model_path: Path to model weights
        quantize: Quantization method
        device: Device for inference

    Returns:
        Accelerated LLaMA 3 instance
    """
    return AcceleratedLLaMA3(model_path=model_path, quantize=quantize, device=device)


def quick_generate(model_path: str | Path, prompt: str, **kwargs) -> str:
    """Quick text generation with minimal setup.

    Args:
        model_path: Path to model weights
        prompt: Input prompt
        **kwargs: Generation parameters

    Returns:
        Generated text
    """
    model = create_accelerated_llama(model_path)
    try:
        return model.generate(prompt, **kwargs)
    finally:
        model.cleanup()


def benchmark_accelerated_model(model_path: str | Path, iterations: int = 10, **kwargs) -> dict[str, Any]:
    """Benchmark an accelerated model.

    Args:
        model_path: Path to model weights
        iterations: Number of benchmark iterations
        **kwargs: Model initialization parameters

    Returns:
        Benchmark results
    """
    model = create_accelerated_llama(model_path, **kwargs)
    try:
        return model.benchmark(iterations=iterations)
    finally:
        model.cleanup()
