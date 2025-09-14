"""TinyGrad LLaMA Model Implementation

TinyGrad-specific LLaMA implementation with TinyGrad optimizations,
JIT compilation, and kernel fusion.
"""

import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from common.model_interface import BaseLLaMAModel, BaseTokenizer, GenerationConfig
from llama.model_config import build_transformer, resolve_model_path
from tinygrad import Device, Tensor
from tinygrad.nn.state import get_parameters

from .tokenizer import TinyGradTokenizer


class TinyGradLLaMAModel(BaseLLaMAModel):
    """TinyGrad-specific LLaMA model implementation."""

    def __init__(self, model_path: str | Path, model_size: str, **kwargs):
        """Initialize TinyGrad LLaMA model.

        Args:
            model_path: Path to model weights
            model_size: Model size identifier
            **kwargs: TinyGrad-specific options (quantize, shard, etc.)
        """
        super().__init__(model_path, **kwargs)
        self._model_size = model_size
        self._model = None
        self._tokenizer = None
        self._device_config = None
        self._param_bytes = 0

        # TinyGrad-specific options
        self.quantize = kwargs.get("quantize")
        self.shard = kwargs.get("shard", 1)
        self.device_override = kwargs.get("device")

    @property
    def framework_name(self) -> str:
        return "tinygrad"

    @property
    def model_size(self) -> str:
        return self._model_size

    def load_model(self) -> None:
        """Load TinyGrad LLaMA model with TinyGrad-specific optimizations."""
        start_time = time.time()

        # Resolve model path using TinyGrad's path resolution
        resolved_path = resolve_model_path(self.model_path, self._model_size, False)
        if resolved_path is None:
            raise ValueError(f"Could not resolve model path: {self.model_path}")

        # Update model_path to the resolved path for tokenizer loading
        self.model_path = resolved_path

        # Configure device placement for TinyGrad
        if self.device_override:
            self._device_config = self.device_override
        elif self.shard > 1:
            # Multi-device sharding for TinyGrad
            self._device_config = tuple(f"{Device.DEFAULT}:{i}" for i in range(self.shard))
        else:
            self._device_config = Device.DEFAULT

        print(f"🔥 Loading TinyGrad LLaMA {self._model_size} with device config: {self._device_config}")

        # Use TinyGrad's build_transformer with TinyGrad-specific optimizations
        self._model = build_transformer(
            resolved_path, model_size=self._model_size, quantize=self.quantize, device=self._device_config
        )

        # Calculate parameter bytes for TinyGrad metrics
        self._param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(self._model))

        # Record loading time
        self._metrics.model_load_time = time.time() - start_time
        self._metrics.model_memory_gb = self._param_bytes / (1024**3)

        print(f"✅ TinyGrad LLaMA model loaded in {self._metrics.model_load_time:.2f}s")
        print(f"📊 Model parameters: {self._param_bytes / 1e9:.1f}B ({self._metrics.model_memory_gb:.2f} GB)")

    def load_tokenizer(self) -> BaseTokenizer:
        """Load TinyGrad-compatible tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        start_time = time.time()

        # Find tokenizer.model file
        if self.model_path.is_dir():
            tokenizer_path = self.model_path / "tokenizer.model"
        else:
            tokenizer_path = self.model_path.parent / "tokenizer.model"

        if not tokenizer_path.exists():
            raise ValueError(f"Tokenizer not found at: {tokenizer_path}")

        self._tokenizer = TinyGradTokenizer(tokenizer_path)
        self._metrics.tokenizer_load_time = time.time() - start_time

        print(f"✅ TinyGrad tokenizer loaded in {self._metrics.tokenizer_load_time:.2f}s")
        return self._tokenizer

    def generate(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        """Generate text using TinyGrad with streaming.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Generated tokens as strings
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self._tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")

        from common import generation
        from tinygrad import GlobalCounters
        from tinygrad.helpers import Profiling, Timing

        # Set TinyGrad generation parameters
        generation.TEMPERATURE = config.temperature

        # Encode prompt
        input_tokens = self._tokenizer.encode(prompt)

        # Initialize generation
        from common.generation import prefill

        start_pos = prefill(self._model, input_tokens)
        last_tok = input_tokens[-1]

        # Track metrics
        first_token = True
        generation_start = time.time()
        tokens_generated = 0

        # TinyGrad generation loop with JIT and profiling
        for _ in range(config.max_length):
            # Reset TinyGrad counters for per-token timing
            GlobalCounters.reset()

            # TinyGrad-specific timing and profiling
            with Profiling(enabled=config.framework_options.get("profile", False)):
                with Timing(
                    "total ",
                    enabled=config.framework_options.get("timing", False),
                    on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, "
                    f"param {self._param_bytes / x:.2f} GB/s",
                ):
                    # TinyGrad model inference with JIT compilation
                    tok = self._model(
                        Tensor([[last_tok]], device=self._device_config),
                        start_pos,
                        generation.TEMPERATURE,
                        generation.TOP_K,
                        generation.TOP_P,
                        generation.ALPHA_F,
                        generation.ALPHA_P,
                    )
                tok = tok.item()

            # Record first token time (includes JIT compilation)
            if first_token:
                self._metrics.first_token_time = time.time() - generation_start
                first_token = False

            # Check for end-of-sequence
            if tok == self._tokenizer.eos_token_id:
                break

            # Decode and yield token
            decoded = self._tokenizer.decode([tok])
            yield decoded

            # Update for next iteration
            start_pos += 1
            last_tok = tok
            tokens_generated += 1

        # Update final metrics
        total_time = time.time() - generation_start
        self._metrics.total_inference_time = total_time
        self._metrics.tokens_generated = tokens_generated
        if total_time > 0:
            self._metrics.tokens_per_second = tokens_generated / total_time

        # Add TinyGrad-specific metrics
        self._metrics.framework_specific.update(
            {
                "jit_compilation_time": self._metrics.first_token_time,
                "global_ops": getattr(GlobalCounters, "global_ops", 0),
                "global_mem": getattr(GlobalCounters, "global_mem", 0),
                "device_config": str(self._device_config),
            }
        )

    def generate_single(self, prompt: str, config: GenerationConfig) -> str:
        """Generate complete text from prompt using TinyGrad.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Complete generated text
        """
        tokens = list(self.generate(prompt, config))
        return "".join(tokens)

    def get_model_config(self) -> dict[str, Any]:
        """Get TinyGrad LLaMA model configuration.

        Returns:
            Dictionary with model architecture config
        """
        # Use the common model configs but return the specific values
        from common.model_configs import get_llama_config

        config = get_llama_config(self._model_size)

        return {
            "framework": self.framework_name,
            "model_type": self.model_type,
            "model_size": self._model_size,
            "vocab_size": config.vocab_size,
            "dim": config.dim,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "hidden_dim": config.hidden_dim,
            "max_seq_len": config.max_seq_len,
            "quantization": self.quantize,
            "device_sharding": self.shard,
            "parameter_bytes": self._param_bytes,
        }

    def get_memory_usage(self) -> dict[str, float]:
        """Get TinyGrad-specific memory usage.

        Returns:
            Memory usage in GB
        """
        try:
            from tinygrad import GlobalCounters

            # TinyGrad-specific memory tracking
            memory_info = {
                "model_memory_gb": self._param_bytes / (1024**3) if self._param_bytes else 0.0,
                "global_mem_gb": getattr(GlobalCounters, "global_mem", 0) / (1024**3),
            }

            # Try to get device-specific memory if available
            if hasattr(Device, "DEFAULT") and "CUDA" in str(Device.DEFAULT):
                try:
                    import torch

                    if torch.cuda.is_available():
                        memory_info.update(
                            {
                                "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                                "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                            }
                        )
                except:
                    pass

            return memory_info

        except Exception:
            return {"model_memory_gb": self._param_bytes / (1024**3) if self._param_bytes else 0.0}

    def cleanup(self) -> None:
        """Clean up TinyGrad model resources."""
        print("🧹 Cleaning up TinyGrad model resources...")

        # Clear TinyGrad caches if available
        try:
            from tinygrad import GlobalCounters

            GlobalCounters.reset()
        except:
            pass

        # Clear model references
        self._model = None
        self._tokenizer = None

        print("✅ TinyGrad cleanup completed")
