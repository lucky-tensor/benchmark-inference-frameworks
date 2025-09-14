"""TinyGrad GPT Model Implementation

TinyGrad-specific GPT implementation with TinyGrad optimizations,
JIT compilation, and kernel fusion.
"""

import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from common.model_interface import BaseGPTModel, BaseTokenizer, GenerationConfig
from tinygrad import Device

from .tokenizer import TinyGradTokenizer


class TinyGradGPTModel(BaseGPTModel):
    """TinyGrad-specific GPT model implementation."""

    def __init__(self, model_path: str | Path, model_size: str, **kwargs):
        """Initialize TinyGrad GPT model.

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
        """Load TinyGrad GPT model with TinyGrad-specific optimizations."""
        time.time()

        # Configure device placement for TinyGrad
        if self.device_override:
            self._device_config = self.device_override
        elif self.shard > 1:
            # Multi-device sharding for TinyGrad
            self._device_config = tuple(f"{Device.DEFAULT}:{i}" for i in range(self.shard))
        else:
            self._device_config = Device.DEFAULT

        print(f"🔥 Loading TinyGrad GPT {self._model_size} with device config: {self._device_config}")

        # TODO: Implement TinyGrad-specific GPT model loading
        # This would use TinyGrad's GPT implementation when available
        raise NotImplementedError("TinyGrad GPT model loading not yet implemented")

    def load_tokenizer(self) -> BaseTokenizer:
        """Load TinyGrad-compatible tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        start_time = time.time()

        # Find tokenizer files for GPT
        if self.model_path.is_dir():
            # Look for various GPT tokenizer formats
            tokenizer_files = ["tokenizer.model", "vocab.bpe", "merges.txt", "tokenizer.json"]
            tokenizer_path = None
            for file in tokenizer_files:
                candidate = self.model_path / file
                if candidate.exists():
                    tokenizer_path = candidate
                    break
        else:
            tokenizer_path = self.model_path.parent / "tokenizer.model"

        if tokenizer_path is None or not tokenizer_path.exists():
            raise ValueError(f"GPT tokenizer not found in: {self.model_path}")

        self._tokenizer = TinyGradTokenizer(tokenizer_path)
        self._metrics.tokenizer_load_time = time.time() - start_time

        print(f"✅ TinyGrad GPT tokenizer loaded in {self._metrics.tokenizer_load_time:.2f}s")
        return self._tokenizer

    def generate(self, _prompt: str, _config: GenerationConfig) -> Generator[str, None, None]:
        """Generate text using TinyGrad GPT with streaming.

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

        # TODO: Implement TinyGrad GPT generation with JIT compilation
        # This would follow similar pattern to LLaMA but for GPT architecture
        raise NotImplementedError("TinyGrad GPT generation not yet implemented")

    def generate_single(self, prompt: str, config: GenerationConfig) -> str:
        """Generate complete text from prompt using TinyGrad GPT.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Complete generated text
        """
        tokens = list(self.generate(prompt, config))
        return "".join(tokens)

    def get_model_config(self) -> dict[str, Any]:
        """Get TinyGrad GPT model configuration.

        Returns:
            Dictionary with model architecture config
        """
        # GPT model sizes and configurations
        gpt_configs = {
            "124M": {"vocab_size": 50257, "dim": 768, "n_layers": 12, "n_heads": 12},
            "355M": {"vocab_size": 50257, "dim": 1024, "n_layers": 24, "n_heads": 16},
            "774M": {"vocab_size": 50257, "dim": 1280, "n_layers": 36, "n_heads": 20},
            "1558M": {"vocab_size": 50257, "dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        config = gpt_configs.get(self._model_size, gpt_configs["124M"])

        return {
            "framework": self.framework_name,
            "model_type": self.model_type,
            "model_size": self._model_size,
            "vocab_size": config["vocab_size"],
            "dim": config["dim"],
            "n_layers": config["n_layers"],
            "n_heads": config["n_heads"],
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
        print("🧹 Cleaning up TinyGrad GPT model resources...")

        # Clear TinyGrad caches if available
        try:
            from tinygrad import GlobalCounters

            GlobalCounters.reset()
        except:
            pass

        # Clear model references
        self._model = None
        self._tokenizer = None

        print("✅ TinyGrad GPT cleanup completed")
