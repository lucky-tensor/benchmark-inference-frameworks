#!/usr/bin/env python3
"""
PyTorch framework executor implementations.
"""

import sys
from pathlib import Path
from typing import Any

from benchmark import BenchRun, FrameworkExecutor


class PyTorchBaseExecutor(FrameworkExecutor):
    """Base executor for PyTorch variants with common functionality."""

    def __init__(self, variant: str = "unoptimized"):
        """
        Initialize PyTorch executor.

        Args:
            variant: PyTorch variant ("unoptimized", "inductor", "eager", "aot_eager")
        """
        self.variant = variant

    def get_framework_name(self) -> str:
        if self.variant == "unoptimized":
            return "pytorch-unoptimized"
        return f"pytorch-{self.variant}"

    def load_model(self, bench_run: BenchRun) -> Any:
        """Load PyTorch model."""
        print(f"🔧 [PYTORCH-{self.variant.upper()}] Loading PyTorch-specific model implementation...")

        import torch

        print(f"🔧 [PYTORCH-{self.variant.upper()}] PyTorch version: {torch.__version__}")

        # Add necessary directories to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent.parent))
        sys.path.insert(0, str(current_dir.parent.parent / "llama"))  # For extra module imports

        from .backend import MODEL_CONFIGS, PyTorchLLaMA, load_pytorch_weights_from_gguf

        # Create model
        model_size = self._extract_model_size(bench_run.model_id)
        config = MODEL_CONFIGS[model_size]
        print(f"🔧 [PYTORCH-{self.variant.upper()}] Using PyTorch backend for {model_size} model")
        model = PyTorchLLaMA(**config)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"🔧 [PYTORCH-{self.variant.upper()}] Model moved to device: {device}")

        # Load weights if provided
        if bench_run.model_path and bench_run.model_path.is_dir():
            gguf_files = list(bench_run.model_path.glob("*.gguf"))
            if gguf_files:
                weight_path = gguf_files[0]
                print(f"🔧 [PYTORCH-{self.variant.upper()}] Loading weights from: {weight_path.name}")
                load_pytorch_weights_from_gguf(model, weight_path)

        model.eval()
        print(
            f"🔧 [PYTORCH-{self.variant.upper()}] Loaded model type: {type(model).__name__} from module: {type(model).__module__}"
        )
        return model

    def compile_model(self, bench_run: BenchRun) -> None:
        """Compile PyTorch model based on variant."""
        if self.variant == "unoptimized":
            print(f"🔧 [PYTORCH-{self.variant.upper()}] Skipping compilation for unoptimized variant")
            return

        import torch

        model = bench_run.model_instance
        if not hasattr(torch, "compile"):
            print("⚠️  torch.compile not available, skipping compilation")
            return

        print(f"🔧 [PYTORCH-{self.variant.upper()}] Compiling PyTorch model with {self.variant} backend...")
        print(f"🔧 [PYTORCH-{self.variant.upper()}] This will use PyTorch-native torch.compile() functionality")

        # Configure Inductor settings for stability
        if self.variant == "inductor":
            self._configure_inductor_settings()

        try:
            if self.variant == "eager":
                model = torch.compile(model, backend="eager", mode="default", dynamic=True)
            elif self.variant == "aot_eager":
                model = torch.compile(model, backend="aot_eager", mode="default", dynamic=True)
            elif self.variant == "inductor":
                model = torch.compile(model, backend="inductor", mode="default", dynamic=True, fullgraph=False)
            else:
                # Generic compilation
                model = torch.compile(model, mode="default", dynamic=True)

            bench_run.model_instance = model
            print(f"✅ [PYTORCH-{self.variant.upper()}] Successfully compiled with {self.variant}")
            print(f"🔧 [PYTORCH-{self.variant.upper()}] Model is now using PyTorch {self.variant} compilation")

        except Exception as e:
            print(f"❌ [PYTORCH-{self.variant.upper()}] Compilation failed: {e}")
            print(f"🔧 [PYTORCH-{self.variant.upper()}] Continuing with uncompiled PyTorch model")
            # Continue with uncompiled model

    def load_tokenizer(self, bench_run: BenchRun) -> Any:
        """Load PyTorch tokenizer."""
        # Add necessary directories to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent.parent))
        sys.path.insert(0, str(current_dir.parent.parent / "llama"))  # For extra module imports
        from .backend import PyTorchTokenizer

        # Determine tokenizer path
        if bench_run.model_path and bench_run.model_path.is_dir():
            tokenizer_path = bench_run.model_path / "tokenizer.model"
        elif bench_run.model_path:
            tokenizer_path = bench_run.model_path.parent / "tokenizer.model"
        else:
            # Default path
            tokenizer_path = Path.home() / "models" / "llama3-1b-instruct" / "tokenizer.model"

        return PyTorchTokenizer(str(tokenizer_path))

    def prepare_input(self, bench_run: BenchRun) -> tuple[Any, int]:
        """Prepare input for PyTorch inference."""
        import torch

        # Add necessary directories to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent.parent))
        sys.path.insert(0, str(current_dir.parent.parent / "llama"))  # For extra module imports

        def encode_message(role: str, content: str, tokenizer) -> list[int]:
            if role == "user":
                return [
                    tokenizer.special_tokens["<|start_header_id|>"],
                    *tokenizer.encode("user"),
                    tokenizer.special_tokens["<|end_header_id|>"],
                    *tokenizer.encode("\n\n" + content.strip()),
                    tokenizer.special_tokens["<|eot_id|>"],
                ]
            return []

        def encode_role(role: str, tokenizer) -> list[int]:
            return [
                tokenizer.special_tokens["<|start_header_id|>"],
                *tokenizer.encode(role),
                tokenizer.special_tokens["<|end_header_id|>"],
                *tokenizer.encode("\n\n"),
            ]

        tokenizer = bench_run.tokenizer_instance
        toks = [tokenizer.bos_id, *encode_message("user", "Hello.", tokenizer), *encode_role("assistant", tokenizer)]

        # For PyTorch, we need to do prefill manually
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store prefill tokens for later use in inference
        prefill_tokens = torch.tensor([toks[:-1]], device=device, dtype=torch.long)
        bench_run.framework_options["_prefill_tokens"] = prefill_tokens
        bench_run.framework_options["_prefilled"] = False

        return toks[-1], len(toks) - 1

    def run_inference(self, bench_run: BenchRun, input_data: Any, start_pos: int) -> Any:
        """Run PyTorch inference."""
        import torch

        # Add necessary directories to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent.parent))
        sys.path.insert(0, str(current_dir.parent.parent / "llama"))  # For extra module imports
        # PyTorch generation parameters (independent of TinyGrad)
        TEMPERATURE = 0.7
        TOP_K = 10
        TOP_P = 0.8
        ALPHA_F = 0.0
        ALPHA_P = 0.0

        model = bench_run.model_instance
        device = next(model.parameters()).device

        # First iteration needs prefill
        if not bench_run.framework_options.get("_prefilled", False):
            prefill_tokens = bench_run.framework_options.get("_prefill_tokens")
            if prefill_tokens is not None:
                with torch.no_grad():
                    try:
                        # Ensure prefill tokens are on the same device as the model
                        if prefill_tokens.device != device:
                            prefill_tokens = prefill_tokens.to(device)
                            bench_run.framework_options["_prefill_tokens"] = prefill_tokens

                        _ = model(prefill_tokens, 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
                    except Exception as e:
                        print(f"❌ First inference failed: {type(e).__name__}: {e}")

                # Synchronize after first compilation/execution
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                bench_run.framework_options["_prefilled"] = True

        # Run inference
        with torch.no_grad():
            input_tensor = torch.tensor([[input_data]], device=device, dtype=torch.long)
            print(
                f"🔧 [PYTORCH-{self.variant.upper()}] Running PyTorch {self.variant} inference with model type: {type(model).__name__}"
            )
            tok = model(input_tensor, start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)

            if hasattr(tok, "item"):
                return tok.item()
            return int(tok[0])

    def get_model_info(self, bench_run: BenchRun) -> dict[str, Any]:
        """Get PyTorch model information."""
        import torch

        model = bench_run.model_instance

        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate memory usage
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_memory = (param_size + buffer_size) / (1024**3)

        # Get peak memory if available
        peak_memory = model_memory
        if torch.cuda.is_available():
            peak_memory = max(model_memory, torch.cuda.memory_reserved() / (1024**3))

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "loaded_parameters": total_params,
            "model_memory_gb": model_memory,
            "peak_memory_gb": peak_memory,
            "precision": "torch.float32",
            "device": str(next(model.parameters()).device),
        }

    def cleanup(self, bench_run: BenchRun) -> None:
        """Clean up PyTorch resources."""
        import gc

        import torch

        print("🧹 Cleaning up PyTorch resources...")

        # Clear model references
        model = bench_run.model_instance
        if model and hasattr(model, "cpu"):
            model.cpu()

        bench_run.model_instance = None
        bench_run.tokenizer_instance = None

        # Clear framework options
        bench_run.framework_options.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_cleared = torch.cuda.memory_reserved() / (1024**3)
            print(f"   Cleared {memory_cleared:.2f} GB CUDA memory")

        print("✅ PyTorch cleanup completed")

    def _extract_model_size(self, model_id: str) -> str:
        """Extract model size from model ID."""
        if "-1b" in model_id.lower():
            return "1B"
        if "-3b" in model_id.lower():
            return "3B"
        if "-7b" in model_id.lower():
            return "7B"
        if "-8b" in model_id.lower():
            return "8B"
        return "1B"  # Default

    def _configure_inductor_settings(self) -> None:
        """Configure Inductor settings for stability."""
        import os

        # Set environment variables for stable Triton compilation
        os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
        os.environ["TORCH_COMPILE_DEBUG"] = "0"

        try:
            import torch._inductor.config as inductor_config

            # Only set configuration options that are confirmed to exist
            inductor_config.triton.unique_kernel_names = True
            inductor_config.coordinate_descent_tuning = True
            inductor_config.fallback_random = True

            # Disable CUDAGraphs which can cause device issues
            if hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
                print("   ✓ Disabled CUDAGraphs for stability")

        except Exception as e:
            print(f"   Warning: Could not configure inductor settings: {e}")

        # Device affinity settings
        import torch

        if torch.cuda.is_available():
            # Ensure consistent device placement
            torch.backends.cudnn.deterministic = False  # Allow cudnn optimizations
            torch.backends.cudnn.benchmark = True  # Enable cudnn benchmarking
            # Enable TensorFloat32 for better performance
            torch.set_float32_matmul_precision("high")


# Specific PyTorch variant executors
class PyTorchUnoptimizedExecutor(PyTorchBaseExecutor):
    """PyTorch without any compilation optimizations."""

    def __init__(self):
        super().__init__("unoptimized")


class PyTorchInductorExecutor(PyTorchBaseExecutor):
    """PyTorch with Inductor compilation."""

    def __init__(self):
        super().__init__("inductor")


class PyTorchEagerExecutor(PyTorchBaseExecutor):
    """PyTorch with Eager compilation."""

    def __init__(self):
        super().__init__("eager")


class PyTorchAOTEagerExecutor(PyTorchBaseExecutor):
    """PyTorch with AOT Eager compilation."""

    def __init__(self):
        super().__init__("aot_eager")
