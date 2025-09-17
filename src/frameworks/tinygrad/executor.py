#!/usr/bin/env python3
"""
TinyGrad framework executor implementation.
"""

import os
import sys
from pathlib import Path
from typing import Any

from benchmark import BenchRun, FrameworkExecutor


class TinyGradExecutor(FrameworkExecutor):
    """Executor for TinyGrad framework."""

    def __init__(self):
        """Initialize TinyGrad executor with optimal environment settings."""
        super().__init__()
        # Set TinyGrad environment variables for better performance
        # Note: BEAM=1 provides 4x speed improvement but has very long compile times
        # Start with moderate optimizations for now
        os.environ["JIT"] = "1"          # Enable JIT compilation
        os.environ["CUDA"] = "1"         # Enable CUDA if available
        os.environ["FASTMATH"] = "1"     # Enable fast math optimizations
        os.environ["OPT"] = "2"          # Maximum optimization level
        os.environ["CLCACHE"] = "1"      # Enable kernel cache
        os.environ["CUDACACHE"] = "1"    # Enable CUDA kernel cache
        # TODO: Add BEAM=1 after optimizing for compilation time

    def get_framework_name(self) -> str:
        return "tinygrad"

    def load_model(self, bench_run: BenchRun) -> Any:
        """Load TinyGrad model."""
        print("🔧 [TINYGRAD] Loading TinyGrad-specific model implementation...")

        # Add necessary directories to path for imports
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir.parent.parent))
        sys.path.insert(0, str(current_dir.parent.parent / "llama"))  # For extra module imports

        from frameworks.tinygrad.backends.tinygrad_backend import get_tinygrad_model

        model_size = self._extract_model_size(bench_run.model_id)
        print(f"🔧 [TINYGRAD] Using TinyGrad backend for {model_size} model")
        model = get_tinygrad_model(model_size, bench_run.model_path)
        print(f"🔧 [TINYGRAD] Loaded model type: {type(model).__name__} from module: {type(model).__module__}")
        return model

    def load_tokenizer(self, bench_run: BenchRun) -> Any:
        """Load TinyGrad tokenizer."""
        from frameworks.tinygrad.backends.tinygrad_backend import get_tinygrad_tokenizer

        # Determine tokenizer path
        if bench_run.model_path.is_dir():
            tokenizer_path = bench_run.model_path / "tokenizer.model"
        else:
            tokenizer_path = bench_run.model_path.parent / "tokenizer.model"

        return get_tinygrad_tokenizer(tokenizer_path)

    def prepare_input(self, bench_run: BenchRun) -> tuple[Any, int]:
        """Prepare input for TinyGrad inference."""
        from frameworks.tinygrad.backends.tinygrad_backend import prepare_tinygrad_input

        return prepare_tinygrad_input(bench_run.model_instance, bench_run.tokenizer_instance)

    def run_inference(self, bench_run: BenchRun, input_data: Any, start_pos: int) -> Any:
        """Run TinyGrad inference."""
        from frameworks.tinygrad.backends.tinygrad_backend import run_tinygrad_inference

        print(
            f"🔧 [TINYGRAD] Running TinyGrad-specific inference with model type: {type(bench_run.model_instance).__name__}"
        )
        return run_tinygrad_inference(bench_run.model_instance, input_data, start_pos)

    def get_model_info(self, bench_run: BenchRun) -> dict[str, Any]:
        """Get TinyGrad model information."""
        from frameworks.tinygrad.backends.tinygrad_backend import get_tinygrad_model_info

        return get_tinygrad_model_info(bench_run.model_instance)

    def cleanup(self, bench_run: BenchRun) -> None:
        """Clean up TinyGrad resources."""
        import gc

        print("🧹 Cleaning up TinyGrad resources...")

        # Clear model and tokenizer references
        bench_run.model_instance = None
        bench_run.tokenizer_instance = None

        # Force garbage collection
        gc.collect()

        # TinyGrad-specific cleanup
        try:
            from tinygrad import Device

            if hasattr(Device, "DEFAULT") and hasattr(Device.DEFAULT, "synchronize"):
                Device.DEFAULT.synchronize()
            if hasattr(Device, "DEFAULT") and hasattr(Device.DEFAULT, "empty_cache"):
                Device.DEFAULT.empty_cache()
        except Exception as e:
            print(f"   TinyGrad device cleanup failed: {e}")

        print("✅ TinyGrad cleanup completed")

    def _extract_model_size(self, model_id: str) -> str:
        """Extract model size from model ID (e.g., 'llama3-1b' -> '1B')."""
        if "-1b" in model_id.lower():
            return "1B"
        if "-3b" in model_id.lower():
            return "3B"
        if "-7b" in model_id.lower():
            return "7B"
        if "-8b" in model_id.lower():
            return "8B"
        return "1B"  # Default
