"""PyTorch-TinyGrad Bridge Module

This module provides zero-copy tensor conversion between PyTorch and TinyGrad tensors,
enabling hybrid execution that leverages PyTorch's ecosystem with TinyGrad's optimizations.
"""

from typing import Any

import torch
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class TensorBridge:
    """Provides zero-copy conversion between PyTorch and TinyGrad tensors."""

    @staticmethod
    def torch_to_tinygrad(torch_tensor: torch.Tensor) -> Tensor:
        """Convert PyTorch tensor to TinyGrad with zero-copy when possible.

        Args:
            torch_tensor: PyTorch tensor to convert

        Returns:
            TinyGrad tensor sharing the same memory when possible
        """
        # Map PyTorch dtype to TinyGrad dtype
        dtype_map = {
            torch.float32: dtypes.float32,
            torch.float16: dtypes.float16,
            torch.int32: dtypes.int32,
            torch.int64: dtypes.int64,
            torch.int8: dtypes.int8,
            torch.uint8: dtypes.uint8,
            torch.bool: dtypes.bool,
        }

        tg_dtype = dtype_map.get(torch_tensor.dtype)
        if tg_dtype is None:
            raise ValueError(f"Unsupported dtype for conversion: {torch_tensor.dtype}")

        # Ensure tensor is contiguous
        if not torch_tensor.is_contiguous():
            torch_tensor = torch_tensor.contiguous()

        # Determine device
        device = "CUDA" if torch_tensor.is_cuda else "CPU"
        if torch_tensor.device.type == "mps":
            # For MPS (Apple Silicon), we need to copy to CPU first
            torch_tensor = torch_tensor.cpu()
            device = "CPU"

        # Create TinyGrad tensor from raw data pointer
        try:
            # Use TinyGrad's from_blob method for zero-copy conversion
            return Tensor.from_blob(torch_tensor.data_ptr(), torch_tensor.shape, dtype=tg_dtype, device=device)
        except Exception as e:
            # Fallback to copying data if zero-copy fails
            print(f"Warning: Zero-copy conversion failed ({e}), falling back to copy")
            return Tensor(torch_tensor.cpu().numpy(), dtype=tg_dtype, device=device)

    @staticmethod
    def tinygrad_to_torch(tg_tensor: Tensor) -> torch.Tensor:
        """Convert TinyGrad tensor to PyTorch.

        Args:
            tg_tensor: TinyGrad tensor to convert

        Returns:
            PyTorch tensor
        """
        # Map TinyGrad dtype to PyTorch dtype
        dtype_map = {
            dtypes.float32: torch.float32,
            dtypes.float16: torch.float16,
            dtypes.int32: torch.int32,
            dtypes.int64: torch.int64,
            dtypes.int8: torch.int8,
            dtypes.uint8: torch.uint8,
            dtypes.bool: torch.bool,
        }

        torch_dtype = dtype_map.get(tg_tensor.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype for conversion: {tg_tensor.dtype}")

        # Convert to numpy first, then to PyTorch
        numpy_array = tg_tensor.numpy()
        torch_tensor = torch.from_numpy(numpy_array).to(torch_dtype)

        # Move to appropriate device
        if tg_tensor.device.startswith("CUDA"):
            torch_tensor = torch_tensor.cuda()

        return torch_tensor

    @staticmethod
    def sync_device(torch_tensor: torch.Tensor) -> None:
        """Synchronize device before operations to ensure data consistency.

        Args:
            torch_tensor: PyTorch tensor whose device to synchronize
        """
        if torch_tensor.device.type == "mps":
            torch.mps.synchronize()
        elif torch_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        # CPU tensors don't need synchronization

    @staticmethod
    def get_device_info(tensor: torch.Tensor | Tensor) -> dict[str, Any]:
        """Get device information for a tensor.

        Args:
            tensor: PyTorch or TinyGrad tensor

        Returns:
            Dictionary with device information
        """
        if isinstance(tensor, torch.Tensor):
            return {
                "framework": "PyTorch",
                "device_type": tensor.device.type,
                "device_index": tensor.device.index,
                "is_cuda": tensor.is_cuda,
                "dtype": tensor.dtype,
                "shape": tuple(tensor.shape),
                "is_contiguous": tensor.is_contiguous(),
            }
        if isinstance(tensor, Tensor):
            return {
                "framework": "TinyGrad",
                "device": tensor.device,
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "realized": hasattr(tensor, "lazydata") and tensor.lazydata is not None,
            }
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


class HybridModelLoader:
    """Model loader that supports both PyTorch and TinyGrad weight formats."""

    def __init__(self, model_path: str, use_torch_weights: bool = False):
        """Initialize hybrid model loader.

        Args:
            model_path: Path to model weights
            use_torch_weights: Whether to load with PyTorch for better HuggingFace compatibility
        """
        self.model_path = model_path
        self.use_torch_weights = use_torch_weights

    def load_weights(self) -> dict[str, Any]:
        """Load model weights using the appropriate method.

        Returns:
            Dictionary containing model weights
        """
        if self.use_torch_weights:
            # Load with PyTorch for better HuggingFace compatibility
            try:
                torch_model = torch.load(self.model_path, map_location="cpu")
                return self.convert_torch_to_tinygrad_weights(torch_model)
            except Exception as e:
                print(f"Failed to load with PyTorch: {e}")
                print("Falling back to TinyGrad loading...")
                return self._load_with_tinygrad()
        else:
            # Use existing TinyGrad loading
            return self._load_with_tinygrad()

    def _load_with_tinygrad(self) -> dict[str, Any]:
        """Load weights using TinyGrad's loading mechanism."""
        # Import here to avoid circular imports
        from llama.model_config import load_weights

        return load_weights(str(self.model_path))

    def convert_torch_to_tinygrad_weights(self, torch_weights: dict[str, torch.Tensor]) -> dict[str, Tensor]:
        """Convert PyTorch model weights to TinyGrad tensors.

        Args:
            torch_weights: Dictionary of PyTorch tensors

        Returns:
            Dictionary of TinyGrad tensors
        """
        tg_weights = {}
        for name, weight in torch_weights.items():
            if isinstance(weight, torch.Tensor):
                tg_weights[name] = TensorBridge.torch_to_tinygrad(weight)
            else:
                # Keep non-tensor values as is
                tg_weights[name] = weight

        return tg_weights


class MemoryOptimizer:
    """Optimize memory usage across PyTorch-TinyGrad boundary."""

    @staticmethod
    def optimize_model_sharding(model: Any, device_count: int) -> Any:
        """Implement intelligent sharding across devices.

        Args:
            model: Model to shard
            device_count: Number of devices available

        Returns:
            Sharded model
        """
        # Placeholder for device sharding implementation
        # This will be implemented based on the specific model architecture
        print(f"Optimizing model sharding across {device_count} devices")
        return model

    @staticmethod
    def dynamic_quantization(model: Any, target_precision: str = "int8") -> Any:
        """Apply dynamic quantization using TinyGrad's quantization.

        Args:
            model: Model to quantize
            target_precision: Target precision ('int8', 'float16', etc.)

        Returns:
            Quantized model
        """
        # Placeholder for quantization implementation
        print(f"Applying dynamic quantization to {target_precision}")
        return model

    @staticmethod
    def estimate_memory_usage(_model: Any) -> dict[str, float]:
        """Estimate memory usage for the model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with memory usage estimates in GB
        """
        # This would analyze the model and estimate memory usage
        return {
            "model_weights": 0.0,
            "activation_memory": 0.0,
            "overhead": 0.0,
            "total": 0.0,
        }


# Helper functions for common operations
def ensure_tensor_compatibility(
    tensor1: torch.Tensor | Tensor, tensor2: torch.Tensor | Tensor
) -> tuple[torch.Tensor | Tensor, torch.Tensor | Tensor]:
    """Ensure two tensors are compatible for operations.

    Args:
        tensor1: First tensor
        tensor2: Second tensor

    Returns:
        Tuple of compatible tensors (both PyTorch or both TinyGrad)
    """
    # If both are the same type, return as-is
    if type(tensor1) == type(tensor2):
        return tensor1, tensor2

    # Convert to the same framework (prefer TinyGrad for computation)
    if isinstance(tensor1, torch.Tensor):
        tensor1 = TensorBridge.torch_to_tinygrad(tensor1)
    if isinstance(tensor2, torch.Tensor):
        tensor2 = TensorBridge.torch_to_tinygrad(tensor2)

    return tensor1, tensor2


def validate_conversion(
    original: torch.Tensor | Tensor,
    converted: torch.Tensor | Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Validate that tensor conversion preserves data integrity.

    Args:
        original: Original tensor
        converted: Converted tensor
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if conversion is valid within tolerance
    """
    import numpy as np

    # Convert both to numpy for comparison
    orig_np = original.cpu().detach().numpy() if isinstance(original, torch.Tensor) else original.numpy()

    conv_np = converted.cpu().detach().numpy() if isinstance(converted, torch.Tensor) else converted.numpy()

    # Check shape match
    if orig_np.shape != conv_np.shape:
        print(f"Shape mismatch: {orig_np.shape} vs {conv_np.shape}")
        return False

    # Check value match within tolerance
    if not np.allclose(orig_np, conv_np, rtol=rtol, atol=atol):
        print(f"Value mismatch beyond tolerance (rtol={rtol}, atol={atol})")
        return False

    return True
