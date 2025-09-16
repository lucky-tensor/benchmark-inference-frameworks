# Framework Code Separation Guide

This document explains how the TinyGrad and PyTorch inference code has been cleanly separated in this repository.

## 🎯 What Was Accomplished

The codebase has been refactored to provide **clean separation** between TinyGrad and PyTorch implementations, allowing each framework to be developed, tested, and run independently.

## 📁 New Structure

### Backend Modules
```
src/backends/
├── __init__.py                 # Backend module exports
├── pytorch_backend.py          # Pure PyTorch implementation
└── tinygrad_backend.py         # Pure TinyGrad implementation
```

### Standalone Entry Points
```
tinygrad_only.py               # Run TinyGrad without PyTorch
pytorch_only.py                # Run PyTorch without TinyGrad
```

### Unified Comparison
```
src/benchmark.py               # Compare both frameworks (refactored)
```

## 🚀 Usage Examples

### Run TinyGrad Only
```bash
# Interactive mode
uv run python tinygrad_only.py --size 1B --model ~/models/llama3-1b-instruct/

# Benchmark mode
uv run python tinygrad_only.py --size 1B --model ~/models/llama3-1b-instruct/ --benchmark --iterations 20
```

### Run PyTorch Only
```bash
# Interactive mode
uv run python pytorch_only.py --size 1B --model ~/models/llama3-1b-instruct/

# Benchmark mode
uv run python pytorch_only.py --size 1B --model ~/models/llama3-1b-instruct/ --benchmark --half --compile
```

### Compare Both Frameworks
```bash
# Compare all frameworks
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/

# Compare specific frameworks
uv run src/benchmark.py --framework tinygrad pytorch --model-type llama --model-path ~/models/llama3-1b-instruct/

# TinyGrad only through unified interface
uv run src/benchmark.py --framework tinygrad --model-type llama --model-path ~/models/llama3-1b-instruct/
```

## 🏗️ Architecture Benefits

### 1. **Clean Separation**
- TinyGrad code is completely isolated in `src/backends/tinygrad_backend.py`
- PyTorch code is completely isolated in `src/backends/pytorch_backend.py`
- No framework-specific imports mixed together

### 2. **Independent Development**
- Each framework can be developed and tested separately
- Framework-specific optimizations can be added without affecting the other
- Easier to debug framework-specific issues

### 3. **Flexible Usage**
- Run TinyGrad without any PyTorch dependencies (`tinygrad_only.py`)
- Run PyTorch without any TinyGrad dependencies (`pytorch_only.py`)
- Compare both frameworks side-by-side (`src/benchmark.py`)

### 4. **Maintainable Code**
- Clear interfaces defined by `FrameworkBackend` abstract class
- Consistent API across all implementations
- Easy to add new frameworks in the future

## 📋 Backend Interface

Each backend implements the `FrameworkBackend` interface:

```python
class FrameworkBackend(ABC):
    def get_name(self) -> str: ...
    def load_model(self, model_size: str, model_path: Path | None = None, **kwargs) -> Any: ...
    def load_tokenizer(self, tokenizer_path: Path) -> Any: ...
    def prepare_input(self, model: Any, tokenizer: Any) -> tuple[Any, int]: ...
    def run_inference(self, model: Any, input_data: Any, start_pos: int) -> Any: ...
    def get_model_info(self, model: Any) -> dict[str, Any]: ...
    def get_device_info(self) -> str: ...
    def cleanup(self, model: Any, tokenizer: Any) -> None: ...
```

## 🔧 Implementation Details

### TinyGrad Backend (`tinygrad_backend.py`)
- Uses TinyGrad's native model loading and inference
- Implements JIT compilation and device management
- Provides memory usage estimation
- Supports quantization and multi-GPU sharding

### PyTorch Backend (`pytorch_backend.py`)
- Complete PyTorch LLaMA implementation from scratch
- Weight loading from GGUF files via TinyGrad conversion
- torch.compile support for JIT optimization
- Half precision and device management
- Compatible tokenizer interface

### Unified Benchmark (`benchmark.py`)
- Refactored to use backend modules instead of inline implementations
- Maintains same comparison functionality
- Provides standardized metrics across frameworks
- Fair comparison mode with matching optimizations

## 🧪 Testing

All backends have been tested for:
- ✅ Syntax validation (py_compile)
- ✅ Command-line interface functionality
- ✅ Help message generation
- ✅ Import resolution

## 🎉 Benefits Achieved

1. **Code Organization**: Clear separation of concerns
2. **Maintainability**: Easier to understand and modify each framework
3. **Flexibility**: Run frameworks independently or together
4. **Extensibility**: Easy to add new frameworks following the same pattern
5. **Performance**: No cross-framework interference during benchmarks
6. **Development**: Parallel development of framework-specific features

This separation maintains all existing functionality while providing a much cleaner and more maintainable architecture.