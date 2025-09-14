# Architecture Overview

Technical implementation details and system architecture of the TinyGrad Inference Engine Benchmark Suite.

## System Architecture

### High-Level Design

The system uses a unified inference engine with pluggable framework backends, ensuring consistent performance measurements across different frameworks while eliminating code duplication.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface Layer                      │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Interface    │  Interactive Mode  │  Benchmark Mode  │  API    │
├─────────────────────────────────────────────────────────────────────┤
│                      Unified Inference Engine                       │
├─────────────────────────────────────────────────────────────────────┤
│  TinyGrad Backend │  PyTorch Backend  │  Hybrid Backend  │ Future  │
├─────────────────────────────────────────────────────────────────────┤
│        Model Loading      │     Tokenization     │   Memory Mgmt    │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Unified Inference Engine (`src/common/inference_engine.py`)
**Purpose**: Eliminates code duplication and ensures consistent metrics across all usage modes

**Key Features**:
- **Framework Abstraction**: Uniform interface for all backends
- **Consistent Metrics**: Same performance measurement code for interactive, benchmark, and API modes
- **Memory Management**: Unified GPU/CPU memory tracking
- **Error Handling**: Graceful fallbacks and resource cleanup

```python
class InferenceEngine:
    def __init__(self, backend: FrameworkBackend):
        self.backend = backend
        self.metrics = PerformanceMetrics()
        self.memory_tracker = MemoryTracker()

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        # Consistent generation logic across all frameworks
        with self.metrics.time_generation():
            result = self.backend.generate(prompt, **kwargs)
        return self._wrap_result(result)
```

#### Framework Backend System

**Abstract Base Class**: `FrameworkBackend`
```python
class FrameworkBackend:
    def load_model(self, model_path: str, **kwargs) -> Any: ...
    def load_tokenizer(self, model_path: str) -> Any: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def get_memory_usage(self) -> MemoryInfo: ...
    def cleanup(self) -> None: ...
```

**TinyGrad Backend** (`src/llama/tinygrad_backend.py`):
- Native TinyGrad model loading and inference
- JIT compilation management
- Multi-GPU tensor sharding
- GGUF format support

**PyTorch Backend** (`src/llama/pytorch_backend.py`):
- HuggingFace transformers integration
- torch.compile optimization
- Weight conversion from GGUF format
- Device management (CUDA/CPU/MPS)

**Hybrid Backend** (`src/llama/accelerated_llama.py`):
- Zero-copy tensor conversion via PyTorch Bridge
- TinyGrad kernel fusion with PyTorch ecosystem
- Fallback mechanisms for optimization failures
- Cross-framework memory management

### Framework Implementation Details

#### TinyGrad Implementation

**Model Architecture**:
```python
# LLaMA implementation using TinyGrad primitives
class TinyGradLLaMA:
    def __init__(self, config):
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.embed_tokens = Embedding(config.vocab_size, config.dim)
        self.norm = RMSNorm(config.dim)

    def forward(self, tokens, start_pos=0):
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x, start_pos)
        return self.norm(x)
```

**Key Optimizations**:
- **JIT Compilation**: Automatic kernel fusion using `@TinyJit` decorator
- **Memory Layout**: Contiguous tensor allocation for optimal GPU usage
- **Multi-GPU Sharding**: `tensor.shard_(devices, axis=axis)` for distribution
- **Quantization**: Runtime precision reduction (int8, nf4, float16)

#### PyTorch Implementation

**Model Loading Strategy**:
```python
# Weight conversion from GGUF to PyTorch format
class PyTorchBackend:
    def load_model(self, model_path, **kwargs):
        # Load GGUF weights
        gguf_weights = load_gguf_file(model_path)

        # Convert to PyTorch format
        pytorch_weights = self.convert_weights(gguf_weights)

        # Create model with converted weights
        model = LlamaForCausalLM(config)
        model.load_state_dict(pytorch_weights)

        # Apply optimizations
        if kwargs.get('use_half', True):
            model = model.half()
        if kwargs.get('use_compile', True):
            model = torch.compile(model)

        return model
```

**Performance Optimizations**:
- **torch.compile**: JIT compilation for equivalent performance to TinyGrad
- **Precision Control**: Automatic float16 for fair comparison
- **Memory Management**: Efficient GPU memory allocation and cleanup
- **Device Synchronization**: Proper CUDA/MPS synchronization

#### Hybrid Implementation

**PyTorch-TinyGrad Bridge**:
```python
class TensorBridge:
    @staticmethod
    def torch_to_tinygrad(torch_tensor):
        """Zero-copy conversion when possible"""
        if torch_tensor.is_contiguous():
            return Tensor.from_blob(
                torch_tensor.data_ptr(),
                torch_tensor.shape,
                dtype=_from_torch_dtype(torch_tensor.dtype)
            )
        return Tensor(torch_tensor.detach().numpy())

    @staticmethod
    def tinygrad_to_torch(tg_tensor):
        """Convert TinyGrad tensor to PyTorch"""
        return torch.from_numpy(tg_tensor.numpy())
```

**Hybrid Architecture Benefits**:
- **Ecosystem Access**: Full HuggingFace compatibility via PyTorch
- **Kernel Optimization**: TinyGrad's optimized kernels for compute-heavy operations
- **Memory Efficiency**: Zero-copy operations where architecturally compatible
- **Fallback Safety**: Graceful degradation when optimizations fail

### Performance Measurement System

#### BenchmarkResult Structure
```python
@dataclass
class BenchmarkResult:
    framework: str
    model_size: str
    iterations: int
    total_time: float
    first_token_time: float
    avg_token_time: float
    min_token_time: float
    max_token_time: float
    avg_tokens_per_second: float
    peak_tokens_per_second: float
    peak_memory_gb: float
    model_memory_gb: float
    precision: str
    optimization_level: str
    jit_compilation_time: float
    model_loading_time: float
```

#### Memory Tracking
```python
class MemoryTracker:
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()

    def get_memory_delta(self) -> MemoryInfo:
        current = self._get_memory_usage()
        return MemoryInfo(
            gpu_used=current.gpu_used - self.baseline_memory.gpu_used,
            cpu_used=current.cpu_used - self.baseline_memory.cpu_used,
            gpu_total=current.gpu_total,
            cpu_total=current.cpu_total
        )
```

### Model Configuration System

#### Model Registration
```python
# Model configuration registry
MODEL_CONFIGS = {
    "llama3-1b": LLaMAConfig(
        name="Llama-3.2-1B-Instruct",
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        max_seq_len=131072,
        url="https://huggingface.co/...",
        format="gguf"
    ),
    "llama3-8b": LLaMAConfig(
        # Configuration for 8B model
    )
}
```

#### Dynamic Model Loading
```python
def load_model_config(model_name: str) -> LLaMAConfig:
    if model_name not in MODEL_CONFIGS:
        # Auto-detection from model path
        config = detect_model_config(model_name)
        if config:
            return config
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_CONFIGS[model_name]
```

### Multi-GPU and Distributed Support

#### Tensor Sharding
```python
# Multi-GPU tensor distribution
def shard_model(model, device_count: int):
    devices = tuple(f"cuda:{i}" for i in range(device_count))

    # Shard embedding and output layers along vocabulary dimension
    model.embed_tokens.weight.shard_(devices, axis=0)
    model.lm_head.weight.shard_(devices, axis=0)

    # Shard attention layers along head dimension
    for layer in model.layers:
        layer.self_attn.q_proj.weight.shard_(devices, axis=0)
        layer.self_attn.k_proj.weight.shard_(devices, axis=0)
        layer.self_attn.v_proj.weight.shard_(devices, axis=0)

        # Shard feed-forward networks
        layer.mlp.gate_proj.weight.shard_(devices, axis=0)
        layer.mlp.up_proj.weight.shard_(devices, axis=0)
```

#### KV Cache Sharding
```python
# Distributed attention cache
if getenv("SHARD_KVCACHE"):
    cache_kv.shard_(devices, axis=3)  # Shard along head dimension
```

### Caching and JIT Compilation

#### TinyGrad JIT Architecture
```python
@TinyJit
def generate_token(model, tokens, start_pos):
    """JIT-compiled token generation loop"""
    logits = model(tokens, start_pos)
    return logits.argmax(dim=-1)

# First call: Compilation overhead (~75ms)
# Subsequent calls: Optimized execution (~1ms)
```

#### Kernel Cache Management
```python
class KernelCacheManager:
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "tinygrad"
        self.cache_db = self.cache_dir / "cache.db"

    def get_cache_stats(self) -> CacheStats:
        """Analyze kernel cache contents and performance"""
        return CacheStats(
            total_kernels=self._count_cached_kernels(),
            cache_size_mb=self._get_cache_size(),
            hit_rate=self._calculate_hit_rate()
        )
```

### Error Handling and Fallbacks

#### Graceful Degradation
```python
class BackendManager:
    def __init__(self, preferred_backends: List[str]):
        self.backends = self._initialize_backends(preferred_backends)

    def get_working_backend(self) -> FrameworkBackend:
        """Try backends in preference order until one works"""
        for backend_name in self.backends:
            try:
                backend = self._create_backend(backend_name)
                if self._test_backend(backend):
                    return backend
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                continue

        raise RuntimeError("No working backend available")
```

#### Resource Cleanup
```python
class ResourceManager:
    def __enter__(self):
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup GPU memory
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

## Adding New Frameworks

### Implementation Steps

1. **Create Backend Class**:
```python
class NewFrameworkBackend(FrameworkBackend):
    def __init__(self):
        super().__init__()

    def load_model(self, model_path: str, **kwargs):
        # Framework-specific model loading
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        # Framework-specific inference
        pass
```

2. **Register Backend**:
```python
# In main.py or benchmark.py
AVAILABLE_BACKENDS = {
    "tinygrad": TinyGradBackend,
    "pytorch": PyTorchBackend,
    "hybrid": HybridBackend,
    "new_framework": NewFrameworkBackend,  # Add new backend
}
```

3. **Add CLI Support**:
```python
parser.add_argument("--framework",
                   choices=["tinygrad", "pytorch", "hybrid", "new_framework"],
                   default="tinygrad")
```

### Integration Requirements

**Required Methods**:
- `load_model()`: Model initialization and weight loading
- `load_tokenizer()`: Tokenizer setup
- `generate()`: Text generation with consistent interface
- `get_memory_usage()`: Memory consumption reporting
- `cleanup()`: Resource deallocation

**Performance Integration**:
- Timing measurement integration
- Memory tracking compatibility
- Error handling and fallback support
- Benchmark result formatting

## Related Documentation

- **[User Guide](user-guide.md)**: Usage instructions and examples
- **[Benchmarking Guide](benchmarking.md)**: Performance testing methodology
- **[Development Guide](development.md)**: Code quality and development workflow
- **[Research Findings](research.md)**: Advanced optimization techniques and analysis

---

**Next Steps**: See the [Development Guide](development.md) for contribution guidelines and code quality standards.