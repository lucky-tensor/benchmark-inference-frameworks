# Inference Engine Benchmark Suite

This repository benchmarks leading inference engines for LLaMA 3 models, providing comprehensive performance comparisons across different ML frameworks and optimization levels.

## Repository Goal

The goal of this repository is to **benchmark leading inference engines** using standardized, fair, and reproducible tests. We focus on:

- **Inference-only testing**: Pure token generation performance without serving optimizations
- **No batching or caching**: Tests individual request performance, not serving throughput
- **Unquantized LLaMA 3-1B**: Consistent model choice across all frameworks
- **Fair comparisons**: Same model architecture, precision, and hardware conditions

## Framework Coverage

We test the following configurations:

| Framework | Configuration | Optimization Level |
|-----------|---------------|-------------------|
| **PyTorch** | Unoptimized | Baseline float32, no compilation |
| **PyTorch** | Optimized | float16 + torch.compile |
| **TinyGrad** | Minimalist | Out-of-the-box with auto-JIT |
| **TensorRT** | Unoptimized | Basic TensorRT engine |
| **TensorRT** | Optimized | Full optimization pipeline |
| **Ollama** | Out-of-the-box | Default Ollama serving |

## Model Choice: LLaMA 3-1B

We standardize on **unquantized LLaMA 3-1B** for all tests because:

- **Mature ecosystem**: LLaMA 3 has been available long enough for all frameworks to optimize for it
- **Consistent architecture**: Same transformer design across all implementations
- **Reasonable size**: 1B parameters fit in consumer GPUs while being representative
- **Broad support**: All target frameworks have LLaMA 3 implementations

## Features

- **Multi-Framework Support**: Unified benchmarking across PyTorch, TinyGrad, TensorRT, and Ollama
- **Fair Comparison Controls**: Configurable optimization levels and precision settings
- **Comprehensive Metrics**: Model loading time, cold start latency, steady-state throughput, memory usage
- **Standardized Testing**: Same model, same hardware, same inference patterns across all engines
- **Reproducible Results**: Detailed configuration tracking and consistent test conditions
- **Extensible Architecture**: Clean backend system for adding new inference engines

## Quick Start

### Prerequisites

```bash
# Install dependencies using uv (recommended)
uv sync
```

### Running Models

#### Simple Model Execution

```bash
# Run LLaMA 3 1B model (default) - Q&A interface
uv run

# Run with different model sizes
uv run -- --model llama3-8b
uv run -- --model llama3-70b

# Run with quantization
uv run -- --model llama3-1b --quantize int8

# Run benchmark
uv run -- --model llama3-1b --benchmark
```

#### Available Main CLI Options

- `--model {llama3-1b,llama3-8b,llama3-70b,llama3-405b}`: Model to run (default: llama3-1b)
- `--quantize {int8,nf4,float16}`: Quantization method
- `--shard N`: Shard the model across multiple devices (default: 1)
- `--download`: Force download of model
- `--temperature FLOAT`: Temperature for sampling (default: 0.85)
- `--seed INT`: Random seed
- `--benchmark`: Run a benchmark
- `--timing`: Print timing per token
- `--profile`: Output profile data
- `--debug`: Enable debug mode

### Framework Benchmarking

#### Single Framework Benchmarks

```bash
# Benchmark TinyGrad only
uv run src/benchmark.py --framework tinygrad --model-type llama --model-path ~/models/llama3-1b-instruct/

# Benchmark PyTorch only
uv run src/benchmark.py --framework pytorch --model-type llama --model-path ~/models/llama3-1b-instruct/
```

#### Framework Comparison (Automatic)

```bash
# Compare all frameworks (automatic comparison)
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/

# Compare specific frameworks
uv run src/benchmark.py --framework tinygrad pytorch --model-type llama --model-path ~/models/llama3-1b-instruct/
```

### Benchmark Command Line Options

- `--framework {tinygrad,pytorch,tensorrt,ollama} [...]`: Framework(s) to benchmark. If not specified, compares all frameworks.
- `--model-type {llama}`: Model type to load (standardized on llama)
- `--model-path PATH`: Path to unquantized LLaMA 3-1B model (required)
- `--iterations N`: Number of benchmark iterations (default: 20)

#### Framework-Specific Options

- `--pytorch-no-compile`: Disable torch.compile optimization (test unoptimized PyTorch)
- `--pytorch-no-half`: Use float32 instead of float16 (test precision impact)
- `--fair-comparison`: Enable fair comparison mode (default: enabled)

#### Advanced Options

- `--shard N`: Number of devices for model sharding
- `--quantize {int8,nf4,float16}`: Quantization method (not recommended for benchmark standardization)

### Model Type Validation

The system automatically detects the model type from your model path and validates it against your `--model-type` argument:

- **Automatic Detection**: Analyzes filenames, directory contents, and file extensions
- **LLaMA Detection**: Looks for "llama", ".gguf", "tokenizer.model", "consolidated", "model.safetensors"
- **GPT Detection**: Looks for "gpt", "vocab.bpe", "encoder.json", "pytorch_model"
- **Validation Error**: Fails fast if model type doesn't match the detected type

```bash
# This will fail with validation error:
uv run python benchmark.py --framework tinygrad --model-type gpt --model-path ~/models/llama3-1b-instruct/
# Error: Model type mismatch: Expected 'gpt' but detected 'llama'
```

#### Currently Supported Frameworks

| Framework | Status | Configuration | Notes |
|-----------|--------|---------------|-------|
| **TinyGrad** | ✅ Implemented | Out-of-the-box + auto-JIT | Reference implementation |
| **PyTorch** | ✅ Implemented | Configurable optimization levels | Unoptimized & optimized modes |
| **TensorRT** | 🚧 Planned | Engine + optimization pipeline | High-performance GPU inference |
| **Ollama** | 🚧 Planned | Default serving configuration | Popular serving framework |

#### Implementation Roadmap

- **Phase 1** ✅: TinyGrad + PyTorch baseline comparison
- **Phase 2** 🚧: TensorRT integration (unoptimized + optimized)
- **Phase 3** 🚧: Ollama integration
- **Phase 4** 🚧: Additional frameworks (ONNX Runtime, vLLM, etc.)

## Benchmark Results

Our standardized inference-only benchmark provides comprehensive performance metrics across all supported frameworks.

### Key Performance Metrics

#### Inference Performance
- **Model Loading Time**: Time to load model and tokenizer from disk
- **Cold Start Latency**: First inference time (includes JIT compilation and warmup)
- **Steady-State Latency**: Mean time per token after warmup
- **Peak Throughput**: Maximum tokens per second achieved
- **Average Throughput**: Mean tokens per second across all test iterations

#### Memory Efficiency
- **Model Memory**: GPU memory used by model weights
- **Peak Memory**: Maximum memory usage during inference
- **Memory Overhead**: Framework-specific memory management costs

#### Optimization Impact
- **Warmup Improvement**: Performance gain from cold start to steady-state
- **Precision Impact**: Performance difference between float32 and float16
- **Compilation Benefit**: JIT compilation vs uncompiled performance

### Example Results (LLaMA 3-1B Unquantized)

Current benchmark results comparing TinyGrad vs PyTorch (fair comparison mode):

```
📊 Framework Comparison - Inference Engine Benchmark
================================================================================
Metric                           TinyGrad |    PyTorch-Opt |  PyTorch-Base
-------------------------------------------------------------
Model Load (s)                      15.66 |          31.24 |         38.43
Cold Start (ms)                     50.89 |         392.86 |        223.13
Steady-State Latency (ms)           12.29 |          24.28 |         24.43
Average Throughput (tok/s)           81.4 |           41.2 |          40.9
Peak Memory (GB)                     5.59 |           3.06 |          6.10
Precision                           mixed |     float16    |      float32

🏁 Performance Analysis:
  📥 TinyGrad loads 2.0x faster than optimized PyTorch
  ❄️  TinyGrad has 7.7x faster cold start than optimized PyTorch
  🔥 TinyGrad achieves 2.0x higher steady-state throughput
  💾 Memory usage varies by precision: float16 < mixed < float32
```

**Note**: Results demonstrate inference-only performance without batching, caching, or serving optimizations. See [fair-benchmark.md](fair-benchmark.md) for detailed fairness analysis.

### Coming Soon

Full benchmark matrix including TensorRT and Ollama across multiple optimization configurations.

## Architecture

### Framework Backends

The system uses an abstract base class `FrameworkBackend` with framework-specific implementations:

- `TinyGradBackend`: Integrates with TinyGrad's model loading and inference
- `PyTorchBackend`: Provides PyTorch implementation with weight conversion from GGUF format

### Standardized Results

All benchmarks produce `BenchmarkResult` objects with consistent metrics:

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
    # ... additional metrics
```

## Model Support

### Supported Models
- **1B**: Llama 3.2 1B Instruct (GGUF format)
- **8B**: SFR-Iterative-DPO-LLaMA-3-8B-R (Safetensors format)

### Model Path Handling
The system automatically detects model formats:
- Directory paths: Automatically finds GGUF files within directories
- Direct file paths: Uses the specified model file
- Auto-download: Downloads models if not found locally

## Alternative Interfaces

The original framework-specific scripts are still available:

```bash
# PyTorch backend interface
uv run src/pytorch-backend.py --size 1B --benchmark --model ~/models/llama3-1b-instruct/
```

## Development

### Adding New Frameworks

To add support for a new framework:

1. Create a new backend class inheriting from `FrameworkBackend`
2. Implement required methods: `load_model`, `load_tokenizer`, `prepare_input`, `run_inference`, etc.
3. Add framework choice to CLI arguments
4. Register the backend in the main function

### Testing

```bash
# Test main CLI interface
uv run -- --model llama3-1b --benchmark

# Test unified benchmarking system (compares all frameworks)
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/ --iterations 5

# Test model type validation
uv run src/benchmark.py --framework tinygrad --model-type gpt --model-path ~/models/llama3-1b-instruct/
# Should fail with validation error
```