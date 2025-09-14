# TinyGrad Demo - LLaMA 3 Implementation

This project provides a simplified LLaMA 3 implementation using TinyGrad with a unified benchmarking system for comparing performance across different ML frameworks including TinyGrad and PyTorch.

## Features

- **Simplified CLI Interface**: Easy-to-use command-line interface for running LLaMA models
- **Unified Benchmarking**: Compare performance across TinyGrad and PyTorch frameworks
- **Standardized Metrics**: Consistent reporting of latency, throughput, and memory usage
- **Framework Abstraction**: Clean backend system for adding new frameworks
- **Side-by-side Comparison**: Direct performance comparisons with relative speedup metrics

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

- `--framework {tinygrad,pytorch} [...]`: Framework(s) to benchmark. If not specified, compares all frameworks.
- `--model-type {llama,gpt}`: Model type to load (required)
- `--model-path PATH`: Path to model directory or file (required)
- `--iterations N`: Number of benchmark iterations (default: 20)
- `--quantize {int8,nf4,float16}`: Quantization method
- `--shard N`: Number of devices for model sharding

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

#### Currently Supported Model Types

- **LLaMA**: Fully supported (1B, 8B, 70B, 405B)
- **GPT**: Not yet implemented (will show clear error message)

## Benchmark Results

The unified benchmark system provides comprehensive metrics:

### Performance Metrics
- **Average latency**: Mean time per token generation
- **First token latency**: Time to generate first token (includes prefill)
- **Peak throughput**: Maximum tokens per second achieved
- **Average throughput**: Mean tokens per second across all iterations

### Memory Metrics  
- **Model memory**: Memory used by model weights
- **Peak memory**: Maximum memory usage during inference
- **Memory efficiency**: Framework-specific memory optimizations

### Example Results

```
📊 Framework Comparison
================================================================================
Metric                           TinyGrad |         PyTorch
-------------------------------------------------------------
Avg Latency (ms)                    15.89 |           25.47
Avg Throughput (tok/s)               62.9 |            39.3
Peak Memory (GB)                     5.59 |            6.10

🏁 Performance Comparison (TinyGrad vs PyTorch):
  TinyGrad is 1.6x faster in throughput
  TinyGrad has 1.6x lower latency
  TinyGrad uses 1.1x less memory
```

Results show TinyGrad consistently outperforming PyTorch across all metrics for LLaMA inference.

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