# User Guide

Complete guide to using the TinyGrad Inference Engine Benchmark Suite for LLM inference and performance testing.

## Overview

This platform provides three primary usage modes:
1. **Interactive Demo**: Chat-style interface with real-time performance metrics
2. **Benchmarking**: Automated performance testing across frameworks
3. **Single Inference**: Quick testing with detailed timing analysis

All modes use the same underlying inference engine, ensuring consistent performance measurements across different user interactions.

For detailed benchmarking methodology, see the [Benchmarking Guide](benchmarking.md). For technical implementation details, see the [Architecture Overview](architecture.md).

## Installation and Setup

### Prerequisites
- Python ≥ 3.12
- CUDA-compatible GPU (recommended)
- 16GB+ GPU memory for larger models
- `uv` package manager (recommended)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd tinygrad-demo

# Install dependencies
uv sync

# Verify installation
uv run -- --help
```

### Environment Setup
```bash
# Optional: Enable specific optimizations
export TINYGRAD_JIT=1
export TINYGRAD_FUSION=1
export CUDA_VISIBLE_DEVICES=0,1  # Multi-GPU support
```

## Usage Modes

### Interactive Demo Mode (Default)

The default mode provides a chat-style interface for natural interaction with the model while displaying real-time performance metrics.

```bash
# Start interactive session
uv run

# With specific model
uv run -- --model llama3-8b --temperature 0.7

# With quantization
uv run -- --model llama3-1b --quantize int8
```

**Interactive Features**:
- Natural conversation interface
- Real-time token generation statistics
- Performance metrics after each response
- Graceful exit with `quit`, `exit`, or Ctrl+C

**Example Session**:
```
🚀 TinyGrad LLaMA Demo - Interactive Mode
Model: llama3-1b | Device: CUDA | Framework: TinyGrad

You: Explain machine learning in simple terms

Assistant: Machine learning is like teaching computers to recognize patterns...

📊 Performance: Generated 45 tokens in 0.612s (73.5 tok/s)
💾 Memory: 3.2GB GPU usage
⚡ JIT: Optimized (75x speedup after warmup)

You: quit
Goodbye! 👋
```

### Benchmarking Mode

Automated performance testing with standardized prompts and comprehensive metrics.

```bash
# Basic benchmark
uv run -- --benchmark

# Extended benchmark
uv run -- --benchmark --iterations 20 --model llama3-8b

# Framework comparison
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/
```

**Benchmark Features**:
- Standardized test prompts for consistency
- Multiple iterations for statistical significance
- Framework comparison reports
- Performance regression testing
- Comprehensive timing analysis

**Example Output**:
```
📊 BENCHMARK RESULTS - TINYGRAD
============================================================
Configuration:
  Model: llama3-1b
  Iterations: 10
  Prompts: 3 standard prompts

Performance Metrics:
  Average generation time: 1.234s
  Average tokens/second: 81.4
  First token latency: 50.89ms
  Model loading time: 15.66s
  Peak memory usage: 5.59GB
============================================================
```

### Single Inference Mode

Quick testing with a specific prompt and detailed timing information.

```bash
# Single inference
uv run -- --prompt "Explain quantum computing"

# With detailed timing
uv run -- --prompt "What is AI?" --timing --profile

# Framework comparison
uv run -- --prompt "Hello world" --framework pytorch
```

**Single Inference Features**:
- Custom prompt input
- Detailed per-token timing
- Memory usage profiling
- Framework selection
- Temperature and sampling control

## Configuration Options

### Model Configuration

| Option | Values | Description | Example |
|--------|--------|-------------|---------|
| `--model` | `llama3-1b`, `llama3-8b`, `llama3-70b`, `llama3-405b` | Model size selection | `--model llama3-8b` |
| `--quantize` | `int8`, `nf4`, `float16` | Quantization method | `--quantize int8` |
| `--model-path` | PATH | Custom model path | `--model-path ~/models/custom/` |
| `--download` | FLAG | Force model download | `--download` |

### Performance Configuration

| Option | Values | Description | Example |
|--------|--------|-------------|---------|
| `--shard` | INTEGER | Multi-GPU sharding | `--shard 2` |
| `--temperature` | FLOAT | Generation temperature | `--temperature 0.7` |
| `--seed` | INTEGER | Random seed | `--seed 42` |
| `--iterations` | INTEGER | Benchmark iterations | `--iterations 20` |

### Framework Configuration

| Option | Values | Description | Example |
|--------|--------|-------------|---------|
| `--framework` | `tinygrad`, `pytorch`, `hybrid` | Framework selection | `--framework pytorch` |
| `--pytorch-no-compile` | FLAG | Disable torch.compile | `--pytorch-no-compile` |
| `--pytorch-no-half` | FLAG | Use float32 precision | `--pytorch-no-half` |
| `--fair-comparison` | FLAG | Apply fairness controls | `--fair-comparison` |

### Debug and Profiling

| Option | Values | Description | Example |
|--------|--------|-------------|---------|
| `--timing` | FLAG | Show per-token timing | `--timing` |
| `--profile` | FLAG | Enable profiling | `--profile` |
| `--debug` | FLAG | Verbose debug output | `--debug` |

## Model Support

### Supported Models

#### LLaMA 3 Models (Primary Focus)
- **llama3-1b**: Llama 3.2 1B Instruct (GGUF format, ~2GB)
- **llama3-8b**: SFR-Iterative-DPO-LLaMA-3-8B-R (Safetensors, ~16GB)
- **llama3-70b**: DeepSeek-R1-Distill-Llama-70B (Safetensors, ~140GB)
- **llama3-405b**: Meta LLaMA 3.1 405B (Configuration available)

### Model Loading

**Automatic Download**: Models are downloaded automatically on first use:
```bash
# Downloads model if not present
uv run -- --model llama3-1b

# Force re-download
uv run -- --model llama3-8b --download
```

**Local Models**: Use custom model paths:
```bash
# Load from local directory
uv run -- --model-path ~/models/my-custom-llama/

# Load specific GGUF file
uv run -- --model-path ~/models/llama-model.gguf
```

**Model Format Support**:
- **GGUF**: TinyGrad native format with efficient loading
- **Safetensors**: HuggingFace format with PyTorch compatibility
- **PyTorch**: Standard .pt/.bin model files

## Performance Optimization

### Memory Management

**Quantization**: Reduce memory usage while maintaining performance:
```bash
# INT8 quantization (2x memory reduction)
uv run -- --model llama3-8b --quantize int8

# NF4 quantization (4x memory reduction)
uv run -- --model llama3-70b --quantize nf4

# Float16 precision (2x memory reduction)
uv run -- --model llama3-8b --quantize float16
```

**Multi-GPU Sharding**: Distribute model across multiple GPUs:
```bash
# Shard across 2 GPUs
uv run -- --model llama3-70b --shard 2

# Automatic GPU detection
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run -- --model llama3-405b --shard 4
```

### JIT Compilation Optimization

**TinyGrad JIT**: Automatic kernel fusion and optimization:
```bash
# Enable JIT compilation (default)
export TINYGRAD_JIT=1
export TINYGRAD_FUSION=1

# Cache analysis
uv run -- --model llama3-1b --timing
# Shows JIT compilation benefits (up to 75x speedup)
```

**PyTorch Optimization**: torch.compile for competitive performance:
```bash
# Optimized PyTorch (default in fair comparison)
uv run -- --framework pytorch --model llama3-1b

# Disable optimization for testing
uv run -- --framework pytorch --pytorch-no-compile --pytorch-no-half
```

## Troubleshooting

### Common Issues

**GPU Memory Errors**:
```bash
# Reduce memory usage with quantization
uv run -- --model llama3-8b --quantize int8

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
uv run -- --model llama3-1b
```

**Model Loading Issues**:
```bash
# Force model re-download
uv run -- --model llama3-1b --download

# Check model path
uv run -- --model-path ~/models/llama3-1b-instruct/ --debug
```

**Performance Issues**:
```bash
# Check JIT compilation status
uv run -- --model llama3-1b --timing --profile

# Compare frameworks
uv run src/benchmark.py --framework tinygrad pytorch --model-type llama
```

### Debug Mode

```bash
# Enable verbose debugging
uv run -- --model llama3-1b --debug

# Profile performance bottlenecks
uv run -- --model llama3-1b --profile --timing
```

## Integration Examples

### Python API Usage

```python
# Direct model usage
from src.main import load_model, generate_response

model = load_model("llama3-1b", quantize="int8")
response = generate_response(model, "Explain AI", temperature=0.7)
print(response)
```

### Batch Processing

```python
# Process multiple prompts
prompts = ["What is ML?", "Explain quantum computing", "Define AI"]

for prompt in prompts:
    response = generate_response(model, prompt)
    print(f"Q: {prompt}\nA: {response}\n")
```

### Custom Benchmarking

```python
# Custom benchmark script
from src.benchmark import run_benchmark

results = run_benchmark(
    framework="tinygrad",
    model_type="llama",
    model_path="~/models/llama3-1b-instruct/",
    iterations=10
)

print(f"Throughput: {results.avg_tokens_per_second:.1f} tok/s")
```

## Related Documentation

- **[Benchmarking Guide](benchmarking.md)**: Detailed performance testing methodology
- **[Architecture Overview](architecture.md)**: Technical implementation details
- **[Development Guide](development.md)**: Contributing and code quality
- **[Research Findings](research.md)**: Advanced optimization research

---

**Next Steps**: See the [Benchmarking Guide](benchmarking.md) for comprehensive performance analysis techniques.