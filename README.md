# TinyGrad Inference Engine Benchmark Suite

A comprehensive benchmarking platform for Large Language Model (LLM) inference across multiple ML frameworks, with interactive demonstrations and detailed performance analysis.

## 🎯 Project Purpose

This repository **benchmarks leading inference engines** using standardized, fair, and reproducible tests across:

- **TinyGrad**: Minimal deep learning framework with optimized kernels
- **PyTorch**: Industry-standard framework with extensive ecosystem
- **Hybrid PyTorch-TinyGrad**: Custom implementation combining both frameworks

We focus on **inference-only testing** with unquantized LLaMA 3-1B for consistent comparisons without serving optimizations or batching.

## 🚀 Quick Start

### Installation
```bash
# Install dependencies using uv (recommended)
uv sync

# Verify installation
uv run --help
```

### Basic Usage
```bash
# Interactive Q&A with default model (LLaMA 3-1B)
uv run

# Run benchmark comparison
uv run -- --benchmark

# Single inference with timing
uv run -- --prompt "Explain quantum computing" --timing

# Use different model sizes
uv run -- --model llama3-8b --benchmark
```

## 📊 Framework Comparison Results

Our fair benchmark comparing TinyGrad vs PyTorch (LLaMA 3-1B):

```
📊 Framework Comparison - Inference Engine Benchmark
================================================================
Metric                    TinyGrad |  PyTorch-Opt |  PyTorch-Base
----------------------------------------------------------------
Model Load (s)               15.66 |        31.24 |        38.43
Cold Start (ms)              50.89 |       392.86 |       223.13
Steady-State Latency (ms)    12.29 |        24.28 |        24.43
Average Throughput (tok/s)    81.4 |         41.2 |         40.9
Peak Memory (GB)              5.59 |         3.06 |         6.10
```

**Key Findings**: TinyGrad achieves 2.0x throughput advantage with faster model loading and cold start performance.

## 🎮 Usage Modes

### 1. Interactive Demo Mode (Default)
Chat-style interface with live performance metrics:
```bash
uv run
```

### 2. Benchmarking Mode
Automated performance testing:
```bash
uv run -- --benchmark --iterations 10
```

### 3. Single Inference Mode
Quick performance testing:
```bash
uv run -- --prompt "Your question here" --timing
```

## 🔧 Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model size selection | `llama3-1b`, `llama3-8b`, `llama3-70b` |
| `--quantize` | Quantization method | `int8`, `nf4`, `float16` |
| `--benchmark` | Run performance benchmark | `--benchmark --iterations 20` |
| `--timing` | Show detailed timing info | `--timing` |
| `--temperature` | Generation temperature | `--temperature 0.7` |
| `--shard` | Multi-GPU sharding | `--shard 2` |

## 📚 Documentation

- **[User Guide](docs/user-guide.md)** - Detailed usage instructions and examples
- **[Benchmarking Guide](docs/benchmarking.md)** - Performance testing and analysis
- **[Architecture Overview](docs/architecture.md)** - Technical implementation details
- **[Development Guide](docs/development.md)** - Code quality, linting, and workflow
- **[Research Findings](docs/research.md)** - Advanced optimization research and TinyGrad analysis

## 🏗️ Supported Frameworks

### TinyGrad
- Optimized kernels with JIT compilation
- Multi-GPU tensor sharding
- Efficient memory management
- 75x speedup after JIT warmup

### PyTorch
- Full HuggingFace ecosystem
- torch.compile optimization
- Industry-standard reliability
- Configurable optimization levels

### Hybrid Implementation
- Zero-copy tensor operations between frameworks
- TinyGrad kernel fusion with PyTorch ecosystem
- Best-of-both-worlds approach

## 🎯 Model Support

**Primary Focus**: LLaMA 3-1B (unquantized) for standardized benchmarking

**Supported Models**:
- **1B**: Llama 3.2 1B Instruct (GGUF format)
- **8B**: SFR-Iterative-DPO-LLaMA-3-8B-R (Safetensors format)
- **70B**: DeepSeek-R1-Distill-Llama-70B (Safetensors format)

## 🚧 Development

### Code Quality
```bash
# Lint and format code
uv run ruff check --fix
uv run ruff format

# Run all quality checks
python scripts/lint.py --all
```

### Adding New Frameworks
See [Architecture Guide](docs/architecture.md#adding-new-frameworks) for implementation details.

## 📈 Performance Metrics

- **Model Loading Time**: Time to initialize model and tokenizer
- **Cold Start Latency**: First inference time including JIT compilation
- **Steady-State Throughput**: Tokens per second after warmup
- **Memory Usage**: Peak GPU/CPU memory consumption
- **Framework Comparison**: Side-by-side performance analysis

---

**Fast, comprehensive ML framework benchmarking with TinyGrad** 🚀

For detailed usage instructions, see the [User Guide](docs/user-guide.md).