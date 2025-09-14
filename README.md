# TinyGrad LLM Inference Demo

High-performance LLM inference system using **TinyGrad** with support for LLaMA 3 and GPT-2 models, featuring multi-GPU tensor sharding, comprehensive performance metrics, and interactive chat interfaces.

## Quick Start

### Interactive Chat Mode
Start an interactive Q&A session with real-time performance statistics:

```bash
# LLaMA 3 1B model (recommended)
uv run python inference.py llama3-1b --chat --temperature 0.7

# GPT-2 Medium model
uv run python inference.py gpt2-medium --chat --temperature 0.7 --count 50
```

### Single Generation Mode
Generate text for a specific prompt:

```bash
# LLaMA 3 with custom prompt
uv run python inference.py llama3-1b --prompt "Explain quantum computing" --temperature 0.7

# GPT-2 with longer generation
uv run python inference.py gpt2-medium --prompt "The future of AI" --count 100 --temperature 0.8
```

### Multi-GPU Support
Enable tensor sharding across multiple GPUs:

```bash
# Shard model across available GPUs
uv run python inference.py llama3-1b --chat --shard --temperature 0.7
```

## Files

### Core Implementation
- **`inference.py`** - Unified CLI entry point with interactive chat and performance metrics
- **`chat_interface.py`** - Generic chat interface abstraction for multiple model types
- **`llama3.py`** - LLaMA 3 implementation with chat format support
- **`gpt2.py`** - GPT-2 implementation with text continuation
- **`setup_verification.py`** - Automated setup and dependency verification

## Performance Metrics

The system provides comprehensive real-time statistics:
- **TTFT** (Time to First Token) - Latency until first response token
- **TPT** (Time Per Token) - Average generation speed per token
- **Tokens/Second** - Overall throughput measurement
- **Response Time** - Total time for complete response
- **GPU Memory Usage** - Real-time memory consumption tracking

**Example Output:**
```
Q: What is machine learning?
A: [Generated response...]
⏱️  Response: 3.45s | 🎯 Tokens: 28 | 🚀 Speed: 8.1 tok/s | ⚡ TTFT: 0.52s
```

## Supported Models

### LLaMA 3
- **llama3-1b** - 1B parameter model (recommended for most use cases)
- Full chat format support with system/user/assistant messages
- Optimized for conversational AI applications

### GPT-2
- **gpt2** - Base 124M parameter model
- **gpt2-medium** - 355M parameter model
- **gpt2-large** - 774M parameter model
- **gpt2-xl** - 1.5B parameter model
- Simple text continuation format

## Key Features

1. **Unified CLI Interface**: Single entry point (`inference.py`) for all models and modes
2. **Interactive Chat Mode**: Real-time Q&A with performance statistics
3. **Multi-GPU Support**: Automatic tensor sharding across available GPUs
4. **Industry-Standard Metrics**: TTFT, TPT, tokens/sec tracking
5. **Generic Chat Interface**: Abstract model support for easy extensibility
6. **Verbose Logging**: Detailed step-by-step timing and memory usage
7. **CUDA Acceleration**: Full GPU support with memory introspection

## Architecture

### Generic Chat Interface
- **ChatInterface** abstract base class for model-agnostic operations
- **ResponseStats** comprehensive metrics collection
- **VerboseLogger** detailed performance tracking

### Model Implementations
- **LLaMA 3**: Chat format with system/user/assistant messages
- **GPT-2**: Simple text continuation with configurable parameters
- **Multi-GPU**: Tensor sharding using `.shard_()` method

### Performance Optimization
- **JIT Compilation**: Leverages tinygrad's kernel caching (`~/.cache/tinygrad/`)
- **Memory Management**: Real-time GPU memory tracking
- **Batch Processing**: Efficient token generation loops

## Requirements

- **Python** >= 3.12
- **TinyGrad** >= 0.11.0
- **CUDA-compatible GPU** (recommended) or CPU fallback
- **uv** package manager for dependency management

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tinygrad-demo

# Install dependencies
uv sync

# Verify setup
uv run python setup_verification.py
```

## Advanced Usage

### Custom Generation Parameters
```bash
# Higher temperature for more creative responses
uv run python inference.py llama3-1b --chat --temperature 1.2

# Longer responses with custom token count
uv run python inference.py gpt2-medium --chat --count 150 --temperature 0.8

# Single prompt with specific parameters
uv run python inference.py llama3-1b --prompt "Write a poem about AI" --temperature 0.9
```

### GPU Memory Optimization
```bash
# Monitor GPU memory usage with verbose logging
uv run python inference.py llama3-1b --chat --verbose

# Use tensor sharding for large models
uv run python inference.py llama3-1b --chat --shard
```

### Performance Monitoring
The system automatically tracks and displays:
- **Cold Start Time**: Initial model loading duration
- **TTFT (Time to First Token)**: Latency before first response token
- **Generation Speed**: Tokens per second throughout the response
- **Memory Usage**: Real-time GPU memory consumption

---

**High-performance LLM inference with TinyGrad - unified, fast, and feature-complete!** 🚀