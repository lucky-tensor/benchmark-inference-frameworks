# Tinygrad Demo - LLaMA 3 Implementation

## Overview

This is a LLaMA 3 implementation using tinygrad, a lightweight deep learning framework. The codebase provides support for running various LLaMA model sizes (1B, 8B, 70B, 405B) with different quantization options and serving capabilities.

## Key Components

### Main Script: `llama3.py`
The primary implementation containing:
- **Tokenizer**: Custom tokenizer for LLaMA 3 with special tokens support
- **Model Architecture**: Transformer implementation with quantization support
- **Quantization**: Int8 and NF4 quantization for memory efficiency
- **Web API**: REST API server for chat completions and token operations
- **CLI Interface**: Interactive command-line interface for local usage

### Supported Models
- **1B**: Llama 3.2 1B Instruct (GGUF format)
- **8B**: SFR-Iterative-DPO-LLaMA-3-8B-R (Safetensors format)
- **70B**: DeepSeek-R1-Distill-Llama-70B (Safetensors format)
- **405B**: Meta LLaMA 3.1 405B (configuration available, no download URLs)

### Dependencies
Managed via `pyproject.toml`:
- `tinygrad>=0.11.0` - Core ML framework
- `transformers>=4.56.1` - Model utilities
- `tiktoken>=0.5.0` - Tokenization
- `safetensors>=0.6.2` - Model loading
- `numpy>=2.3.3` - Numerical operations

## Usage

### Environment Setup
```bash
# Using uv (recommended)
uv run python llama3.py [options]

# Or activate virtual environment
source .venv/bin/activate
python llama3.py [options]
```

### Download and Run Models
```bash
# Download and run 1B model
uv run python llama3.py --size 1B

# Download and run 8B model
uv run python llama3.py --size 8B

# Run with quantization
uv run python llama3.py --size 8B --quantize int8

# Run web API server
uv run python llama3.py --size 8B --host 0.0.0.0 --port 7776
```

### Command Line Options
- `--size`: Model size (1B, 8B, 70B, 405B)
- `--quantize`: Quantization method (int8, nf4, float16)
- `--shard`: Number of devices for model sharding
- `--benchmark`: Run performance benchmark
- `--no_api`: Use CLI interface instead of web server
- `--temperature`: Sampling temperature (default: 0.85)
- `--host`: Web server bind address (default: 0.0.0.0)
- `--port`: Web server port (default: 7776)

### API Endpoints
When running in API mode:
- `POST /v1/chat/completions` - Chat completions (streaming)
- `POST /v1/completions` - Text completions (streaming)
- `POST /v1/chat/token/encode` - Tokenize chat messages
- `GET /v1/models` - List available models

## Development Tools

### Package Management
- **uv**: Fast Python package manager and resolver
- Configured via `pyproject.toml` and `uv.lock`

### Project Structure
```
tinygrad-demo/
├── llama3.py          # Main implementation
├── pyproject.toml     # Project dependencies
├── uv.lock           # Dependency lock file
├── extra/            # Additional tinygrad modules
├── downloads/        # Downloaded model cache
└── weights/          # Local model weights
```

### Running Tests and Benchmarks
```bash
# Run benchmark mode
uv run python llama3.py --benchmark --size 8B

# Enable timing information
uv run python llama3.py --timing --size 8B

# Enable profiling
uv run python llama3.py --profile --size 8B
```

## Notes for Claude Code

- **Lint Command**: Not specified - check project for standard Python linting tools
- **Type Check**: Not specified - check for mypy or similar type checking setup
- **Test Command**: No test files present - benchmarking available via `--benchmark` flag
- **Model Downloads**: Automatic download on first run, cached in `downloads/` directory
- **GPU Support**: Leverages tinygrad's device abstraction for GPU acceleration