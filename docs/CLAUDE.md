# Tinygrad Demo - LLaMA 3 Implementation

## Overview

This is a LLaMA 3 implementation using tinygrad, a lightweight deep learning framework. The codebase provides support for running various LLaMA model sizes (1B, 8B, 70B, 405B) with different quantization options and serving capabilities.

## Key Components

### Main Script: `llama3.py`
The primary implementation containing:
- **Tokenizer**: Custom tokenizer for LLaMA 3 with special tokens support
- **Model Architecture**: Transformer implementation with quantization support
- **Quantization**: Int8 and NF4 quantization for memory efficiency
- **Interactive CLI**: Command-line interface for local usage

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

# Run benchmark mode
uv run python llama3.py --size 8B --benchmark
```

### Command Line Options
- `--size`: Model size (1B, 8B, 70B, 405B)
- `--quantize`: Quantization method (int8, nf4, float16)
- `--shard`: Number of devices for model sharding
- `--benchmark`: Run performance benchmark
- `--temperature`: Sampling temperature (default: 0.85)
- `--timing`: Print timing information per token
- `--profile`: Enable performance profiling

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

## Code Quality and Development Tools

### Linting and Code Quality
The project uses **Ruff** for comprehensive Python linting and formatting:

```bash
# Install development dependencies
uv sync --extra dev

# Check code quality (recommended before commits)
uv run ruff check

# Auto-fix issues where possible
uv run ruff check --fix

# Format code
uv run ruff format

# Run security checks
uv run bandit -r .

# Use the convenience script for all checks
python scripts/lint.py --all
```

### Pre-commit Hooks
Automatic code quality checks on commit:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks on all files manually
uv run pre-commit run --all-files
```

### Development Workflow
1. Make code changes
2. Run `python scripts/lint.py --fix` to auto-fix issues
3. Commit changes (pre-commit hooks will run automatically)
4. Push changes

### Linting Configuration
- **Ruff**: Configured in `pyproject.toml` with comprehensive rules
- **Pre-commit**: Configured in `.pre-commit-config.yaml`
- **Security**: Bandit checks for common security issues
- **Unused Code Detection**: Automatically detects unused imports and variables

## Notes for Claude Code

- **Lint Command**: `uv run ruff check` (fast, comprehensive Python linting)
- **Format Command**: `uv run ruff format` (automatic code formatting)
- **Type Check**: Not specified - Ruff includes basic type checking rules
- **Test Command**: No test files present - benchmarking available via `--benchmark` flag
- **Security Check**: `uv run bandit -r .` (security vulnerability scanning)
- **Model Downloads**: Automatic download on first run, cached in `~/models/` directory
- **GPU Support**: Leverages tinygrad's device abstraction for GPU acceleration