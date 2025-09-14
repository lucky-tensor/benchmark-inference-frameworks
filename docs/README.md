# TinyGrad LLM Inference Demo

High-performance LLM inference using TinyGrad with LLaMA 3 and GPT-2 support, multi-GPU sharding, and comprehensive performance metrics.

## Quick Start

### Setup
```bash
# Install dependencies
uv sync

# Verify setup
uv run python setup_verification.py
```

### Usage
```bash
# Interactive chat (recommended)
uv run python inference.py --model llama3-1b

# Single prompt generation
uv run python inference.py --model llama3-1b --prompt "Explain quantum computing" --temperature 0.7

# Multi-GPU sharding
uv run python inference.py --model llama3-1b --shard 2
```

## Supported Models
- **LLaMA 3**: `llama3-1b` (recommended), `llama3-8b`, `llama3-70b`
- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

## Key Features
- Interactive chat with real-time performance metrics
- Multi-GPU tensor sharding
- JIT compilation with cache analysis
- TTFT, tokens/sec, memory usage tracking
- CUDA acceleration with CPU fallback

## Requirements
- Python >= 3.12
- TinyGrad >= 0.11.0
- CUDA-compatible GPU (recommended)
- `uv` package manager

---
**Fast, feature-complete LLM inference with TinyGrad** 🚀