# Extra Files Source Documentation

## Overview

This project uses a minimal subset of files from the tinygrad `extra/` directory. Instead of including the entire 9MB `extra/` folder, we've extracted only the 4 files (28KB total) that our scripts actually require.

## Source Information

**Repository:** https://github.com/tinygrad/tinygrad
**Branch:** master
**Commit Hash:** 19d9d29b7e55a513e47cbd73985f682dba35e00b
**Date Extracted:** 2025-09-13

## Files Extracted

### Core Dependencies (4 files, ~28KB total)

| File | Size | Purpose | Used By |
|------|------|---------|---------|
| `extra/bench_log.py` | 4KB | Performance benchmarking utilities | `llama3.py`, `gpt2.py` |
| `extra/training.py` | 4KB | Training loops and evaluation functions | `transformer.py` |
| `extra/models/llama.py` | 16KB | LLaMA model architecture implementation | `llama3.py` |
| `extra/models/transformer.py` | 4KB | Basic transformer architecture | `transformer.py` |

## What We Excluded

The original `extra/` directory contained ~9MB of additional files including:
- Dataset loaders (ImageNet, COCO, etc.)
- Hardware-specific backends (HSA, Triton, AMD, NVIDIA drivers)
- Assembly optimizations
- Kernel optimization tools
- PyTorch compatibility layers
- Development tools and examples

**Space Savings:** 99.7% reduction (9MB → 28KB)

## Direct URLs to Source Files

For reference, the extracted files can be found at:

- [`bench_log.py`](https://github.com/tinygrad/tinygrad/blob/19d9d29b7e55a513e47cbd73985f682dba35e00b/extra/bench_log.py)
- [`training.py`](https://github.com/tinygrad/tinygrad/blob/19d9d29b7e55a513e47cbd73985f682dba35e00b/extra/training.py)
- [`models/llama.py`](https://github.com/tinygrad/tinygrad/blob/19d9d29b7e55a513e47cbd73985f682dba35e00b/extra/models/llama.py)
- [`models/transformer.py`](https://github.com/tinygrad/tinygrad/blob/19d9d29b7e55a513e47cbd73985f682dba35e00b/extra/models/transformer.py)

## Updating These Files

To update these files to newer versions:

```bash
# Download specific files from GitHub
curl -O https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/bench_log.py
curl -O https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/training.py
curl -O https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/models/llama.py
curl -O https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/models/transformer.py

# Update this documentation with new commit hash
```

## Compatibility

These files are designed to work with:
- **tinygrad core:** >=0.11.0 (installed via PyPI)
- **Python:** >=3.12
- **Dependencies:** As specified in `pyproject.toml`

## Notes

- These files maintain their original licensing from the tinygrad project
- No modifications have been made to the extracted files
- All imports and dependencies are preserved as-is from the original