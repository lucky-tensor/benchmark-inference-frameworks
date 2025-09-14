# Claude Code Project Settings

## Project Overview
This is a simplified tinygrad demonstration project focused on LLaMA 3 implementation with direct, working inference.

## Common Commands
- `uv run ruff check` - Lint code (ALWAYS run after every task)
- `uv run ruff check --fix` - Auto-fix lint issues
- `uv run ruff format` - Format code
- `uv run main.py --prompt "your question"` - Run direct inference

## Development Workflow
1. Make code changes
2. Run `uv run ruff check --fix` to auto-fix issues
3. Fix any remaining lint errors manually
4. Only proceed to next task when linting passes

## Simplified Structure

This project now uses a **direct, simple approach** instead of complex factory patterns:

- `main.py` - Simple, direct inference script (no factory pattern)
- `src/frameworks/tinygrad/llama/` - Core LLaMA implementation
- `src/common/` - Shared utilities (tokenizer, generation)

## Usage

```bash
# Simple inference
uv run main.py --prompt "what is machine learning?" --max-length 30

# With specific model size
uv run main.py --prompt "explain quantum computing" --size 1B --max-length 50
```

## Key Insights from Refactoring

**What worked:** Simple, direct approach based on the original working version

**What broke:** Complex factory patterns, registration systems, and over-abstraction

**The fix:** Removed all factory complexity and went back to a straightforward, working implementation that:
- Directly imports what it needs
- Uses the actual working modules
- No intermediate abstractions
- Clear, linear execution flow

## Working Implementation

The current `main.py` is based on the original working `llama3.py` from commit `bb7a213` and works perfectly with:
- ✅ Proper LLaMA-3 Instruct prompt formatting
- ✅ CUDA GPU acceleration
- ✅ Coherent text generation
- ✅ Simple, maintainable code structure