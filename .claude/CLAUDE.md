# Claude Code Project Settings

## Project Overview
This is a tinygrad demonstration project focused on LLaMA 3 implementation and PyTorch comparison benchmarking.

## Common Commands
- `uv run ruff check` - Lint code (ALWAYS run after every task)
- `uv run ruff check --fix` - Auto-fix lint issues
- `uv run ruff format` - Format code
- `uv run python src/benchmark.py` - Run benchmarks
- `uv run python src/main.py` - Run main inference demo
- `uv run python llama3.py --size 1B` - Run LLaMA 1B model

## Development Workflow
1. Make code changes
2. Run `uv run ruff check --fix` to auto-fix issues
3. Fix any remaining lint errors manually
4. Only proceed to next task when linting passes

## Key Files
- `llama3.py` - Main LLaMA implementation
- `src/` - Additional inference engine components
- `pyproject.toml` - UV package configuration