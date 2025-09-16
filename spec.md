
# Benchmark and Chat Interface Specification

## Overview

This program provides a unified interface for both interactive chat and automated benchmarking of inference frameworks (e.g., PyTorch, Tinygrad) using large language models. The main entrypoint is `main.py`, which supports two primary subcommands: `chat` and `benchmark`. Both subcommands share core logic for question/answer routines and timing analysis.

## Features

### 1. Entry Points
- **Chat Mode (`main.py chat`)**
	- Starts an interactive Q&A session with the user.
	- Iteratively prompts the user for questions and provides answers.
	- Accepts user feedback after each answer.
	- Supports a `--prompt` argument to run a single Q&A and exit.

- **Benchmark Mode (`main.py benchmark`)**
	- Runs automated benchmarks using a pre-defined list of questions.
	- The list of questions is randomly sorted for each benchmark run.
	- Supports an `--iterations` argument to specify the number of benchmark runs.
	- For each iteration, runs the Q&A routine and records timing data.
	- Allows comparison of different frameworks (e.g., PyTorch, Tinygrad) for a given algorithm (e.g., llama).

### 2. Timing and Logging
- Each Q&A routine includes checkpoints for timing, managed by a `TimeLog` class.
- Timing data is collected for each question/answer and stored in an array.
- In benchmark mode, timing data is aggregated across iterations.

### 3. Benchmark Analysis
- At the end of benchmarking, the program analyzes the timing log array.
- Analysis results (e.g., average, min, max, stddev) are displayed to the user.
- If multiple frameworks are benchmarked, a final comparison is shown.

### 4. Framework Comparison
- Benchmark mode supports running the same set of questions across multiple frameworks.
- After analysis, a summary table or chart compares performance metrics between frameworks.

## Command-Line Arguments

- `chat`:
	- `--prompt <text>`: Run a single Q&A with the provided prompt and exit.

- `benchmark`:
	- `--iterations <N>`: Number of benchmark iterations to run (default: 1).
	- `--frameworks <list>`: Comma-separated list of frameworks to compare (e.g., pytorch,tinygrad).
	- `--algorithm <name>`: Algorithm/model to benchmark (e.g., llama, gpt2).

## Class and Module Structure

- `main.py`: Entry point, argument parsing, and command dispatch.
- `TimeLog`: Class for managing timing checkpoints and logs.
- `QARoutine`: Shared logic for question/answer routines (used by both chat and benchmark).
- `BenchmarkAnalysis`: Functions for analyzing and comparing timing data.
- `FrameworkInterface`: Abstract interface for framework-specific implementations.
- `frameworks/`: Directory containing framework-specific backends (e.g., `pytorch_backend.py`, `tinygrad_backend.py`).

## Acceptance Criteria

- [ ] User can run `main.py chat` for interactive Q&A.
- [ ] User can run `main.py chat --prompt "<text>"` for a single Q&A.
- [ ] User can run `main.py benchmark --iterations N` to benchmark with N iterations.
- [ ] User can specify frameworks and algorithms for benchmarking.
- [ ] Timing data is collected and analyzed for each run.
- [ ] Benchmark results and framework comparisons are clearly displayed.
- [ ] Code is modular, with shared logic between chat and benchmark modes.

## Future Extensions
- Support for additional frameworks and algorithms.
- Enhanced analysis and visualization of benchmark results.
- Integration with external logging or experiment tracking tools.
