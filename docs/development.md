# Development Guide

Development workflow, code quality standards, and contribution guidelines for the TinyGrad Inference Engine Benchmark Suite.

## Quick Setup

### Development Environment
```bash
# Clone and setup
git clone <repository-url>
cd tinygrad-demo

# Install dependencies including dev tools
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run ruff check
python scripts/lint.py --check
```

## Code Quality Standards

### Linting and Formatting

We use **Ruff** as our primary linting and formatting tool for speed and comprehensiveness.

#### Core Tools
- **Ruff**: Lightning-fast Python linter and formatter (10-100x faster than traditional tools)
- **Bandit**: Security vulnerability scanning
- **Pre-commit**: Automated quality gates before commits

#### Quick Commands
```bash
# Essential workflow commands
uv run ruff check --fix    # Auto-fix issues
uv run ruff format         # Format code
python scripts/lint.py --all  # Run all checks
```

### Ruff Configuration

Our Ruff setup (`pyproject.toml`) includes:

**Enabled Rules**:
- **Pyflakes (F)**: Unused imports, variables, and undefined names
- **PEP 8 (E/W)**: Style guide compliance
- **Import Sorting (I)**: Automatic import organization
- **Performance (PERF)**: Performance improvement suggestions
- **Security (S)**: Basic security vulnerability detection
- **Complexity (C90)**: Cyclomatic complexity analysis

**Project-Specific Settings**:
```toml
[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["F", "E", "W", "I", "PERF", "S", "C90"]
ignore = [
    "E501",    # Line too long (handled by formatter)
    "S101",    # Assert statements (useful for ML debugging)
    "C901",    # Complex functions (ML algorithms can be complex)
]
```

### Development Workflow

#### Standard Development Process
1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Write code**: Follow existing patterns and conventions
3. **Lint and format**: `python scripts/lint.py --fix`
4. **Test changes**: Run relevant benchmarks or tests
5. **Commit**: Pre-commit hooks run automatically
6. **Push**: Create pull request for review

#### Pre-commit Integration
```bash
# Pre-commit hooks run automatically on git commit
git add .
git commit -m "Add new feature"  # Linting runs automatically

# To run hooks manually
uv run pre-commit run --all-files
```

### Code Style Guidelines

#### Naming Conventions
- **Classes**: PascalCase (`InferenceEngine`, `TinyGradBackend`)
- **Functions**: snake_case (`load_model`, `generate_response`)
- **Constants**: UPPER_SNAKE_CASE (`MODEL_CONFIGS`, `DEFAULT_TEMPERATURE`)
- **Private methods**: Leading underscore (`_internal_method`)

#### Documentation Standards
```python
def generate_response(
    model: Any,
    prompt: str,
    temperature: float = 0.85,
    max_tokens: int = 512
) -> GenerationResult:
    """Generate text response using specified model.

    Args:
        model: Loaded model instance
        prompt: Input text prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        GenerationResult with text and performance metrics

    Raises:
        ModelError: If model fails to generate response
    """
```

#### Type Hints
Use type hints consistently for better code clarity:
```python
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    framework: str
    avg_tokens_per_second: float
    peak_memory_gb: float
    iterations: int

def run_benchmark(
    backend: FrameworkBackend,
    iterations: int = 10
) -> BenchmarkResult:
    # Implementation
    pass
```

## Project Structure

### Directory Organization
```
tinygrad-demo/
├── src/                          # Source code
│   ├── main.py                   # CLI entry point
│   ├── benchmark.py              # Framework comparison benchmarks
│   ├── common/                   # Shared utilities
│   │   ├── inference_engine.py   # Unified inference logic
│   │   ├── model_configs.py      # Model configuration registry
│   │   ├── rotary_embeddings.py  # Shared model components
│   │   └── ...
│   ├── llama/                    # LLaMA-specific implementations
│   │   ├── accelerated_llama.py  # Hybrid PyTorch-TinyGrad
│   │   ├── pytorch_bridge.py     # Framework interop
│   │   ├── hybrid_model.py       # Hybrid model architecture
│   │   └── ...
│   └── pytorch-pure.py           # Pure PyTorch implementation
├── docs/                         # Documentation
│   ├── user-guide.md            # User-focused documentation
│   ├── benchmarking.md          # Performance testing guide
│   ├── architecture.md          # Technical implementation
│   ├── development.md           # This file
│   └── research.md              # Research findings
├── scripts/                      # Development utilities
│   └── lint.py                   # Linting convenience script
├── .github/                      # CI/CD workflows
├── pyproject.toml               # Project configuration
└── README.md                    # Project overview
```

### Module Organization Principles

#### Separation of Concerns
- **`src/common/`**: Framework-agnostic shared utilities
- **`src/llama/`**: LLaMA-specific implementations
- **`src/benchmark.py`**: Performance testing and comparison
- **`src/main.py`**: User-facing CLI interface

#### Dependency Management
```python
# Good: Clear separation of dependencies
from src.common.inference_engine import InferenceEngine
from src.llama.tinygrad_backend import TinyGradBackend

# Avoid: Cross-framework dependencies in shared code
# from torch import Tensor  # Don't import PyTorch in common modules
```

## Testing and Validation

### Manual Testing
```bash
# Test basic functionality
uv run -- --prompt "Hello world" --timing

# Test framework comparison
uv run src/benchmark.py --framework tinygrad pytorch --iterations 3

# Test error handling
uv run -- --model invalid-model  # Should fail gracefully
```

### Performance Validation
```python
# Benchmark consistency check
def validate_benchmark_consistency():
    results = []
    for _ in range(5):
        result = run_benchmark("tinygrad", iterations=3)
        results.append(result.avg_tokens_per_second)

    cv = np.std(results) / np.mean(results)
    assert cv < 0.1, f"High variance in results: {cv:.3f}"
```

### Output Quality Verification
```python
# Ensure different frameworks produce reasonable outputs
def test_output_quality():
    prompt = "Explain artificial intelligence in one sentence."

    tinygrad_output = generate_with_framework("tinygrad", prompt, seed=42)
    pytorch_output = generate_with_framework("pytorch", prompt, seed=42)

    # Outputs may vary but should be reasonable
    assert len(tinygrad_output) > 10
    assert len(pytorch_output) > 10
    assert "artificial intelligence" in tinygrad_output.lower()
```

## Adding New Features

### Framework Backend Implementation

To add a new ML framework:

1. **Create backend class**:
```python
# src/backends/new_framework.py
from src.common.inference_engine import FrameworkBackend

class NewFrameworkBackend(FrameworkBackend):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def get_name(self) -> str:
        return "NewFramework"

    def load_model(self, model_path: str, **kwargs):
        # Framework-specific model loading
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        # Framework-specific inference
        pass
```

2. **Register in main system**:
```python
# src/main.py
from src.backends.new_framework import NewFrameworkBackend

AVAILABLE_BACKENDS = {
    "tinygrad": TinyGradBackend,
    "pytorch": PyTorchBackend,
    "new_framework": NewFrameworkBackend,
}
```

3. **Add CLI support**:
```python
parser.add_argument("--framework",
                   choices=list(AVAILABLE_BACKENDS.keys()),
                   default="tinygrad")
```

### Model Support Extension

To add support for a new model architecture:

1. **Define model config**:
```python
# src/common/model_configs.py
NEW_MODEL_CONFIG = ModelConfig(
    name="new-model-1b",
    architecture="transformer",
    vocab_size=32000,
    dim=4096,
    n_layers=24,
    # ... other parameters
)

MODEL_CONFIGS["new-model-1b"] = NEW_MODEL_CONFIG
```

2. **Implement model loading**:
```python
def load_new_model_format(model_path: str) -> Dict:
    """Load model weights from new format"""
    # Implementation specific to model format
    pass
```

3. **Add detection logic**:
```python
def detect_model_type(model_path: str) -> str:
    # Auto-detection logic
    if "new-model" in model_path.lower():
        return "new-model"
    # ... existing detection logic
```

## Performance Optimization

### Profiling and Analysis
```python
# Performance profiling utilities
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{description}: {elapsed:.3f}s")

# Usage
with timer("Model loading"):
    model = load_model("llama3-1b")
```

### Memory Management
```python
def optimize_memory_usage():
    """Memory optimization best practices"""

    # Clear GPU cache between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Use context managers for resource cleanup
    with ResourceManager() as resources:
        result = resources.model.generate(prompt)

    # Explicit cleanup for large objects
    del model
    import gc; gc.collect()
```

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/quality.yml
name: Code Quality
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: "3.12"

    - name: Install dependencies
      run: pip install uv && uv sync --extra dev

    - name: Run linting
      run: uv run ruff check

    - name: Check formatting
      run: uv run ruff format --check

    - name: Security scan
      run: uv run bandit -r src/
```

### Performance Monitoring
```python
# scripts/performance_monitor.py
def track_performance_regression():
    """Monitor performance over time"""
    current_performance = run_benchmark("tinygrad")
    baseline_performance = load_baseline_performance()

    regression_threshold = 0.05  # 5% performance drop
    if current_performance < baseline_performance * (1 - regression_threshold):
        raise RuntimeError(f"Performance regression detected: "
                         f"{current_performance:.1f} < {baseline_performance:.1f}")
```

## Troubleshooting Development Issues

### Common Setup Issues

#### Ruff Installation Problems
```bash
# Issue: Ruff not found after installation
# Solution: Ensure proper uv environment
uv sync --extra dev
uv run which ruff  # Should show path

# Alternative: Direct installation
pip install ruff
```

#### Pre-commit Hook Failures
```bash
# Issue: Pre-commit hooks failing
# Solution: Run hooks manually and fix issues
uv run pre-commit run --all-files
python scripts/lint.py --fix

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

#### CUDA Environment Issues
```bash
# Issue: CUDA not available during development
# Solution: Use CPU fallback for basic testing
export CUDA_VISIBLE_DEVICES=""
uv run -- --prompt "test" --debug
```

### Debugging Performance Issues

#### JIT Compilation Analysis
```python
# Debug JIT compilation performance
def analyze_jit_performance():
    with timer("First inference (JIT compilation)"):
        result1 = model.generate("test prompt")

    with timer("Second inference (optimized)"):
        result2 = model.generate("test prompt")

    print(f"JIT speedup: {time1/time2:.1f}x")
```

#### Memory Leak Detection
```python
import psutil
import os

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run operations
    for i in range(10):
        result = run_inference()

    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

    if memory_growth > 100:  # 100MB threshold
        print(f"Potential memory leak: {memory_growth:.1f}MB growth")
```

## Related Documentation

- **[User Guide](user-guide.md)**: Usage instructions and examples
- **[Benchmarking Guide](benchmarking.md)**: Performance testing methodology
- **[Architecture Overview](architecture.md)**: Technical implementation details
- **[Research Findings](research.md)**: Advanced optimization research

---

**Next Steps**: See the [Research Findings](research.md) for advanced optimization techniques and TinyGrad analysis.