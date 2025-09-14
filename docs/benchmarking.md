# Benchmarking Guide

Comprehensive guide to performance testing and analysis with the TinyGrad Inference Engine Benchmark Suite.

## Overview

This guide covers fair benchmarking methodology, framework comparison techniques, and performance analysis tools for objective ML inference evaluation.

## Quick Start

### Basic Benchmarking
```bash
# Single framework benchmark
uv run -- --benchmark --framework tinygrad

# Multi-framework comparison (recommended)
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/

# Extended performance analysis
uv run -- --benchmark --iterations 20 --timing --profile
```

## Benchmarking Methodology

### Fair Comparison Principles

Our benchmarking follows strict fairness guidelines to ensure meaningful framework comparisons:

**✅ Controlled Variables**:
- **Same Model**: Identical LLaMA 3-1B architecture across all frameworks
- **Same Precision**: Consistent float16/mixed precision usage
- **Same Hardware**: Identical GPU, CUDA environment, and system configuration
- **Same Data**: Identical model weights loaded from same GGUF files
- **Same Prompts**: Standardized test inputs for reproducible results

**✅ Fair Optimization Levels**:
- **TinyGrad**: Out-of-the-box JIT compilation and kernel fusion
- **PyTorch**: Equivalent optimization with torch.compile and float16 precision
- **Hybrid**: Best-of-both-worlds approach with cross-framework optimization

**❌ Bias Mitigation**:
- No framework gets unfair advantages through cherry-picked optimizations
- Precision mismatches eliminated (both use 16-bit by default)
- Consistent optimization levels applied automatically
- Real-world representative configurations used

### Benchmark Types

#### 1. Inference-Only Benchmarking
**Focus**: Pure token generation performance without serving optimizations

```bash
# Standard inference benchmark
uv run src/benchmark.py --framework tinygrad --model-type llama --model-path ~/models/llama3-1b-instruct/

# Disable serving optimizations
# - No batching or request queuing
# - No KV cache sharing between requests
# - Individual request processing only
```

**Key Metrics**:
- **Tokens per second**: Pure generation throughput
- **First token latency**: Cold start performance including JIT compilation
- **Memory usage**: Peak GPU/CPU memory consumption
- **Model loading time**: Weight loading and initialization overhead

#### 2. Framework Comparison Benchmarking
**Focus**: Side-by-side framework performance analysis

```bash
# Automatic framework comparison
uv run src/benchmark.py --model-type llama --model-path ~/models/llama3-1b-instruct/

# Specific framework selection
uv run src/benchmark.py --framework tinygrad pytorch --model-type llama --model-path ~/models/llama3-1b-instruct/
```

**Comparison Matrix**:
| Framework | Configuration | Optimization Level |
|-----------|---------------|--------------------|
| **TinyGrad** | Default | JIT + kernel fusion |
| **PyTorch** | Optimized | torch.compile + float16 |
| **PyTorch** | Baseline | No optimization |
| **Hybrid** | Advanced | TinyGrad kernels + PyTorch ecosystem |

#### 3. Extended Performance Analysis
**Focus**: Comprehensive performance characterization

```bash
# Extended benchmark with profiling
uv run -- --benchmark --iterations 50 --timing --profile --model llama3-8b

# Memory analysis
uv run -- --benchmark --iterations 10 --quantize int8 nf4 float16
```

## Performance Metrics

### Core Performance Indicators

#### Throughput Metrics
- **Average Tokens/Second**: Mean generation speed across all iterations
- **Peak Tokens/Second**: Maximum achieved throughput
- **Sustained Throughput**: Performance after JIT warmup stabilization

#### Latency Metrics
- **First Token Latency**: Time from prompt to first generated token
- **Average Token Latency**: Mean time per token generation
- **Model Loading Time**: Time to initialize model and weights
- **Cold Start Overhead**: JIT compilation and warmup costs

#### Memory Metrics
- **Peak Memory Usage**: Maximum GPU/CPU memory consumption
- **Model Memory**: Memory used by model weights
- **Activation Memory**: Memory used by intermediate computations
- **Cache Memory**: Memory used for KV cache and kernel cache

#### Efficiency Metrics
- **Memory Efficiency**: Tokens/second per GB of memory used
- **JIT Compilation Benefit**: Performance improvement after warmup
- **Quantization Impact**: Performance vs memory trade-offs

### Advanced Performance Analysis

#### JIT Compilation Analysis
```bash
# JIT performance profiling
uv run -- --benchmark --timing --iterations 10

# Expected output shows JIT benefits:
# Iteration 1: 75.2ms (JIT compilation overhead)
# Iteration 2: 1.0ms (75x speedup after compilation)
# Iterations 3-10: ~1.0ms (steady-state performance)
```

#### Memory Usage Profiling
```bash
# Memory consumption analysis
uv run -- --benchmark --profile --model llama3-8b

# Shows memory breakdown:
# - Model weights: 8.2GB
# - Activations: 2.1GB
# - KV cache: 0.8GB
# - Framework overhead: 0.3GB
# - Total: 11.4GB
```

## Benchmark Results

### TinyGrad vs PyTorch Comparison (LLaMA 3-1B)

**Fair Comparison Results** (with equivalent optimizations):

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
Precision                    mixed |      float16 |       float32

🏁 Performance Analysis:
  📥 TinyGrad loads 2.0x faster than optimized PyTorch
  ❄️  TinyGrad has 7.7x faster cold start than optimized PyTorch
  🔥 TinyGrad achieves 2.0x higher steady-state throughput
  💾 Memory usage varies by precision: float16 < mixed < float32
```

### Key Performance Insights

#### TinyGrad Advantages
- **2.0x Throughput**: Superior token generation speed
- **2.0x Faster Loading**: More efficient weight loading from GGUF
- **7.7x Faster Cold Start**: Superior JIT compilation architecture
- **Automatic Optimization**: No manual optimization required

#### PyTorch Advantages
- **Better Memory Efficiency**: float16 uses ~45% less memory than mixed precision
- **Ecosystem Integration**: Full HuggingFace compatibility
- **Production Stability**: Battle-tested inference reliability
- **Broader Hardware Support**: CPU, MPS, and diverse GPU support

#### Hybrid Approach Benefits
- **Zero-Copy Operations**: Efficient tensor sharing between frameworks
- **Kernel Fusion**: TinyGrad's optimized kernels with PyTorch ecosystem
- **Fallback Mechanisms**: Graceful degradation when optimizations fail
- **Best-of-Both**: Performance + ecosystem compatibility

## Fairness Analysis

### Historical Fairness Issues (Resolved)

#### Initial Unfair Comparison
**Problems Identified**:
- **Precision Mismatch**: TinyGrad (mixed) vs PyTorch (float32)
- **Optimization Disparity**: TinyGrad JIT vs PyTorch uncompiled
- **Memory Usage**: 2x difference due to precision mismatch

#### Fair Comparison Implementation
**Solutions Applied**:
```python
# Automatic precision matching
use_half = kwargs.get("use_half", True)  # Match TinyGrad mixed precision
if use_half:
    model = model.half()  # Use float16 like TinyGrad

# JIT compilation equivalence
if use_compile and hasattr(torch, 'compile'):
    model = torch.compile(model)  # Equivalent to TinyGrad JIT
```

**Fairness Controls**:
- `--fair-comparison` (default): Applies equivalent optimizations
- `--pytorch-no-compile`: Disables PyTorch optimization for testing
- `--pytorch-no-half`: Forces float32 for precision impact analysis

### What Makes This Comparison Fair

**✅ Equivalent Conditions**:
- Same model architecture and weights
- Same precision (16-bit) and optimization levels
- Same hardware and CUDA environment
- Same input patterns and test methodology

**⚠️ Remaining Legitimate Differences**:
- **JIT Architecture**: TinyGrad's automatic JIT vs PyTorch's optional torch.compile
- **Memory Layout**: Different internal memory management strategies
- **Kernel Fusion**: Different compute graph optimization approaches
- **Weight Loading**: Different GGUF parsing efficiency

These represent genuine framework design differences, not benchmarking artifacts.

## Advanced Benchmarking

### Custom Benchmark Configurations

#### Model Size Scaling
```bash
# Compare performance across model sizes
for model in llama3-1b llama3-8b llama3-70b; do
    uv run src/benchmark.py --framework tinygrad --model-type llama --model-path ~/models/$model-instruct/
done
```

#### Quantization Impact Analysis
```bash
# Compare quantization methods
for quant in int8 nf4 float16; do
    uv run -- --benchmark --model llama3-8b --quantize $quant --iterations 10
done
```

#### Multi-GPU Scaling
```bash
# Test multi-GPU performance scaling
for shards in 1 2 4; do
    uv run -- --benchmark --model llama3-70b --shard $shards --iterations 5
done
```

### Statistical Analysis

#### Performance Variance Analysis
```python
# Statistical significance testing
import scipy.stats as stats

# Collect multiple benchmark runs
tinygrad_results = [run_benchmark("tinygrad") for _ in range(20)]
pytorch_results = [run_benchmark("pytorch") for _ in range(20)]

# Statistical significance test
statistic, p_value = stats.ttest_ind(tinygrad_results, pytorch_results)
print(f"Performance difference significance: p={p_value:.4f}")
```

#### Regression Testing
```python
# Performance regression detection
def detect_performance_regression(baseline_results, current_results, threshold=0.05):
    baseline_mean = np.mean(baseline_results)
    current_mean = np.mean(current_results)

    regression = (baseline_mean - current_mean) / baseline_mean
    return regression > threshold, regression
```

## Troubleshooting Benchmarks

### Common Issues

#### torch.compile Errors
```bash
# Issue: torch.compile causes CUDA driver errors
# Solution: Disable compilation for testing
uv run src/benchmark.py --pytorch-no-compile --model-type llama --model-path ~/models/llama3-1b-instruct/
```

#### Memory Issues
```bash
# Issue: GPU out of memory
# Solution: Use quantization or CPU fallback
uv run -- --benchmark --model llama3-8b --quantize int8

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
uv run -- --benchmark --model llama3-1b
```

#### Model Loading Issues
```bash
# Issue: Model not found or corrupt
# Solution: Force re-download
uv run -- --benchmark --model llama3-1b --download
```

### Benchmark Validation

#### Output Quality Verification
```python
# Ensure frameworks produce equivalent outputs
def validate_output_quality(prompt="Explain quantum computing"):
    tinygrad_output = generate_with_tinygrad(prompt, seed=42)
    pytorch_output = generate_with_pytorch(prompt, seed=42)

    # Semantic similarity check (outputs may vary slightly)
    similarity = compute_semantic_similarity(tinygrad_output, pytorch_output)
    assert similarity > 0.8, f"Output quality differs: {similarity}"
```

#### Performance Consistency Check
```python
# Verify benchmark reproducibility
def check_benchmark_consistency(iterations=10):
    results = [run_benchmark("tinygrad") for _ in range(iterations)]
    coefficient_of_variation = np.std(results) / np.mean(results)
    assert coefficient_of_variation < 0.1, f"High variance: {coefficient_of_variation}"
```

## Integration with CI/CD

### Automated Performance Testing
```yaml
# .github/workflows/performance.yml
name: Performance Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: self-hosted
    steps:
    - uses: actions/checkout@v3
    - name: Run Benchmarks
      run: |
        uv run src/benchmark.py --framework tinygrad pytorch --iterations 5

    - name: Performance Regression Check
      run: |
        python scripts/check_performance_regression.py
```

### Performance Monitoring
```python
# Continuous performance monitoring
def track_performance_over_time():
    result = run_benchmark("tinygrad", iterations=10)

    # Store results with metadata
    performance_data = {
        'timestamp': datetime.now(),
        'git_commit': get_git_commit(),
        'gpu_model': get_gpu_info(),
        'throughput': result.avg_tokens_per_second,
        'latency': result.avg_token_time,
        'memory': result.peak_memory_gb
    }

    store_performance_metrics(performance_data)
```

## Related Documentation

- **[User Guide](user-guide.md)**: Basic usage and configuration options
- **[Architecture Overview](architecture.md)**: Technical implementation details
- **[Development Guide](development.md)**: Contributing and code quality
- **[Research Findings](research.md)**: Advanced optimization research and TinyGrad analysis
- **[Fair Benchmark Analysis](fair-benchmark.md)**: Detailed fairness methodology

---

**Next Steps**: See the [Architecture Overview](architecture.md) for technical implementation details of the benchmarking system.