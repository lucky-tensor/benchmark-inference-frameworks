#!/usr/bin/env python3
"""
Test different PyTorch compilation options to find the fastest one.
"""

import subprocess
import time
from pathlib import Path

# Compilation options to test
COMPILATION_OPTIONS = [
    ("no_compile", "No compilation (baseline)"),
    ("aot_eager", "AOT Eager compilation"),
    ("eager", "Eager compilation"),
    ("inductor", "Inductor compilation"),
    ("inductor_reduce_overhead", "Inductor with reduce-overhead mode"),
]


def run_pytorch_benchmark_with_option(option_name, description, iterations=10):
    """Run PyTorch benchmark with specific compilation option."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Testing: {description}")
    print(f"{'=' * 60}")

    # Get the appropriate framework name for this option
    framework_name = modify_benchmark_for_option(option_name)

    # Run the benchmark using benchmark_v2.py
    cmd = [
        "uv",
        "run",
        "src/benchmark_v2.py",
        "--framework",
        framework_name,
        "--model-id",
        "llama3-1b",
        "--model-path",
        str(Path.home() / "models/llama3-1b-instruct/"),
        "--iterations",
        str(iterations),
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()

        if result.returncode == 0:
            # Parse the output to extract performance metrics
            output_lines = result.stdout.split("\n")

            # Look for throughput metrics
            avg_throughput = None
            peak_throughput = None
            steady_state_throughput = None

            for line in output_lines:
                if "Average throughput:" in line:
                    avg_throughput = float(line.split(":")[1].strip().split()[0])
                elif "Peak throughput:" in line:
                    peak_throughput = float(line.split(":")[1].strip().split()[0])
                elif "Steady-state avg:" in line:
                    steady_state_throughput = float(line.split(":")[1].strip().split()[0])

            total_time = end_time - start_time

            return {
                "option": option_name,
                "description": description,
                "success": True,
                "avg_throughput": avg_throughput,
                "peak_throughput": peak_throughput,
                "steady_state_throughput": steady_state_throughput,
                "total_time": total_time,
                "output": result.stdout,
            }
        print(f"❌ Benchmark failed with return code {result.returncode}")
        print("STDERR:", result.stderr)
        return {
            "option": option_name,
            "description": description,
            "success": False,
            "error": result.stderr,
            "total_time": end_time - start_time,
        }

    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out after 5 minutes")
        return {
            "option": option_name,
            "description": description,
            "success": False,
            "error": "Timeout",
            "total_time": 300,
        }
    except Exception as e:
        print(f"❌ Benchmark failed with exception: {e}")
        return {"option": option_name, "description": description, "success": False, "error": str(e), "total_time": 0}


def modify_benchmark_for_option(option_name):
    """Map compilation option to framework name for benchmark_v2.py."""
    # Map compilation options to framework names in benchmark_v2.py
    option_to_framework = {
        "no_compile": "pytorch-unoptimized",
        "aot_eager": "pytorch-aot_eager",
        "eager": "pytorch-eager",
        "inductor": "pytorch-inductor",
        "inductor_reduce_overhead": "pytorch-inductor",
    }

    return option_to_framework.get(option_name, "pytorch-inductor")


def restore_benchmark(backup_path):
    """No-op function since we no longer modify files."""


def main():
    """Run compilation option comparison."""
    print("🚀 PyTorch Compilation Options Benchmark")
    print("=" * 60)
    print("Testing different torch.compile backends to find optimal performance")

    results = []

    # Test each compilation option
    for option_name, description in COMPILATION_OPTIONS:
        print(f"\n🔄 Preparing to test: {description}")

        # Run the benchmark
        result = run_pytorch_benchmark_with_option(option_name, description)
        results.append(result)

        # Show immediate result
        if result["success"]:
            print(f"✅ {description}: {result['steady_state_throughput']:.1f} tok/s (steady-state)")
        else:
            print(f"❌ {description}: Failed - {result.get('error', 'Unknown error')}")

        # Brief pause between tests
        time.sleep(2)

    # Analysis and reporting
    print(f"\n{'=' * 80}")
    print("📊 COMPILATION OPTIONS COMPARISON")
    print(f"{'=' * 80}")

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("❌ No successful benchmarks to compare!")
        return

    # Sort by steady-state throughput
    successful_results.sort(key=lambda x: x.get("steady_state_throughput", 0), reverse=True)

    print(f"{'Option':<25} {'Description':<30} {'Steady-State (tok/s)':<18} {'Peak (tok/s)':<12}")
    print("-" * 85)

    for result in successful_results:
        option = result["option"]
        desc = result["description"][:28] + "..." if len(result["description"]) > 28 else result["description"]
        steady = result.get("steady_state_throughput", 0)
        peak = result.get("peak_throughput", 0)

        print(f"{option:<25} {desc:<30} {steady:<18.1f} {peak:<12.1f}")

    # Determine winner
    best_result = successful_results[0]
    print(f"\n🏆 WINNER: {best_result['description']}")
    print(f"   Steady-state throughput: {best_result['steady_state_throughput']:.1f} tokens/second")
    print(f"   Peak throughput: {best_result['peak_throughput']:.1f} tokens/second")

    # Performance comparison
    if len(successful_results) > 1:
        baseline = None
        for result in successful_results:
            if result["option"] == "no_compile":
                baseline = result
                break

        if baseline and baseline != best_result:
            improvement = best_result["steady_state_throughput"] / baseline["steady_state_throughput"]
            print(f"   Improvement over no compilation: {improvement:.1f}x faster")

    print(f"\n💡 Recommendation: Use '{best_result['option']}' for best PyTorch performance")


if __name__ == "__main__":
    main()
