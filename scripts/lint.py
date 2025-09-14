#!/usr/bin/env python3
"""
Linting script for the tinygrad-demo project.

This script provides convenient commands for running linting tools.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    else:
        return True


def main():
    parser = argparse.ArgumentParser(description="Run linting tools on the codebase")
    parser.add_argument("--check", action="store_true", help="Check code without fixing issues")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically where possible")
    parser.add_argument("--security", action="store_true", help="Run security checks with Bandit")
    parser.add_argument("--all", action="store_true", help="Run all checks and fixes")
    parser.add_argument("--files", nargs="*", help="Specific files to check")

    args = parser.parse_args()

    if not any([args.check, args.fix, args.security, args.all]):
        args.check = True  # Default to check mode

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    success = True

    if args.all or args.check:
        print("=" * 60)
        print("RUNNING RUFF LINTER (CHECK MODE)")
        print("=" * 60)
        cmd = ["uv", "run", "ruff", "check"]
        if args.files:
            cmd.extend(args.files)
        success = run_command(cmd, "Ruff linter check") and success

    if args.all or args.fix:
        print("\n" + "=" * 60)
        print("RUNNING RUFF LINTER (FIX MODE)")
        print("=" * 60)
        cmd = ["uv", "run", "ruff", "check", "--fix"]
        if args.files:
            cmd.extend(args.files)
        success = run_command(cmd, "Ruff linter fix") and success

        print("\n" + "=" * 60)
        print("RUNNING RUFF FORMATTER")
        print("=" * 60)
        cmd = ["uv", "run", "ruff", "format"]
        if args.files:
            cmd.extend(args.files)
        success = run_command(cmd, "Ruff formatter") and success

    if args.all or args.security:
        print("\n" + "=" * 60)
        print("RUNNING BANDIT SECURITY CHECKS")
        print("=" * 60)
        cmd = ["uv", "run", "bandit", "-r", "."]
        if args.files:
            cmd = ["uv", "run", "bandit", *args.files]
        success = run_command(cmd, "Bandit security check") and success

    if success:
        print("\n✅ All linting checks passed!")
        return 0
    print("\n❌ Some linting checks failed!")
    return 1


if __name__ == "__main__":
    import os

    sys.exit(main())
