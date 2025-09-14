#!/usr/bin/env python3
"""Convenience wrapper for LLaMA 3 implementation."""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main LLaMA implementation
from src.llama.llama3 import main

if __name__ == "__main__":
    main()