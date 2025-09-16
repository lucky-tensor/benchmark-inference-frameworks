#!/usr/bin/env python3
"""
Standalone PyTorch entry point.
Run PyTorch inference without any TinyGrad dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from backends.pytorch_backend import main
    main()
