#!/usr/bin/env python3
"""
Standalone TinyGrad entry point.
Run TinyGrad inference without any PyTorch dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from backends.tinygrad_backend import main
    main()
