#!/usr/bin/env python3
"""Compatibility CLI for the packaged FlashInfer dump sanitizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer_bench.tracing.sanitize import main

if __name__ == "__main__":
    main()
