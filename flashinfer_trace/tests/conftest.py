"""Pytest configuration for flashinfer_trace tests."""

import os
from pathlib import Path

# Skip references directory during test collection
collect_ignore = ["references"]


def pytest_configure(config):
    """Set FIB_DATASET_PATH to flashinfer_trace root for these tests."""
    trace_root = Path(__file__).parent.parent
    os.environ["FIB_DATASET_PATH"] = str(trace_root)
