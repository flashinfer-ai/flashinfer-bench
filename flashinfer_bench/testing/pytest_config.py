"""Pytest configuration tools, such as markers and fixtures."""

import pytest

from flashinfer_bench.utils import is_torch_cuda_available

requires_torch_cuda = pytest.mark.skipif(
    not is_torch_cuda_available(), reason="CUDA not available from PyTorch"
)
