from pathlib import Path
from typing import List

import pytest


def _torch_cuda_available() -> bool:
    """Check if CUDA is available from PyTorch.

    Returns
    -------
    bool
        True if CUDA is available from PyTorch, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Modify pytest collection to skip tests that require CUDA when CUDA is not available."""
    if _torch_cuda_available():
        return

    skip_cuda = pytest.mark.skip(reason="CUDA not available from PyTorch, skip test")
    for item in items:
        if any(item.iter_markers(name="requires_torch_cuda")):
            item.add_marker(skip_cuda)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use isolated temporary directory for cache in all tests.

    This fixture automatically sets FIB_CACHE_PATH to a unique temporary
    directory for each test, preventing cache pollution between tests.
    """
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path))
    return tmp_path
