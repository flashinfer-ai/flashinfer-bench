from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use isolated temporary directory for cache in all tests.

    This fixture automatically sets FIB_CACHE_PATH to a unique temporary
    directory for each test, preventing cache pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    return cache_dir
