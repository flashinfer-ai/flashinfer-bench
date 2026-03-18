"""Tests for compile/utils.py."""

import sys
import time

import pytest

from flashinfer_bench.compile.utils import create_package_name, write_sources_to_path
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages


def test_write_sources_to_path(tmp_path):
    """Test that write_sources_to_path creates files correctly."""
    sources = [
        SourceFile(path="main.py", content="print('hello')"),
        SourceFile(path="pkg/helper.py", content="def helper(): pass"),
    ]

    paths = write_sources_to_path(tmp_path, sources)

    assert len(paths) == 2
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "main.py").read_text() == "print('hello')"
    assert (tmp_path / "pkg" / "helper.py").exists()
    assert (tmp_path / "pkg" / "helper.py").read_text() == "def helper(): pass"


def test_write_sources_to_path_preserves_mtime_for_unchanged_content(tmp_path):
    """Unchanged source files should not be rewritten, so build caches stay valid."""
    source = SourceFile(path="main.py", content="print('hello')")

    [path] = write_sources_to_path(tmp_path, [source])
    before_mtime = path.stat().st_mtime_ns

    time.sleep(0.01)
    [same_path] = write_sources_to_path(tmp_path, [source])
    after_mtime = same_path.stat().st_mtime_ns

    assert same_path == path
    assert after_mtime == before_mtime


def test_create_package_name():
    """Test package name creation."""
    solution = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(): pass")],
    )

    package_name = create_package_name(solution, "fib_python_")

    # Should start with prefix
    assert package_name.startswith("fib_python_")
    # Should contain normalized solution name
    assert "my_solution" in package_name
    # Should end with hash
    assert len(package_name.split("_")[-1]) == 6  # 6-char hash


def test_create_package_name_normalization():
    """Test that special characters are normalized to underscores."""
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    sources = [SourceFile(path="main.py", content="def run(): pass")]

    solution = Solution(
        name="my-solution.v1@test", definition="test_def", author="test", spec=spec, sources=sources
    )
    package_name = create_package_name(solution, "")
    assert package_name.startswith("my_solution_v1_test")

    # Special characters should be replaced with underscores
    assert "-" not in package_name
    assert "." not in package_name
    assert "@" not in package_name

    solution2 = Solution(
        name="123solution", definition="test_def", author="test", spec=spec, sources=sources
    )
    package_name2 = create_package_name(solution2, "")
    assert package_name2.startswith("_123solution")


def test_create_package_name_deterministic():
    """Test that the same solution produces the same package name."""
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    solution1 = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=spec,
        sources=[SourceFile(path="main.py", content="def run(): return 1")],
    )

    name1 = create_package_name(solution1, "prefix_")
    name2 = create_package_name(solution1, "prefix_")

    assert name1 == name2

    solution2 = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=spec,
        sources=[SourceFile(path="main.py", content="def run(): return 2")],
    )
    name3 = create_package_name(solution2, "prefix_")

    assert name1 != name3


if __name__ == "__main__":
    pytest.main(sys.argv)
