import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flashinfer_bench.compile import Builder, Runnable, RunnableMetadata
from flashinfer_bench.compile.registry import BuilderRegistry
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
)


@pytest.fixture(autouse=True)
def _use_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """Automatically use tmp_cache_dir for all tests in this module."""


def test_builder_cache_and_key():
    class DummyBuilder(Builder):
        def __init__(self) -> None:
            super().__init__("dummy_", "dummy")

        @staticmethod
        def is_available() -> bool:
            return True

        def can_build(self, solution: Solution) -> bool:
            return True

        def build(self, definition: Definition, solution: Solution) -> Runnable:
            metadata = RunnableMetadata(
                build_type="python",
                definition_name=definition.name,
                solution_name=solution.name,
                misc={"dummy": True},
            )
            return Runnable(callable=lambda **kw: kw, cleaner=lambda: None, metadata=metadata)

    builder = DummyBuilder()
    definition = Definition(
        name="test_def",
        op_type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )
    solution = Solution(
        name="s1",
        definition="test_def",
        author="me",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(A):\n    return A\n")],
    )
    r1 = builder.build(definition, solution)
    r2 = builder.build(definition, solution)
    # Both builds return Runnable objects
    assert r1 is not None
    assert r2 is not None


def _create_mock_builder(name: str, can_build_result: bool = True) -> MagicMock:
    """Create a mock builder for testing dispatch logic."""
    builder = MagicMock(spec=Builder)
    builder.can_build.return_value = can_build_result
    builder.build.side_effect = lambda *args, **kwargs: Runnable(
        callable=lambda **kw: kw,
        cleaner=lambda: None,
        metadata=RunnableMetadata(build_type=name, definition_name="", solution_name="", misc={}),
    )
    return builder


def _make_test_solution(name: str = "test_solution") -> Solution:
    """Create a simple test solution."""
    return Solution(
        name=name,
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(A):\n    return A\n")],
    )


def _make_test_definition(name: str = "test_def") -> Definition:
    """Create a simple test definition."""
    return Definition(
        name=name,
        op_type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )


def test_dispatch_by_can_build():
    """Test that registry dispatches to the first builder where can_build returns True."""
    builder1 = _create_mock_builder("builder1", can_build_result=False)
    builder2 = _create_mock_builder("builder2", can_build_result=True)

    registry = BuilderRegistry([builder1, builder2])
    registry.build(_make_test_definition(), _make_test_solution())

    builder1.build.assert_not_called()
    builder2.build.assert_called_once()


def test_dispatch_by_is_available(monkeypatch: pytest.MonkeyPatch):
    """Test that only builders where is_available returns True are registered."""
    from flashinfer_bench.compile import registry as registry_module
    from flashinfer_bench.compile.builders import PythonBuilder

    class UnavailableBuilder(PythonBuilder):
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(registry_module, "_BUILDER_PRIORITY", [UnavailableBuilder, PythonBuilder])
    monkeypatch.setattr(registry_module.BuilderRegistry, "_instance", None)

    registry = BuilderRegistry.get_instance()

    # Only PythonBuilder should be registered (UnavailableBuilder.is_available returns False)
    assert len(registry._builders) == 1
    assert isinstance(registry._builders[0], PythonBuilder)
    assert not isinstance(registry._builders[0], UnavailableBuilder)


def test_build_registry_cache_hit():
    """Test that building the same solution twice returns cached result."""
    builder = _create_mock_builder("builder", can_build_result=True)
    registry = BuilderRegistry([builder])
    definition = _make_test_definition()
    solution1 = _make_test_solution()
    solution2 = _make_test_solution()

    runnable1 = registry.build(definition, solution1)
    runnable2 = registry.build(definition, solution2)

    assert builder.build.call_count == 1
    assert runnable1 is runnable2


def test_build_registry_cache_miss():
    """Test that building different solutions results in a cache miss."""
    builder = _create_mock_builder("builder", can_build_result=True)
    registry = BuilderRegistry([builder])
    definition = _make_test_definition()
    solution1 = _make_test_solution("solution1")
    solution2 = _make_test_solution("solution2")

    runnable1 = registry.build(definition, solution1)
    runnable2 = registry.build(definition, solution2)

    assert builder.build.call_count == 2
    assert runnable1 is not runnable2


def test_build_registry_cleanup():
    """Test that cleanup() removes cached runnables."""
    builder = _create_mock_builder("builder", can_build_result=True)
    registry = BuilderRegistry([builder])
    definition = _make_test_definition()
    solution = _make_test_solution()

    registry.build(definition, solution)
    assert builder.build.call_count == 1

    registry.cleanup()
    registry.build(definition, solution)
    assert builder.build.call_count == 2


if __name__ == "__main__":
    pytest.main(sys.argv)
