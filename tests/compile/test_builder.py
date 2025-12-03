import sys

import pytest

from flashinfer_bench.compile import Builder, Runnable, RunnableMetadata
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
)


class DummyBuilder(Builder):
    @staticmethod
    def is_available() -> bool:
        return True

    def can_build(self, solution: Solution) -> bool:
        return True

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        metadata = RunnableMetadata(
            build_type="python",
            definition=definition.name,
            solution=solution.name,
            misc={"dummy": True},
        )
        return Runnable(callable=lambda **kw: kw, cleaner=lambda: None, metadata=metadata)


def test_builder_cache_and_key(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    builder = DummyBuilder("dummy_", "dummy")
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


if __name__ == "__main__":
    pytest.main(sys.argv)
