import sys

import pytest

from flashinfer_bench.compile import Builder, Runnable
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
    def can_build(self, solution: Solution) -> bool:
        return True

    def _make_key(self, solution: Solution) -> str:
        return f"dummy::{solution.name}"

    def _make_closer(self):
        return lambda: None

    def _build(self, definition: Definition, solution: Solution) -> Runnable:
        return Runnable(fn=lambda **kw: kw, closer=self._make_closer(), meta={"dummy": True})


def test_builder_cache_and_key():
    b = DummyBuilder()
    d = Definition(
        name="test_def",
        op_type="op",
        axes={"M": AxisConst(value=1)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    srcs = [SourceFile(path="main.py", content="def run(A):\n    return A\n")]
    s = Solution(name="s1", definition="test_def", author="me", spec=spec, sources=srcs)
    r1 = b.build(d, s)
    r2 = b.build(d, s)
    assert r1 is r2  # cache hit via _make_key
    b.clear_cache()


if __name__ == "__main__":
    pytest.main(sys.argv)
