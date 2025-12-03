import sys
from pathlib import Path

import pytest
import torch

from flashinfer_bench.compile.builders import PythonBuilder
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


def test_python_builder_minimum():
    definition = Definition(
        name="mm",
        op_type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=2)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="import torch\n\ndef run(A, B):\n    return A",
    )
    solution = Solution(
        name="py_sol",
        definition="mm",
        author="me",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="pkg/main.py::run",
        ),
        sources=[SourceFile(path="pkg/main.py", content="def run(A, B):\n    return A")],
    )

    builder = PythonBuilder()
    runnable = builder.build(definition, solution)

    # Call runnable with torch tensors
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    out = runnable(A=A, B=B)
    assert torch.allclose(out, A)


def test_python_builder_add():
    definition = Definition(
        name="add",
        op_type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="import torch\n\ndef run(X, Y):\n    return X + Y",
    )
    solution = Solution(
        name="add_py",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[
            SourceFile(
                path="main.py",
                content="""
import torch
def run(X: torch.Tensor, Y: torch.Tensor):
    return X + Y
""",
            )
        ],
    )

    builder = PythonBuilder()
    runnable = builder.build(definition, solution)
    X = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    Y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    out = runnable(X=X, Y=Y)
    expected = torch.tensor([[6, 8], [10, 12]], dtype=torch.float32)
    assert torch.allclose(out, expected)


if __name__ == "__main__":
    pytest.main(sys.argv)
