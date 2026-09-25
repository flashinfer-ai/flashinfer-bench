import sys
import textwrap
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
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="pkg/main.py", content="def run(A, B):\n    return A")],
    )

    builder = PythonBuilder()
    runnable = builder.build(definition, solution)

    # Call runnable with torch tensors
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    out = runnable(A, B)
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
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
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
    out = runnable(X, Y)
    expected = torch.tensor([[6, 8], [10, 12]], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_python_builder_dps_optional_extra_param():
    definition = Definition(
        name="copy_dps",
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
        name="copy_dps_py",
        definition="copy_dps",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[
            SourceFile(
                path="main.py",
                content="""
def run(A=None, B=None, C=None, stream=None):
    C.copy_(A)
""",
            )
        ],
    )

    builder = PythonBuilder()
    runnable = builder.build(definition, solution)
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
    C = torch.empty_like(A)
    runnable(A, B, C)
    assert torch.allclose(C, A)


def _make_pass_through_def() -> Definition:
    return Definition(
        name="pass",
        op_type="op",
        axes={"M": AxisConst(value=2)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )


def _make_solution(content: str, name: str = "py_sol") -> Solution:
    return Solution(
        name=name,
        definition="pass",
        author="me",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="main.py", content=content)],
    )


def test_python_builder_detects_optional_setup_symbol():
    """PythonBuilder picks up a top-level `setup` symbol and passes it to Runnable."""
    content = textwrap.dedent(
        """
        def setup(A, B):
            return {"scale": 7}

        def run(A, B, *, scale):
            B.copy_(A * scale)
        """
    )
    definition = _make_pass_through_def()
    solution = _make_solution(content, name="py_sol_with_setup")
    builder = PythonBuilder()
    runnable = builder.build(definition, solution)

    assert runnable._setup_callable is not None
    assert callable(runnable._setup_callable)

    # End-to-end: setup_for_workload + call splats kwargs in
    A = torch.tensor([1.0, 2.0], dtype=torch.float32)
    B = torch.zeros((2,), dtype=torch.float32)
    runnable.setup_for_workload(A, B)
    runnable(A, B)
    assert torch.allclose(B, torch.tensor([7.0, 14.0]))


def test_python_builder_no_setup_symbol_keeps_setup_callable_none():
    """When the solution module has no `setup` symbol, setup_callable stays None."""
    content = "def run(A, B):\n    B.copy_(A)\n"
    definition = _make_pass_through_def()
    solution = _make_solution(content, name="py_sol_no_setup")
    builder = PythonBuilder()
    runnable = builder.build(definition, solution)
    assert runnable._setup_callable is None


def test_python_builder_ignores_non_callable_setup_symbol():
    """A `setup` name that is not callable (e.g. a constant) is treated as if absent."""
    content = textwrap.dedent(
        """
        setup = "not callable"

        def run(A, B):
            B.copy_(A)
        """
    )
    definition = _make_pass_through_def()
    solution = _make_solution(content, name="py_sol_bad_setup")
    builder = PythonBuilder()
    runnable = builder.build(definition, solution)
    assert runnable._setup_callable is None


def test_python_builder_ignores_setup_with_mismatched_signature():
    """Name-capture guard: a coincidental top-level `setup` whose signature cannot
    bind run's positional args (e.g. `from setuptools import setup`, or a local
    helper) must NOT silently become a per-workload hook — that would change how
    an existing solution is invoked and timed."""
    content = textwrap.dedent(
        """
        def setup(**kwargs):  # setuptools-style — cannot bind positional args
            raise AssertionError("must never be called as a hook")

        def run(A, B):
            B.copy_(A)
        """
    )
    definition = _make_pass_through_def()
    solution = _make_solution(content, name="py_sol_capture_kwargs")
    runnable = PythonBuilder().build(definition, solution)
    assert runnable._setup_callable is None

    content2 = textwrap.dedent(
        """
        def setup(config):  # arity mismatch vs run(A, B)
            raise AssertionError("must never be called as a hook")

        def run(A, B):
            B.copy_(A)
        """
    )
    solution2 = _make_solution(content2, name="py_sol_capture_arity")
    runnable2 = PythonBuilder().build(definition, solution2)
    assert runnable2._setup_callable is None

    # And the solution still evaluates as a plain (hookless) runnable.
    A = torch.tensor([1.0, 2.0], dtype=torch.float32)
    B = torch.zeros((2,), dtype=torch.float32)
    runnable2.setup_for_workload(A, B)  # no-op
    runnable2(A, B)
    assert torch.allclose(B, A)


def test_python_builder_required_kwonly_run_needs_matching_setup():
    """run() with required kw-only params is only buildable when a matching setup
    hook provides them; if the setup symbol is ignored by the name-capture guard,
    the build must fail at COMPILE time exactly like pre-hook code."""
    from flashinfer_bench.compile.builder import BuildError

    content = textwrap.dedent(
        """
        def setup(config):  # mismatched — ignored by the guard
            return {"scale": 2}

        def run(A, B, *, scale):
            B.copy_(A * scale)
        """
    )
    definition = _make_pass_through_def()
    solution = _make_solution(content, name="py_sol_kwonly_no_hook")
    with pytest.raises(BuildError):
        PythonBuilder().build(definition, solution)


if __name__ == "__main__":
    pytest.main(sys.argv)
