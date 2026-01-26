"""Tests for the transformers activation adapter."""

import sys

import pytest
import torch

from flashinfer_bench.apply import ApplyConfig, ApplyRuntime
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_silu_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the SiLU activation adapter.
    """
    # Small shapes for testing
    B = 2  # batch size
    S = 4  # sequence length
    H = 16  # hidden size

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensor
    inp = torch.randn((B, S, H), dtype=dtype, device=device)

    def_name = f"silu_h{H}"

    definition = Definition(
        name=def_name,
        op_type="activation",
        axes={"H": AxisConst(value=H), "M": AxisVar()},
        inputs={
            "input": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype="bfloat16")},
        reference=(
            "import torch\n"
            "def run(input):\n"
            "    return torch.nn.functional.silu(input)\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(input):\n"
            "    return '__SUB__silu__'\n"
        ),
    )

    solution = Solution(
        name=f"{def_name}__python_direct_call",
        definition=def_name,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Test solution for SiLU adapter",
    )

    workload = Workload(
        axes={"M": B * S},
        inputs={"input": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="cuda", libs={}),
            timestamp="now",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )

    trace_set = TraceSet(
        root=tmp_path,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        traces={def_name: [trace]},
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    runtime = ApplyRuntime(trace_set, ApplyConfig())

    with runtime:
        # Call the function; adapter should patch it
        out = torch.nn.functional.silu(inp)
        assert out == "__SUB__silu__"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gelu_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the GELU activation adapter.
    """
    # Small shapes for testing
    B = 2  # batch size
    S = 4  # sequence length
    H = 16  # hidden size

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensor
    inp = torch.randn((B, S, H), dtype=dtype, device=device)

    def_name = f"gelu_h{H}"

    definition = Definition(
        name=def_name,
        op_type="activation",
        axes={"H": AxisConst(value=H), "M": AxisVar()},
        inputs={
            "input": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype="bfloat16")},
        reference=(
            "import torch\n"
            "def run(input):\n"
            "    return torch.nn.functional.gelu(input)\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(input):\n"
            "    return '__SUB__gelu__'\n"
        ),
    )

    solution = Solution(
        name=f"{def_name}__python_direct_call",
        definition=def_name,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Test solution for GELU adapter",
    )

    workload = Workload(
        axes={"M": B * S},
        inputs={"input": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="cuda", libs={}),
            timestamp="now",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )

    trace_set = TraceSet(
        root=tmp_path,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        traces={def_name: [trace]},
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    runtime = ApplyRuntime(trace_set, ApplyConfig())

    with runtime:
        # Call the function; adapter should patch it
        out = torch.nn.functional.gelu(inp)
        assert out == "__SUB__gelu__"


if __name__ == "__main__":
    pytest.main(sys.argv)
