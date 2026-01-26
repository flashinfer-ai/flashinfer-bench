"""Tests for the transformers sampling adapter."""

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
def test_softmax_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the softmax adapter.
    """
    # Small shapes for testing
    B = 2  # batch size
    V = 100  # vocab size

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensor (logits)
    logits = torch.randn((B, V), dtype=dtype, device=device)

    def_name = f"softmax_d{V}"

    definition = Definition(
        name=def_name,
        op_type="sampling",
        axes={"V": AxisConst(value=V), "M": AxisVar()},
        inputs={
            "input": TensorSpec(shape=["M", "V"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["M", "V"], dtype="bfloat16")},
        reference=(
            "import torch\n"
            "def run(input):\n"
            "    return torch.nn.functional.softmax(input, dim=-1)\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(input):\n"
            "    return '__SUB__softmax__'\n"
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
        description="Test solution for softmax adapter",
    )

    workload = Workload(
        axes={"M": B},
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
        out = torch.nn.functional.softmax(logits, dim=-1)
        assert out == "__SUB__softmax__"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_topk_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the top-k adapter.
    """
    # Small shapes for testing
    B = 2  # batch size
    V = 100  # vocab size
    K = 10  # top-k

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensor (logits)
    logits = torch.randn((B, V), dtype=dtype, device=device)

    def_name = f"topk_d{V}_k{K}"

    definition = Definition(
        name=def_name,
        op_type="sampling",
        axes={"V": AxisConst(value=V), "K": AxisConst(value=K), "M": AxisVar()},
        inputs={
            "input": TensorSpec(shape=["M", "V"], dtype="bfloat16"),
            "k": TensorSpec(shape=None, dtype="int64"),
        },
        outputs={
            "values": TensorSpec(shape=["M", "K"], dtype="bfloat16"),
            "indices": TensorSpec(shape=["M", "K"], dtype="int64"),
        },
        reference=(
            "import torch\n"
            "def run(input, k):\n"
            "    return torch.topk(input, k, dim=-1)\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(input, k):\n"
            "    return '__SUB__topk__'\n"
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
        description="Test solution for top-k adapter",
    )

    workload = Workload(
        axes={"M": B},
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
        out = torch.topk(logits, K, dim=-1)
        assert out == "__SUB__topk__"


if __name__ == "__main__":
    pytest.main(sys.argv)
