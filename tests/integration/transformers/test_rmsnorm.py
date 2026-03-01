"""Tests for the transformers RMSNorm adapter."""

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


def test_rmsnorm_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the transformers RMSNorm adapter.

    It constructs a minimal TraceSet containing a python solution for a
    definition whose name matches the RMSNorm adapter resolver
    (rmsnorm_h{hidden_size}). Then it installs an ApplyRuntime and calls
    torch.nn.functional.rms_norm(...) to ensure the adapter dispatches to
    the python solution.
    """
    # Check if torch.nn.functional.rms_norm is available (PyTorch 2.4+)
    if not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("torch.nn.functional.rms_norm not available (requires PyTorch 2.4+)")

    # Small shapes for testing
    B = 2  # batch size
    S = 3  # sequence length
    H = 8  # hidden size

    device = torch.device("cpu")
    dtype = torch.bfloat16

    # Build tensors
    inp = torch.randn((B, S, H), dtype=dtype, device=device)
    weight = torch.ones((H,), dtype=dtype, device=device)

    def_name = f"rmsnorm_h{H}"

    definition = Definition(
        name=def_name,
        op_type="rmsnorm",
        axes={"M": AxisVar(), "H": AxisConst(value=H)},
        inputs={
            "hidden_states": TensorSpec(shape=["M", "H"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["H"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype="bfloat16")},
        reference=(
            "def run(hidden_states, weight):\n"
            "    return hidden_states\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(hidden_states, weight):\n"
            "    return '__SUB__rmsnorm__'\n"
        ),
    )

    solution = Solution(
        name=f"{def_name}__python_direct_call",
        definition=def_name,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Test solution for RMSNorm adapter",
    )

    workload = Workload(
        axes={"M": B * S},
        inputs={"hidden_states": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="cpu", libs={}),
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
        out = torch.nn.functional.rms_norm(inp, [H], weight, eps=1e-6)
        assert out == "__SUB__rmsnorm__"


if __name__ == "__main__":
    pytest.main(sys.argv)
