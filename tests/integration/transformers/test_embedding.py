"""Tests for the transformers embedding adapter."""

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
def test_embedding_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the transformers embedding adapter.

    It constructs a minimal TraceSet containing a python solution for a
    definition whose name matches the embedding adapter resolver
    (embedding_v{vocab_size}_d{embedding_dim}).
    """
    # Small shapes for testing
    V = 100  # vocab size
    D = 16  # embedding dim
    B = 2  # batch size
    S = 4  # sequence length

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensors
    input_ids = torch.randint(0, V, (B, S), device=device)
    weight = torch.randn((V, D), dtype=dtype, device=device)

    def_name = f"embedding_v{V}_d{D}"

    definition = Definition(
        name=def_name,
        op_type="embedding",
        axes={"V": AxisConst(value=V), "D": AxisConst(value=D), "N": AxisVar()},
        inputs={
            "input_ids": TensorSpec(shape=["N"], dtype="int64"),
            "weight": TensorSpec(shape=["V", "D"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["N", "D"], dtype="bfloat16")},
        reference=(
            "def run(input_ids, weight):\n"
            "    return weight[input_ids]\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(input_ids, weight):\n"
            "    return '__SUB__embedding__'\n"
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
        description="Test solution for embedding adapter",
    )

    workload = Workload(
        axes={"N": B * S},
        inputs={"input_ids": RandomInput()},
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
        out = torch.nn.functional.embedding(input_ids, weight)
        assert out == "__SUB__embedding__"


if __name__ == "__main__":
    pytest.main(sys.argv)
