"""Tests for the transformers MoE adapter."""

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
def test_moe_batched_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the MoE batched_mm adapter.
    """
    try:
        from transformers.integrations.moe import batched_mm_experts_forward
    except ImportError:
        pytest.skip("transformers not installed or MoE integration not available")

    # Small shapes for testing
    N = 8  # num tokens
    E = 4  # num experts
    H = 16  # hidden dim
    I = 32  # intermediate dim
    K = 2  # top-k

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensors
    hidden_states = torch.randn((N, H), dtype=dtype, device=device)
    top_k_index = torch.randint(0, E, (N, K), device=device)
    top_k_weights = torch.randn((N, K), dtype=dtype, device=device)
    top_k_weights = torch.softmax(top_k_weights, dim=-1)

    def_name = f"moe_batched_e{E}_h{H}_i{I}_topk{K}"

    definition = Definition(
        name=def_name,
        op_type="moe",
        axes={
            "E": AxisConst(value=E),
            "H": AxisConst(value=H),
            "I": AxisConst(value=I),
            "K": AxisConst(value=K),
            "N": AxisVar(),
        },
        inputs={
            "hidden_states": TensorSpec(shape=["N", "H"], dtype="bfloat16"),
            "top_k_index": TensorSpec(shape=["N", "K"], dtype="int64"),
            "top_k_weights": TensorSpec(shape=["N", "K"], dtype="bfloat16"),
            "gate_up_proj": TensorSpec(shape=["E", "I*2", "H"], dtype="bfloat16"),
            "down_proj": TensorSpec(shape=["E", "H", "I"], dtype="bfloat16"),
        },
        outputs={"output": TensorSpec(shape=["N", "H"], dtype="bfloat16")},
        reference=(
            "def run(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj):\n"
            "    return hidden_states\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj):\n"
            "    return '__SUB__moe__'\n"
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
        description="Test solution for MoE adapter",
    )

    workload = Workload(
        axes={"N": N},
        inputs={"hidden_states": RandomInput(), "top_k_index": RandomInput(), "top_k_weights": RandomInput()},
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

    # Create a mock module with required attributes
    class MockExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(E, 2 * I, H, dtype=dtype, device=device)
            )
            self.down_proj = torch.nn.Parameter(
                torch.randn(E, H, I, dtype=dtype, device=device)
            )
            self.is_transposed = False
            self.has_bias = False
            self.act_fn = torch.nn.functional.silu

    mock_module = MockExperts()

    with runtime:
        # Call the function; adapter should patch it
        out = batched_mm_experts_forward(mock_module, hidden_states, top_k_index, top_k_weights)
        assert out == "__SUB__moe__"


if __name__ == "__main__":
    pytest.main(sys.argv)
