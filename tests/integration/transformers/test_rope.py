"""Tests for the transformers RoPE (Rotary Position Embedding) adapter."""

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
def test_rope_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the RoPE adapter.

    It constructs a minimal TraceSet containing a python solution for a
    definition whose name matches the RoPE adapter resolver
    (rope_h{num_heads}_d{head_dim}).
    """
    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        pytest.skip("transformers not installed")

    # Small shapes for testing
    B = 2  # batch size
    H = 4  # num heads
    S = 8  # sequence length
    D = 16  # head dim

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build tensors matching RoPE expected shapes
    # q, k: [batch, num_heads, seq_len, head_dim]
    q = torch.randn((B, H, S, D), dtype=dtype, device=device)
    k = torch.randn((B, H, S, D), dtype=dtype, device=device)
    # cos, sin: [batch, seq_len, head_dim]
    cos = torch.randn((B, S, D), dtype=dtype, device=device)
    sin = torch.randn((B, S, D), dtype=dtype, device=device)

    def_name = f"rope_h{H}_d{D}"

    definition = Definition(
        name=def_name,
        op_type="rope",
        axes={
            "H": AxisConst(value=H),
            "D": AxisConst(value=D),
            "B": AxisVar(),
            "S": AxisVar(),
        },
        inputs={
            "q": TensorSpec(shape=["B", "H", "S", "D"], dtype="bfloat16"),
            "k": TensorSpec(shape=["B", "H", "S", "D"], dtype="bfloat16"),
            "cos": TensorSpec(shape=["B", "S", "D"], dtype="bfloat16"),
            "sin": TensorSpec(shape=["B", "S", "D"], dtype="bfloat16"),
        },
        outputs={
            "q_embed": TensorSpec(shape=["B", "H", "S", "D"], dtype="bfloat16"),
            "k_embed": TensorSpec(shape=["B", "H", "S", "D"], dtype="bfloat16"),
        },
        reference=(
            "def run(q, k, cos, sin):\n"
            "    return q, k\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(q, k, cos, sin):\n"
            "    return '__SUB__rope__'\n"
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
        description="Test solution for RoPE adapter",
    )

    workload = Workload(
        axes={"B": B, "S": S},
        inputs={"q": RandomInput(), "k": RandomInput(), "cos": RandomInput(), "sin": RandomInput()},
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
        out = apply_rotary_pos_emb(q, k, cos, sin)
        assert out == "__SUB__rope__"


if __name__ == "__main__":
    pytest.main(sys.argv)
