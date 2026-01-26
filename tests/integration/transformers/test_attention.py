"""Tests for the transformers attention adapter."""

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


def test_attention_adapter_substitution(tmp_path, monkeypatch):
    """
    Integration-style test for the transformers attention adapter.

    It constructs a minimal TraceSet containing a python solution for a
    definition whose name matches the attention adapter resolver
    (gqa_ragged_prefill_causal_h{num_q_heads}_kv{num_kv_heads}_d{head_dim}).
    Then it installs an ApplyRuntime and calls
    transformers.integrations.sdpa_attention.sdpa_attention_forward(...)
    to ensure the adapter dispatches to the python solution.
    """
    try:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
    except ImportError:
        pytest.skip("transformers not installed")

    # Small shapes for testing
    B = 2  # batch size
    H_q = 4  # num query heads
    H_kv = 2  # num kv heads
    S = 3  # sequence length
    D = 8  # head dim

    device = torch.device("cpu")
    dtype = torch.bfloat16

    # Build tensors matching expected shapes: [batch, num_heads, seq_len, head_dim]
    query = torch.randn((B, H_q, S, D), dtype=dtype, device=device)
    key = torch.randn((B, H_kv, S, D), dtype=dtype, device=device)
    value = torch.randn((B, H_kv, S, D), dtype=dtype, device=device)

    def_name = f"gqa_ragged_prefill_causal_h{H_q}_kv{H_kv}_d{D}"

    # Total tokens for ragged format
    total_tokens = B * S

    definition = Definition(
        name=def_name,
        op_type="gqa_ragged",
        axes={
            "num_qo_heads": AxisConst(value=H_q),
            "num_kv_heads": AxisConst(value=H_kv),
            "head_dim": AxisConst(value=D),
            "total_q": AxisVar(),
            "total_kv": AxisVar(),
            "len_indptr": AxisVar(),
        },
        inputs={
            "q": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16"),
            "k": TensorSpec(shape=["total_kv", "num_kv_heads", "head_dim"], dtype="bfloat16"),
            "v": TensorSpec(shape=["total_kv", "num_kv_heads", "head_dim"], dtype="bfloat16"),
            "qo_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16")
        },
        reference=(
            "def run(q, k, v, qo_indptr, kv_indptr, sm_scale):\n"
            "    return q\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "def run(q, k, v, qo_indptr, kv_indptr, sm_scale):\n"
            "    return '__SUB__attention__'\n"
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
        description="Test solution for attention adapter",
    )

    workload = Workload(
        axes={"total_q": total_tokens, "total_kv": total_tokens, "len_indptr": B + 1},
        inputs={"q": RandomInput(), "k": RandomInput(), "v": RandomInput()},
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

    # Create a mock module with required attributes
    class MockModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_causal = True
            self.num_key_value_groups = H_q // H_kv

    mock_module = MockModule()

    with runtime:
        # Call the function; adapter should patch it
        out = sdpa_attention_forward(
            mock_module,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0,
            scaling=1.0 / (D ** 0.5),
            is_causal=True,
        )
        assert out == "__SUB__attention__"


if __name__ == "__main__":
    pytest.main(sys.argv)
