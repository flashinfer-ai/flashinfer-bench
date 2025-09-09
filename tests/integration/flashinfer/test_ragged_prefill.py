import math

import flashinfer
import pytest
import torch

from flashinfer_bench.apply.config import ApplyConfig
from flashinfer_bench.apply.runtime import ApplyRuntime, set_runtime
from flashinfer_bench.data.definition import AxisConst, AxisVar, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Trace,
    Workload,
)
from flashinfer_bench.data.traceset import TraceSet


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_ragged_prefill_adapter_substitution(tmp_path, monkeypatch):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Ragged shapes
    B = 2
    qo_indptr = torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)
    H_q = 32
    H_kv = 4
    D = 128

    q = torch.randn(qo_indptr[-1], H_q, D, dtype=dtype, device=device)
    k = torch.randn(kv_indptr[-1], H_kv, D, dtype=dtype, device=device)
    v = torch.randn(kv_indptr[-1], H_kv, D, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(float(D))

    # Minimal Definition matching canonical JSON (GQA ragged prefill h32 kv4 d128)
    def_name = "gqa_ragged_prefill_causal_h32_kv4_d128"
    definition = Definition(
        name=def_name,
        type="gqa",
        axes={
            "num_qo_heads": AxisConst(value=H_q),
            "num_kv_heads": AxisConst(value=H_kv),
            "head_dim": AxisConst(value=D),
            "len_indptr": AxisVar(),
            "total_q": AxisVar(),
            "total_kv": AxisVar(),
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
        reference=("def run(q, k, v, qo_indptr, kv_indptr, sm_scale):\n    return q\n"),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "import flashinfer\n"
            "def run(q, k, v, qo_indptr, kv_indptr, sm_scale):\n"
            "    return '__SUB__gqa_ragged_prefill__'\n"
        ),
    )

    solution = Solution(
        name=f"{def_name}__python_direct_call",
        definition=def_name,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="main.py::run",
        ),
        sources=[sol_src],
        description="Tests",
    )

    wl = Workload(
        axes={
            "len_indptr": B + 1,
            "total_q": int(qo_indptr[-1].item()),
            "total_kv": int(kv_indptr[-1].item()),
        },
        inputs={"q": RandomInput(), "k": RandomInput(), "v": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=wl,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log_file="/dev/null",
            environment=Environment(hardware="gpu", libs={}),
            timestamp="now",
            correctness=Correctness(0.0, 0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )

    ts = TraceSet(
        root=tmp_path,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        traces={def_name: [trace]},
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))
    rt = ApplyRuntime(ts, ApplyConfig())
    set_runtime(rt)

    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        H_q,
        H_kv,
        D,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_apply = wrapper.run(q, k, v)
    assert out_apply == "__SUB__gqa_ragged_prefill__"
