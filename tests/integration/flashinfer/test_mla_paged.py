import math

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-capable PyTorch")
def test_mla_paged_decode_and_prefill(tmp_path, monkeypatch):
    import flashinfer  # type: ignore

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Config constrained by adapter checks
    H = 16
    D_ckv = 512
    D_kpe = 64
    PS = 1

    # Build tiny problem (page_size = 1, two requests)
    B = 2
    kv_len_arr = torch.tensor([3, 2], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    kv_indices = torch.arange(5, dtype=torch.int32, device=device)

    # Decode case: qo_indptr increments by 1
    qo_indptr_decode = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    q_nope_decode = torch.randn(B, H, D_ckv, dtype=dtype, device=device)
    q_pe_decode = torch.zeros(B, H, D_kpe, dtype=dtype, device=device)
    ckv = torch.randn(kv_indptr[-1], PS, D_ckv, dtype=dtype, device=device)
    kpe = torch.zeros(kv_indptr[-1], PS, D_kpe, dtype=dtype, device=device)
    sm_scale = 1.0 / math.sqrt(float(D_ckv + D_kpe))

    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    mla = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws)
    mla.plan(
        qo_indptr_decode,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        H,
        D_ckv,
        D_kpe,
        PS,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope_decode.dtype,
        kv_data_type=ckv.dtype,
    )
    out_decode_direct = mla.run(q_nope_decode, q_pe_decode, ckv, kpe)

    # Minimal MLA decode Definition matching canonical JSON
    def_name_decode = "mla_paged_decode_h16_ckv512_kpe64_ps1"
    def_decode = Definition(
        name=def_name_decode,
        type="mla",
        axes={
            "batch_size": AxisVar(),
            "num_qo_heads": AxisConst(value=H),
            "head_dim_ckv": AxisConst(value=D_ckv),
            "head_dim_kpe": AxisConst(value=D_kpe),
            "page_size": AxisConst(value=PS),
            "num_pages": AxisVar(),
            "len_indptr": AxisVar(),
            "num_kv_indices": AxisVar(),
        },
        inputs={
            "q_nope": TensorSpec(
                shape=["batch_size", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            ),
            "q_pe": TensorSpec(
                shape=["batch_size", "num_qo_heads", "head_dim_kpe"], dtype="bfloat16"
            ),
            "ckv_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_ckv"], dtype="bfloat16"
            ),
            "kpe_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_kpe"], dtype="bfloat16"
            ),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["num_kv_indices"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(
                shape=["batch_size", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            )
        },
        reference=(
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):\n    return q_nope\n"
        ),
    )
    sol_decode = Solution(
        name=f"{def_name_decode}__python_direct_call",
        definition=def_name_decode,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="main.py::run",
        ),
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "import torch\n"
                    "import flashinfer\n"
                    "def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):\n"
                    "    return '__SUB__mla_decode__'\n"
                ),
            )
        ],
        description="Tests",
    )

    # Prefill (incremental) case: qo_indptr arbitrary
    qo_indptr_prefill = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    q_nope_prefill = torch.randn(qo_indptr_prefill[-1], H, D_ckv, dtype=dtype, device=device)
    q_pe_prefill = torch.zeros(qo_indptr_prefill[-1], H, D_kpe, dtype=dtype, device=device)

    mla2 = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.zeros_like(ws))
    mla2.plan(
        qo_indptr_prefill,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        H,
        D_ckv,
        D_kpe,
        PS,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope_prefill.dtype,
        kv_data_type=ckv.dtype,
    )
    out_prefill_direct = mla2.run(q_nope_prefill, q_pe_prefill, ckv, kpe)

    def_name_prefill = "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1"
    def_prefill = Definition(
        name=def_name_prefill,
        type="mla",
        axes={
            "num_qo_heads": AxisConst(value=H),
            "head_dim_ckv": AxisConst(value=D_ckv),
            "head_dim_kpe": AxisConst(value=D_kpe),
            "page_size": AxisConst(value=PS),
            "len_indptr": AxisVar(),
            "total_q": AxisVar(),
            "num_kv_indices": AxisVar(),
            "num_pages": AxisVar(),
        },
        inputs={
            "q_nope": TensorSpec(
                shape=["total_q", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            ),
            "q_pe": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim_kpe"], dtype="bfloat16"),
            "ckv_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_ckv"], dtype="bfloat16"
            ),
            "kpe_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_kpe"], dtype="bfloat16"
            ),
            "qo_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["num_kv_indices"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(
                shape=["total_q", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            )
        },
        reference=(
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n    return q_nope\n"
        ),
    )
    sol_prefill = Solution(
        name=f"{def_name_prefill}__python_direct_call",
        definition=def_name_prefill,
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="main.py::run",
        ),
        sources=[
            SourceFile(
                path="main.py",
                content=(
                    "import torch\n"
                    "import flashinfer\n"
                    "def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n"
                    "    return '__SUB__mla_prefill__'\n"
                ),
            )
        ],
        description="Tests",
    )

    # Enable apply runtime with both definitions registered
    # Build a single TraceSet carrying both
    wl_dec = Workload(
        axes={
            "batch_size": B,
            "num_pages": int(kv_indptr[-1].item()),
            "len_indptr": B + 1,
            "num_kv_indices": int(kv_indptr[-1].item()),
        },
        inputs={},
        uuid="wd",
    )
    wl_pre = Workload(
        axes={
            "len_indptr": B + 1,
            "total_q": int(qo_indptr_prefill[-1].item()),
            "num_kv_indices": int(kv_indptr[-1].item()),
            "num_pages": int(kv_indptr[-1].item()),
        },
        inputs={},
        uuid="wp",
    )
    ts = TraceSet(
        root=tmp_path,
        definitions={def_name_decode: def_decode, def_name_prefill: def_prefill},
        solutions={def_name_decode: [sol_decode], def_name_prefill: [sol_prefill]},
        traces={
            def_name_decode: [
                Trace(
                    definition=def_name_decode,
                    workload=wl_dec,
                    solution=sol_decode.name,
                    evaluation=Evaluation(
                        status=EvaluationStatus.PASSED,
                        log_file="/dev/null",
                        environment=Environment(hardware="gpu", libs={}),
                        timestamp="now",
                        correctness=Correctness(0.0, 0.0),
                        performance=Performance(
                            latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0
                        ),
                    ),
                )
            ],
            def_name_prefill: [
                Trace(
                    definition=def_name_prefill,
                    workload=wl_pre,
                    solution=sol_prefill.name,
                    evaluation=Evaluation(
                        status=EvaluationStatus.PASSED,
                        log_file="/dev/null",
                        environment=Environment(hardware="gpu", libs={}),
                        timestamp="now",
                        correctness=Correctness(0.0, 0.0),
                        performance=Performance(
                            latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0
                        ),
                    ),
                )
            ],
        },
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))
    rt = ApplyRuntime(ts, ApplyConfig())
    set_runtime(rt)

    # Decode through adapter
    mla_d = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.zeros_like(ws))
    mla_d.plan(
        qo_indptr_decode,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        H,
        D_ckv,
        D_kpe,
        PS,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope_decode.dtype,
        kv_data_type=ckv.dtype,
    )
    out_decode_apply = mla_d.run(q_nope_decode, q_pe_decode, ckv, kpe)
    assert out_decode_apply == "__SUB__mla_decode__"

    # Prefill through adapter
    mla_p = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.zeros_like(ws))
    mla_p.plan(
        qo_indptr_prefill,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        H,
        D_ckv,
        D_kpe,
        PS,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope_prefill.dtype,
        kv_data_type=ckv.dtype,
    )
    out_prefill_apply = mla_p.run(q_nope_prefill, q_pe_prefill, ckv, kpe)
    assert out_prefill_apply == "__SUB__mla_prefill__"
