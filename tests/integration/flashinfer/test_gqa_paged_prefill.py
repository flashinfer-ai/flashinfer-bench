import math
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


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_gqa_paged_prefill_adapter_substitution_ps1(tmp_path, monkeypatch):
    import flashinfer  # type: ignore

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Shapes following canonical definition constants (GQA prefill h32 kv4 d128 ps1)
    B = 2
    H_q = 32
    H_kv = 4
    D = 128
    PS = 1

    # Indices for qo and kv (page_size=1)
    qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 4, 7], dtype=torch.int32, device=device)
    kv_indices = torch.arange(kv_indptr[-1].item(), dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(B, dtype=torch.int32, device=device)

    q = torch.randn(qo_indptr[-1], H_q, D, dtype=dtype, device=device)
    k_cache = torch.randn(kv_indptr[-1], PS, H_kv, D, dtype=dtype, device=device)
    v_cache = torch.randn(kv_indptr[-1], PS, H_kv, D, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(float(D))

    # Minimal Definition matching canonical JSON
    def_name = "gqa_paged_prefill_causal_h32_kv4_d128_ps1"
    definition = Definition(
        name=def_name,
        op_type="gqa",
        axes={
            "num_qo_heads": AxisConst(value=H_q),
            "num_kv_heads": AxisConst(value=H_kv),
            "head_dim": AxisConst(value=D),
            "page_size": AxisConst(value=PS),
            "len_indptr": AxisVar(),
            "total_q": AxisVar(),
            "num_kv_indices": AxisVar(),
            "num_pages": AxisVar(),
        },
        inputs={
            "q": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16"),
            "k_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "v_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "qo_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["num_kv_indices"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16")
        },
        reference=(
            "def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n    return q\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "import flashinfer\n"
            "def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n"
            "    return '__SUB__gqa_paged_prefill__'\n"
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
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Tests",
    )

    workload = Workload(
        axes={
            "len_indptr": B + 1,
            "total_q": int(qo_indptr[-1].item()),
            "num_kv_indices": int(kv_indptr[-1].item()),
            "num_pages": int(kv_indptr[-1].item()),
        },
        inputs={"q": RandomInput(), "k_cache": RandomInput(), "v_cache": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="gpu", libs={}),
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
        ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            H_q,
            H_kv,
            D,
            PS,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        out_apply = wrapper.run(q, (k_cache, v_cache))
        assert out_apply == "__SUB__gqa_paged_prefill__"


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_gqa_paged_prefill_adapter_substitution_ps64(tmp_path, monkeypatch):
    import flashinfer  # type: ignore

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Shapes following canonical definition constants (GQA prefill h32 kv4 d128 ps64)
    B = 2
    H_q = 32
    H_kv = 4
    D = 128
    PS = 64

    # Indices for qo and kv (page_size=64)
    # Seq0: 96 query tokens, 128 kv tokens (2 pages)
    # Seq1: 64 query tokens, 192 kv tokens (3 pages)
    # Total pages = 5
    num_pages = 5
    qo_indptr = torch.tensor([0, 96, 160], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([64, 64], dtype=torch.int32, device=device)

    q = torch.randn(qo_indptr[-1], H_q, D, dtype=dtype, device=device)
    k_cache = torch.randn(num_pages, PS, H_kv, D, dtype=dtype, device=device)
    v_cache = torch.randn(num_pages, PS, H_kv, D, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(float(D))

    # Minimal Definition matching canonical JSON
    def_name = "gqa_paged_prefill_causal_h32_kv4_d128_ps64"
    definition = Definition(
        name=def_name,
        op_type="gqa",
        axes={
            "num_qo_heads": AxisConst(value=H_q),
            "num_kv_heads": AxisConst(value=H_kv),
            "head_dim": AxisConst(value=D),
            "page_size": AxisConst(value=PS),
            "len_indptr": AxisVar(),
            "total_q": AxisVar(),
            "num_kv_indices": AxisVar(),
            "num_pages": AxisVar(),
        },
        inputs={
            "q": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16"),
            "k_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "v_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "qo_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["num_kv_indices"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16")
        },
        reference=(
            "def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n    return q\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "import flashinfer\n"
            "def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n"
            "    return '__SUB__gqa_paged_prefill__'\n"
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
            destination_passing_style=False,
        ),
        sources=[sol_src],
        description="Tests",
    )

    workload = Workload(
        axes={
            "len_indptr": B + 1,
            "total_q": int(qo_indptr[-1].item()),
            "num_kv_indices": num_pages,
            "num_pages": num_pages,
        },
        inputs={"q": RandomInput(), "k_cache": RandomInput(), "v_cache": RandomInput()},
        uuid="w0",
    )
    trace = Trace(
        definition=def_name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="/dev/null",
            environment=Environment(hardware="gpu", libs={}),
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
        ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            H_q,
            H_kv,
            D,
            PS,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        out_apply = wrapper.run(q, (k_cache, v_cache))
        assert out_apply == "__SUB__gqa_paged_prefill__"


if __name__ == "__main__":
    pytest.main(sys.argv)
