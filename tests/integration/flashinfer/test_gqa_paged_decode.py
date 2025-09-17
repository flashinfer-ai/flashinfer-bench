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


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_gqa_paged_decode_adapter_substitution(tmp_path, monkeypatch):
    import flashinfer  # type: ignore

    device = torch.device("cuda")

    # Shapes following canonical definition constants (GQA decode h32 kv4 d128 ps1)
    B = 2
    H_q = 32
    H_kv = 4
    D = 128
    PS = 1
    dtype = torch.bfloat16

    # Page table for page_size=1
    # 0..2..5 => total pages = 5
    indptr = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    indices = torch.arange(indptr[-1].item(), dtype=torch.int32, device=device)
    last_page_len = torch.ones(B, dtype=torch.int32, device=device)

    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    k_cache = torch.randn(indptr[-1], PS, H_kv, D, dtype=dtype, device=device)
    v_cache = torch.randn(indptr[-1], PS, H_kv, D, dtype=dtype, device=device)

    # Minimal Definition matching canonical JSON
    def_name = "gqa_paged_decode_h32_kv4_d128_ps1"
    definition = Definition(
        name=def_name,
        type="gqa",
        axes={
            "batch_size": AxisVar(),
            "num_qo_heads": AxisConst(value=H_q),
            "num_kv_heads": AxisConst(value=H_kv),
            "head_dim": AxisConst(value=D),
            "num_pages": AxisVar(),
            "page_size": AxisConst(value=PS),
            "len_indptr": AxisVar(),
            "num_kv_indices": AxisVar(),
        },
        inputs={
            "q": TensorSpec(shape=["batch_size", "num_qo_heads", "head_dim"], dtype="bfloat16"),
            "k_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "v_cache": TensorSpec(
                shape=["num_pages", "page_size", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "kv_indptr": TensorSpec(shape=["len_indptr"], dtype="int32"),
            "kv_indices": TensorSpec(shape=["num_kv_indices"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch_size", "num_qo_heads", "head_dim"], dtype="bfloat16")
        },
        reference=(
            "def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):\n    return q\n"
        ),
    )

    sol_src = SourceFile(
        path="main.py",
        content=(
            "import torch\n"
            "import flashinfer\n"
            "def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):\n"
            "    return '__SUB__gqa_decode__'\n"
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

    # A single successful trace to select this solution
    wl = Workload(
        axes={
            "batch_size": B,
            "num_pages": int(indptr[-1].item()),
            "len_indptr": B + 1,
            "num_kv_indices": int(indptr[-1].item()),
        },
        inputs={
            "q": RandomInput(),
            "k_cache": RandomInput(),
            "v_cache": RandomInput(),
        },
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
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )

    ts = TraceSet(
        root=tmp_path,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        traces={def_name: [trace]},
    )

    # Enable apply with our in-memory traceset
    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))
    rt = ApplyRuntime(ts, ApplyConfig())
    set_runtime(rt)

    # New wrapper instance to exercise the patched adapter path
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros_like(ws), kv_layout="NHD"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        H_q,
        H_kv,
        D,
        PS,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_apply = wrapper.run(q, (k_cache, v_cache))
    assert out_apply == "__SUB__gqa_decode__"
