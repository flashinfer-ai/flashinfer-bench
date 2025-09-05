from flashinfer_bench.apply import enable_apply
from flashinfer_bench.apply.config import ApplyConfig
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    ScalarInput,
    Trace,
    Workload,
)
from flashinfer_bench.data.traceset import TraceSet

ts = TraceSet.from_path("../flashinfer-trace")  # path to your flashinfer-trace

# Add a pseudo solution with an unrealistically large speedup so it gets selected
def_name = "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
if def_name in ts.definitions:
    pseudo_code = (
        "import torch\n\n"
        "def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):\n"
        "    total_q, num_qo_heads, head_dim = q.shape\n"
        "    out = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)\n"
        "    lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=q.device)\n"
        "    print('Hello from FlashInfer Bench!')\n"
        "    return out, lse\n"
    )

    pseudo_sol = Solution(
        name="pseudo_hello_gqa_kv8",
        definition=def_name,
        author="pseudo",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="main.py::run",
        ),
        sources=[SourceFile(path="main.py", content=pseudo_code)],
        description="Pseudo solution that prints greeting and returns zero tensors.",
    )

    # Axes chosen to match this script's shapes below
    wl_axes = {
        "len_indptr": 8,  # batch_size + 1
        "total_q": 100,
        "num_kv_indices": 128,
        "num_pages": 128,
    }
    pseudo_trace = Trace(
        definition=def_name,
        solution=pseudo_sol.name,
        workload=Workload(
            axes=wl_axes,
            inputs={
                "q": RandomInput(),
                "k_cache": RandomInput(),
                "v_cache": RandomInput(),
                "qo_indptr": RandomInput(),
                "kv_indptr": RandomInput(),
                "kv_indices": RandomInput(),
                "sm_scale": ScalarInput(value=1.0),
            },
            uuid="pseudo-hello-1",
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log_file="pseudo.log",
            environment=Environment(hardware="gpu", libs={}),
            timestamp="now",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(
                latency_ms=1.0, reference_latency_ms=100.0, speedup_factor=100.0
            ),
        ),
    )

    ts.solutions.setdefault(def_name, []).append(pseudo_sol)
    ts.traces.setdefault(def_name, []).append(pseudo_trace)

# Enable apply against the in-memory augmented trace set
enable_apply(ts, ApplyConfig(on_miss_policy="use_def_best"))

# FlashInfer official example
import torch
import flashinfer

num_layers = 32
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
max_num_pages = 128
page_size = 1
# allocate 128MB workspace buffer
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
batch_size = 7
nnz_qo = 100
qo_indptr = torch.tensor([0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0")
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)
# 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.tensor([1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0")
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
kv_cache_at_layer = torch.randn(
    num_layers,
    max_num_pages,
    2,
    page_size,
    num_kv_heads,
    head_dim,
    dtype=torch.float16,
    device="cuda:0",
)
# create auxiliary data structures for batch prefill attention
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
)
outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    # compute batch prefill attention, reuse auxiliary data structures
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)

# Pseudo result
print("output[0]:", outputs[0])
