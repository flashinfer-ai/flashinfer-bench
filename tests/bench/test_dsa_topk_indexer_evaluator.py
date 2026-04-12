"""Tests for DsaTopkIndexerEvaluator."""

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import (
    DefaultEvaluator,
    DsaSparseAttentionEvaluator,
    DsaTopkIndexerEvaluator,
    resolve_evaluator,
)
from flashinfer_bench.bench.utils import gen_inputs
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Definition,
    EvaluationStatus,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Workload,
)

_NUM_HEADS = 8
_HEAD_DIM = 16
_PAGE_SIZE = 8
_TOPK = 16
_NUM_PAGES = 9
_HEAD_DIM_WITH_SCALE = _HEAD_DIM + 4

_REFERENCE_CODE = f"""\
import torch

def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size = q_index_fp8.shape[0]
    num_heads = {_NUM_HEADS}
    head_dim = {_HEAD_DIM}
    page_size = {_PAGE_SIZE}
    topk = {_TOPK}
    device = q_index_fp8.device

    # dequant
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages = k_uint8.shape[0]
    hds = {_HEAD_DIM_WITH_SCALE}
    flat = k_uint8.view(num_pages, page_size * hds)
    fp8 = flat[:, :page_size * head_dim].contiguous().view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn).float()
    sc = flat[:, page_size * head_dim:].contiguous().view(num_pages, page_size, 4).view(torch.float32)
    K_all = fp8 * sc

    q = q_index_fp8.float()
    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        sl = int(seq_lens[b].item())
        if sl == 0:
            continue
        npb = (sl + page_size - 1) // page_size
        pids = block_table[b, :npb].long()
        K = K_all[pids].reshape(-1, head_dim)[:sl]
        scores = torch.relu(q[b] @ K.T)
        final = (scores * weights[b, :, None]).sum(dim=0)
        actual_k = min(topk, sl)
        _, idx = torch.topk(final, actual_k)
        page_of = idx // page_size
        off = idx % page_size
        topk_indices[b, :actual_k] = (pids[page_of] * page_size + off).int()
    return (topk_indices,)
"""


def _make_definition() -> Definition:
    return Definition(
        name="dsa_topk_indexer_fp8_test",
        op_type="dsa_paged",
        axes={
            "batch_size": AxisVar(),
            "num_index_heads": AxisConst(value=_NUM_HEADS),
            "index_head_dim": AxisConst(value=_HEAD_DIM),
            "page_size": AxisConst(value=_PAGE_SIZE),
            "topk": AxisConst(value=_TOPK),
            "max_num_pages": AxisVar(),
            "num_pages": AxisVar(),
            "kv_cache_num_heads": AxisConst(value=1),
            "head_dim_with_scale": AxisConst(value=_HEAD_DIM_WITH_SCALE),
        },
        inputs={
            "q_index_fp8": TensorSpec(
                shape=["batch_size", "num_index_heads", "index_head_dim"], dtype="float8_e4m3fn"
            ),
            "k_index_cache_fp8": TensorSpec(
                shape=["num_pages", "page_size", "kv_cache_num_heads", "head_dim_with_scale"],
                dtype="int8",
            ),
            "weights": TensorSpec(shape=["batch_size", "num_index_heads"], dtype="float32"),
            "seq_lens": TensorSpec(shape=["batch_size"], dtype="int32"),
            "block_table": TensorSpec(shape=["batch_size", "max_num_pages"], dtype="int32"),
        },
        outputs={"topk_indices": TensorSpec(shape=["batch_size", "topk"], dtype="int32")},
        reference=_REFERENCE_CODE,
    )


def _make_workload() -> Workload:
    return Workload(
        axes={"batch_size": 4, "num_pages": _NUM_PAGES, "max_num_pages": _NUM_PAGES},
        inputs={
            "q_index_fp8": RandomInput(),
            "k_index_cache_fp8": RandomInput(),
            "weights": RandomInput(),
            "seq_lens": RandomInput(),
            "block_table": RandomInput(),
        },
        uuid="test-dsa-topk-indexer",
    )


def _make_solution(name: str, body: str) -> Solution:
    return Solution(
        name=name,
        definition="dsa_topk_indexer_fp8_test",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["NVIDIA_B200"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=body)],
    )


def _make_valid_inputs(definition: Definition, device: str):
    """Build deterministic valid inputs for the small test definition."""
    torch.manual_seed(42)
    batch_size = 4
    q = torch.randn(batch_size, _NUM_HEADS, _HEAD_DIM, device=device).to(torch.float8_e4m3fn)

    k_bf16 = torch.randn(_NUM_PAGES, _PAGE_SIZE, 1, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    amax = k_bf16.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    scale = amax / 448.0
    fp8_data = (k_bf16 * (1.0 / scale)).to(torch.float8_e4m3fn)
    packed = torch.empty(_NUM_PAGES, _PAGE_SIZE * (_HEAD_DIM + 4), device=device, dtype=torch.uint8)
    packed[:, : _PAGE_SIZE * _HEAD_DIM] = fp8_data.view(_NUM_PAGES, _PAGE_SIZE * _HEAD_DIM).view(
        dtype=torch.uint8
    )
    packed[:, _PAGE_SIZE * _HEAD_DIM :] = scale.view(_NUM_PAGES, _PAGE_SIZE).view(dtype=torch.uint8)
    k_fp8 = packed.view(_NUM_PAGES, _PAGE_SIZE, 1, _HEAD_DIM_WITH_SCALE).view(torch.int8)

    weights = torch.randn(batch_size, _NUM_HEADS, device=device, dtype=torch.float32).abs()
    sequence_lengths = torch.tensor(
        [
            _NUM_PAGES * _PAGE_SIZE,
            _NUM_PAGES * _PAGE_SIZE - 3,
            _NUM_PAGES * _PAGE_SIZE - 7,
            _NUM_PAGES * _PAGE_SIZE - 11,
        ],
        dtype=torch.int32,
        device=device,
    )
    block_table = (
        torch.arange(_NUM_PAGES, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    return [q, k_fp8, weights, sequence_lengths, block_table]


def _build_ref_outputs(definition: Definition, inputs, device: str):
    from flashinfer_bench.bench.evaluators.utils import normalize_result

    registry = BuilderRegistry.get_instance()
    ref_runnable = registry.build_reference(definition)
    with torch.no_grad():
        result = ref_runnable(*inputs)
    torch.cuda.synchronize(device)
    return normalize_result(definition, result, device)


# ---- resolve tests ----


def _make_dsa_sparse_attention_def() -> Definition:
    return Definition(
        name="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
        op_type="dsa_paged",
        axes={"num_tokens": AxisVar(), "num_pages": AxisVar()},
        inputs={"x": TensorSpec(shape=["num_tokens"], dtype="float32")},
        outputs={"y": TensorSpec(shape=["num_tokens"], dtype="float32")},
        reference="def run(x): return x",
    )


@pytest.mark.parametrize(
    "definition,expected",
    [
        (_make_definition(), DsaTopkIndexerEvaluator),
        (_make_dsa_sparse_attention_def(), DsaSparseAttentionEvaluator),
    ],
)
def test_resolve(definition, expected):
    assert resolve_evaluator(definition) is expected


# ---- correctness tests ----


@pytest.mark.requires_torch_cuda
def test_correct(tmp_path, tmp_cache_dir):
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    solution = _make_solution("correct", _REFERENCE_CODE)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is None
    assert correctness is not None
    assert torch.isfinite(torch.tensor(correctness.max_absolute_error))
    assert torch.isfinite(torch.tensor(correctness.max_relative_error))
    assert correctness.max_absolute_error < 1e-3


@pytest.mark.requires_torch_cuda
def test_shuffled(tmp_path, tmp_cache_dir):
    """Indices in different order should still pass — the core test."""
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    shuffle_code = _REFERENCE_CODE.replace(
        "_, idx = torch.topk(final, actual_k)",
        "idx = torch.argsort(final, descending=True)[:actual_k]\n        idx = idx[torch.randperm(actual_k, device=device)]",
    )
    solution = _make_solution("shuffled", shuffle_code)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is None
    assert correctness is not None


@pytest.mark.requires_torch_cuda
def test_wrong(tmp_path, tmp_cache_dir):
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    wrong_code = (
        "import torch\n\n"
        f"def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):\n"
        f"    batch_size = q_index_fp8.shape[0]\n"
        f"    page_size = {_PAGE_SIZE}\n"
        f"    topk = {_TOPK}\n"
        "    base_page = block_table[:, :1].to(torch.int32)\n"
        "    offsets = torch.arange(topk, dtype=torch.int32, device=q_index_fp8.device).unsqueeze(0)\n"
        "    wrong_indices = base_page * page_size + offsets\n"
        "    return (wrong_indices.expand(batch_size, -1).contiguous(),)\n"
    )
    solution = _make_solution("wrong", wrong_code)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert correctness is not None
    assert torch.isfinite(torch.tensor(correctness.max_absolute_error))
    assert torch.isfinite(torch.tensor(correctness.max_relative_error))


@pytest.mark.requires_torch_cuda
def test_duplicate(tmp_path, tmp_cache_dir):
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    dup_code = (
        _REFERENCE_CODE
        + "\n"
        + (
            "_original_run = run\n"
            "def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):\n"
            "    result = _original_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)\n"
            "    idx = result[0].clone()\n"
            "    idx[:, 0] = idx[:, 1]\n"
            "    return (idx,)\n"
        )
    )
    solution = _make_solution("duplicate", dup_code)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "duplicate" in evaluation.log.lower()


@pytest.mark.requires_torch_cuda
def test_out_of_range(tmp_path, tmp_cache_dir):
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    oor_code = (
        _REFERENCE_CODE
        + "\n"
        + (
            "_original_run = run\n"
            "def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):\n"
            "    result = _original_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)\n"
            "    idx = result[0].clone()\n"
            f"    idx[:, 0] = {_NUM_PAGES * _PAGE_SIZE + 100}\n"
            "    return (idx,)\n"
        )
    )
    solution = _make_solution("out_of_range", oor_code)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "out-of-range" in evaluation.log.lower()


@pytest.mark.requires_torch_cuda
def test_unreachable_index(tmp_path, tmp_cache_dir):
    definition = _make_definition()
    device = "cuda:0"
    inputs = _make_valid_inputs(definition, device)
    ref_outputs = _build_ref_outputs(definition, inputs, device)

    unreachable_code = (
        _REFERENCE_CODE
        + "\n"
        + (
            "_original_run = run\n"
            "def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):\n"
            "    result = _original_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)\n"
            "    idx = result[0].clone()\n"
            f"    idx[1:, 0] = {_NUM_PAGES * _PAGE_SIZE - 1}\n"
            "    return (idx,)\n"
        )
    )
    solution = _make_solution("unreachable_index", unreachable_code)
    registry = BuilderRegistry.get_instance()
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaTopkIndexerEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
    assert "out-of-range" in evaluation.log.lower()
