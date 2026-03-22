"""Tests for DsaSparseAttentionEvaluator."""

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import (
    DefaultEvaluator,
    DsaSparseAttentionEvaluator,
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
    ScalarInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Workload,
)


def _make_dsa_sparse_attention_def(name: str) -> Definition:
    return Definition(
        name=name,
        op_type="dsa_paged",
        axes={
            "num_tokens": AxisVar(),
            "num_qo_heads": AxisConst(value=16),
            "head_dim_ckv": AxisConst(value=512),
            "head_dim_kpe": AxisConst(value=64),
            "page_size": AxisConst(value=64),
            "topk": AxisConst(value=2048),
            "num_pages": AxisVar(),
        },
        inputs={
            "q_nope": TensorSpec(
                shape=["num_tokens", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            ),
            "q_pe": TensorSpec(
                shape=["num_tokens", "num_qo_heads", "head_dim_kpe"], dtype="bfloat16"
            ),
            "ckv_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_ckv"], dtype="bfloat16"
            ),
            "kpe_cache": TensorSpec(
                shape=["num_pages", "page_size", "head_dim_kpe"], dtype="bfloat16"
            ),
            "sparse_indices": TensorSpec(shape=["num_tokens", "topk"], dtype="int32"),
            "sm_scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={
            "output": TensorSpec(
                shape=["num_tokens", "num_qo_heads", "head_dim_ckv"], dtype="bfloat16"
            ),
            "lse": TensorSpec(shape=["num_tokens", "num_qo_heads"], dtype="float32"),
        },
        reference=(
            "import torch\n\n"
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):\n"
            "    return q_nope, torch.zeros(q_nope.shape[:2], dtype=torch.float32, device=q_nope.device)\n"
        ),
    )


def _make_dsa_topk_indexer_def() -> Definition:
    return Definition(
        name="dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
        op_type="dsa_paged",
        axes={
            "batch_size": AxisVar(),
            "num_index_heads": AxisConst(value=64),
            "index_head_dim": AxisConst(value=128),
            "page_size": AxisConst(value=64),
            "topk": AxisConst(value=2048),
            "max_num_pages": AxisVar(),
            "num_pages": AxisVar(),
            "kv_cache_num_heads": AxisConst(value=1),
            "head_dim_with_scale": AxisConst(value=132),
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
        reference=(
            "import torch\n\n"
            "def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):\n"
            "    return torch.zeros((q_index_fp8.shape[0], 2048), dtype=torch.int32, device=q_index_fp8.device)\n"
        ),
    )


def _make_workload() -> Workload:
    return Workload(
        axes={"num_tokens": 4, "num_pages": 5},
        inputs={
            "q_nope": RandomInput(),
            "q_pe": RandomInput(),
            "ckv_cache": RandomInput(),
            "kpe_cache": RandomInput(),
            "sparse_indices": RandomInput(),
            "sm_scale": ScalarInput(value=0.1),
        },
        uuid="test-dsa-sparse-attention",
    )


def _make_solution(definition_name: str, solution_name: str, body: str) -> Solution:
    return Solution(
        name=solution_name,
        definition=definition_name,
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["NVIDIA_B200"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=body)],
    )


def _make_trial(defn: Definition, device: str):
    registry = BuilderRegistry.get_instance()
    wl = _make_workload()
    trial_inputs = gen_inputs(defn, wl, device=device)
    ref_runnable = registry.build_reference(defn)
    with torch.no_grad():
        ref_result = ref_runnable(*trial_inputs)
    return registry, [trial_inputs], [list(ref_result)]


@pytest.mark.parametrize(
    "definition,expected",
    [
        (
            _make_dsa_sparse_attention_def("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1"),
            DsaSparseAttentionEvaluator,
        ),
        (
            _make_dsa_sparse_attention_def("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"),
            DsaSparseAttentionEvaluator,
        ),
        (_make_dsa_topk_indexer_def(), DefaultEvaluator),
    ],
)
def test_resolve(definition, expected):
    assert resolve_evaluator(definition) is expected


@pytest.mark.requires_torch_cuda
def test_both_outputs(tmp_path, tmp_cache_dir):
    definition = _make_dsa_sparse_attention_def(
        "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    )
    device = "cuda:0"
    registry, inputs, ref_outputs = _make_trial(definition, device)
    solution = _make_solution(
        definition.name,
        "full_outputs",
        (
            "import torch\n\n"
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):\n"
            "    return q_nope, torch.zeros(q_nope.shape[:2], dtype=torch.float32, device=q_nope.device)\n"
        ),
    )
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=inputs,
        ref_outputs=ref_outputs,
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )

    assert evaluation is None
    assert correctness is not None
    assert correctness.max_absolute_error == 0.0


@pytest.mark.requires_torch_cuda
def test_output_only(tmp_path, tmp_cache_dir):
    definition = _make_dsa_sparse_attention_def(
        "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    )
    device = "cuda:0"
    registry, inputs, ref_outputs = _make_trial(definition, device)
    solution = _make_solution(
        definition.name,
        "output_only",
        (
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):\n"
            "    return q_nope\n"
        ),
    )
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=inputs,
        ref_outputs=ref_outputs,
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )

    assert evaluation is None
    assert correctness is not None
    assert correctness.max_absolute_error == 0.0


@pytest.mark.requires_torch_cuda
def test_wrong_output(tmp_path, tmp_cache_dir):
    definition = _make_dsa_sparse_attention_def(
        "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    )
    device = "cuda:0"
    registry, inputs, ref_outputs = _make_trial(definition, device)
    solution = _make_solution(
        definition.name,
        "wrong_output",
        (
            "import torch\n\n"
            "def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):\n"
            "    return torch.zeros_like(q_nope)\n"
        ),
    )
    sol_runnable = registry.build(definition, solution)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)

    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=definition,
        sol_runnable=sol_runnable,
        inputs=inputs,
        ref_outputs=ref_outputs,
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device=device,
    )

    assert correctness is not None
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
