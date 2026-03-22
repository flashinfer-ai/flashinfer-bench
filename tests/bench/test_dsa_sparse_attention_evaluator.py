"""Tests for DsaSparseAttentionEvaluator with real workloads."""

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import (
    DefaultEvaluator,
    DsaSparseAttentionEvaluator,
    resolve_evaluator,
)
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry, Runnable
from flashinfer_bench.compile.runnable import RunnableMetadata
from flashinfer_bench.data import EvaluationStatus, TraceSet
from flashinfer_bench.env import get_fib_dataset_path

TRACE_PATH = str(get_fib_dataset_path())
DEF_NAME = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"


@pytest.fixture(scope="module")
def trace_set():
    return TraceSet.from_path(TRACE_PATH)


@pytest.fixture(scope="module")
def registry():
    return BuilderRegistry.get_instance()


@pytest.fixture(scope="module")
def ref_data(trace_set, registry):
    """Build reference runnable and generate inputs/outputs for one workload."""
    defn = trace_set.definitions[DEF_NAME]
    ref_runnable = registry.build_reference(defn)
    wl = trace_set.workloads[DEF_NAME][0].workload
    loaded_st = load_safetensors(defn, wl, trace_set.root)
    inputs = gen_inputs(defn, wl, device="cuda", safe_tensors=loaded_st)
    with torch.no_grad():
        ref_out = ref_runnable(*inputs)
    ref_outputs = list(ref_out)
    return defn, inputs, ref_outputs


def _wrap_callable(fn):
    """Wrap a plain callable as a value-returning Runnable."""
    meta = RunnableMetadata(
        build_type="python",
        definition_name=DEF_NAME,
        solution_name="test_wrapper",
        destination_passing_style=False,
    )
    return Runnable(callable=fn, metadata=meta)


@pytest.mark.parametrize(
    "def_name,expected",
    [
        ("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1", DsaSparseAttentionEvaluator),
        ("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64", DsaSparseAttentionEvaluator),
        ("dsa_topk_indexer_fp8_h64_d128_topk2048_ps64", DefaultEvaluator),
    ],
)
def test_resolve(trace_set, def_name, expected):
    assert resolve_evaluator(trace_set.definitions[def_name]) is expected


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA not available")
def test_both_outputs(ref_data, tmp_path):
    defn, inputs, ref_outputs = ref_data
    output_tensor, lse_tensor = ref_outputs

    sol = _wrap_callable(lambda *args: (output_tensor, lse_tensor))
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=defn,
        sol_runnable=sol,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device="cuda",
    )
    assert evaluation is None
    assert correctness is not None
    assert correctness.max_absolute_error == 0.0


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA not available")
def test_output_only(ref_data, tmp_path):
    defn, inputs, ref_outputs = ref_data
    output_tensor = ref_outputs[0]

    sol = _wrap_callable(lambda *args: output_tensor)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=defn,
        sol_runnable=sol,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device="cuda",
    )
    assert evaluation is None
    assert correctness is not None
    assert correctness.max_absolute_error == 0.0


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA not available")
def test_wrong_output(ref_data, tmp_path):
    defn, inputs, ref_outputs = ref_data
    wrong = torch.zeros_like(ref_outputs[0])

    sol = _wrap_callable(lambda *args: wrong)
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
    correctness, evaluation = DsaSparseAttentionEvaluator.check_correctness(
        definition=defn,
        sol_runnable=sol,
        inputs=[inputs],
        ref_outputs=[ref_outputs],
        cfg=cfg,
        log_path=str(tmp_path / "log"),
        device="cuda",
    )
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
