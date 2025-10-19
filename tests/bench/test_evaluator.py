import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from flashinfer_bench.bench.benchmark_config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import default as default_eval_module
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.evaluators import sampling as sampling_eval_module
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
from flashinfer_bench.bench.evaluators.lowbit import LowBitEvaluator
from flashinfer_bench.bench.evaluators.sampling import SamplingEvaluator
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import AxisConst, Definition, EvaluationStatus, TensorSpec


def _simple_def() -> Definition:
    return Definition(
        name="simple_op",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )


def _sampling_def() -> Definition:
    return Definition(
        name="top_k_sampling",
        op_type="sampling",
        axes={"batch_size": AxisConst(value=2), "vocab_size": AxisConst(value=100)},
        inputs={
            "probs": TensorSpec(shape=["batch_size", "vocab_size"], dtype="float32"),
            "top_k": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"samples": TensorSpec(shape=["batch_size"], dtype="int32")},
        reference="import torch\n\ndef run(probs, top_k):\n    return torch.multinomial(probs, 1).squeeze(-1)\n",
    )


def _lowbit_def() -> Definition:
    return Definition(
        name="moe_fp8_block_scale",
        op_type="moe",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )


@pytest.fixture(autouse=True)
def _patch_time_runnable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(default_eval_module, "time_runnable", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(sampling_eval_module, "time_runnable", lambda *args, **kwargs: 1.0)


class TestDefaultEvaluator:
    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_pass(self, tmp_path: Path):
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        runnable.return_value = ref_out["B"]

        evaluation = DefaultEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.performance is not None

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_shape_error(self, tmp_path: Path):
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock(spec=Runnable)
        runnable.return_value = torch.tensor([1.0, 2.0], device=dev)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}

        evaluation = DefaultEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_SHAPE

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_performance_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        def failing_timer(*args, **kwargs):  # raises on first perf measurement
            raise RuntimeError("perf failure")

        monkeypatch.setattr(default_eval_module, "time_runnable", failing_timer)
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        runnable.return_value = ref_out["B"]

        evaluation = DefaultEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR


class TestSamplingEvaluator:
    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_detects_out_of_vocab(self, tmp_path: Path):
        defn = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock(spec=Runnable)
        probs = torch.softmax(torch.randn(2, 100, device=dev), dim=-1)
        inp = {"probs": probs, "top_k": torch.tensor(10, device=dev, dtype=torch.int32)}
        runnable.return_value = torch.tensor([50, 150], device=dev)  # 150 is invalid
        ref_out = {"frequency_distribution": torch.zeros(100, device=dev)}

        evaluation = SamplingEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_sampling_runtime_error(self, tmp_path: Path):
        defn = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock(spec=Runnable)
        runnable.side_effect = RuntimeError("sampling fail")
        probs = torch.softmax(torch.randn(2, 100, device=dev), dim=-1)
        inp = {"probs": probs, "top_k": torch.tensor(10, device=dev, dtype=torch.int32)}
        ref_out = {"frequency_distribution": torch.zeros(100, device=dev)}

        evaluation = SamplingEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR


class TestLowBitEvaluator:
    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_matched_ratio_included(self, tmp_path: Path):
        defn = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)

        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        runnable.return_value = ref_out["B"]

        evaluation = LowBitEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.correctness.extra is not None
        assert evaluation.correctness.extra["matched_ratio"] == pytest.approx(1.0)

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_matched_ratio_on_failure(self, tmp_path: Path):
        defn = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
        device = "cuda:0"
        dev = torch.device(device)

        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)}
        # Introduce one element outside tolerance to force failure.
        runnable.return_value = torch.tensor([1.0, 2.0, 3.0, 6.0], device=dev)

        evaluation = LowBitEvaluator.evaluate(
            defn=defn,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert evaluation.correctness is not None
        assert evaluation.correctness.extra is not None
        matched_ratio = evaluation.correctness.extra["matched_ratio"]
        assert matched_ratio == pytest.approx(3.0 / 4.0)


def test_resolve_evaluator_selects_sampling():
    evaluator = resolve_evaluator(_sampling_def())
    assert evaluator is SamplingEvaluator


def test_resolve_evaluator_selects_lowbit():
    evaluator = resolve_evaluator(_lowbit_def())
    assert evaluator is LowBitEvaluator


def test_resolve_evaluator_selects_default():
    evaluator = resolve_evaluator(_simple_def())
    assert evaluator is DefaultEvaluator


if __name__ == "__main__":
    pytest.main(sys.argv)
