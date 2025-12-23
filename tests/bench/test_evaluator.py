import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import default as default_eval_module
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.evaluators import sampling as sampling_eval_module
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
from flashinfer_bench.bench.evaluators.lowbit import LowBitEvaluator
from flashinfer_bench.bench.evaluators.sampling import SamplingEvaluator
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


def _make_dps_mock(result_tensor):
    """Create a mock runnable that writes result_tensor to output in DPS style."""
    mock = MagicMock()
    mock.metadata.destination_passing_style = True

    def dps_side_effect(*args):
        output = args[-1]
        output.copy_(result_tensor)

    mock.side_effect = dps_side_effect
    return mock


def _make_vr_mock(result_tensor):
    """Create a mock runnable that returns result_tensor in value-returning style."""
    mock = MagicMock()
    mock.metadata.destination_passing_style = False
    mock.return_value = result_tensor
    return mock


# =============================================================================
# DefaultEvaluator Tests
# =============================================================================


class TestDefaultEvaluatorDPS:
    """Tests for DefaultEvaluator with destination-passing style (DPS) runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_pass_dps(self, tmp_path: Path):
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_dps_mock(ref_tensor)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.performance is not None

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_shape_error_dps(self, tmp_path: Path):
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        wrong_result = torch.tensor([1.0, 2.0], device=dev)
        runnable = _make_dps_mock(wrong_result)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        # With DPS, output tensors are pre-allocated with correct shape,
        # so shape errors manifest as numerical errors due to partial copy
        assert evaluation.status in (
            EvaluationStatus.INCORRECT_SHAPE,
            EvaluationStatus.INCORRECT_NUMERICAL,
            EvaluationStatus.RUNTIME_ERROR,
        )

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_performance_failure_dps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        def failing_timer(*args, **kwargs):
            raise RuntimeError("perf failure")

        monkeypatch.setattr(default_eval_module, "time_runnable", failing_timer)
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_dps_mock(ref_tensor)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR


class TestDefaultEvaluatorVR:
    """Tests for DefaultEvaluator with value-returning (VR) style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_pass_vr(self, tmp_path: Path):
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_vr_mock(ref_tensor)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.performance is not None

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_shape_error_vr(self, tmp_path: Path):
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        wrong_result = torch.tensor([1.0, 2.0], device=dev)
        runnable = _make_vr_mock(wrong_result)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        # VR style directly returns wrong shape
        assert evaluation.status == EvaluationStatus.INCORRECT_SHAPE

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_numerical_error_vr(self, tmp_path: Path):
        definition = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        wrong_result = torch.tensor([1.0, 2.0, 3.0, 99.0], device=dev)
        runnable = _make_vr_mock(wrong_result)

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL


# =============================================================================
# SamplingEvaluator Tests
# =============================================================================


class TestSamplingEvaluatorDPS:
    """Tests for SamplingEvaluator with DPS style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_detects_out_of_vocab_dps(self, tmp_path: Path):
        definition = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )
        device = "cuda:0"
        dev = torch.device(device)
        probs = torch.softmax(torch.randn(2, 100, device=dev), dim=-1)
        top_k = torch.tensor(10, device=dev, dtype=torch.int32)
        inp = [probs, top_k]
        invalid_samples = torch.tensor([50, 150], device=dev, dtype=torch.int32)
        runnable = _make_dps_mock(invalid_samples)
        expected_probs = torch.zeros(2, 100, device=dev)

        evaluation = SamplingEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[expected_probs]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_sampling_runtime_error_dps(self, tmp_path: Path):
        definition = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )
        device = "cuda:0"
        dev = torch.device(device)
        runnable = MagicMock()
        runnable.metadata.destination_passing_style = True
        runnable.side_effect = RuntimeError("sampling fail")
        probs = torch.softmax(torch.randn(2, 100, device=dev), dim=-1)
        top_k = torch.tensor(10, device=dev, dtype=torch.int32)
        inp = [probs, top_k]
        expected_probs = torch.zeros(2, 100, device=dev)

        evaluation = SamplingEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[expected_probs]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR


class TestSamplingEvaluatorVR:
    """Tests for SamplingEvaluator with VR style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_detects_out_of_vocab_vr(self, tmp_path: Path):
        definition = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )
        device = "cuda:0"
        dev = torch.device(device)
        probs = torch.softmax(torch.randn(2, 100, device=dev), dim=-1)
        top_k = torch.tensor(10, device=dev, dtype=torch.int32)
        inp = [probs, top_k]
        invalid_samples = torch.tensor([50, 150], device=dev, dtype=torch.int32)
        runnable = _make_vr_mock(invalid_samples)
        expected_probs = torch.zeros(2, 100, device=dev)

        evaluation = SamplingEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[expected_probs]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL


# =============================================================================
# LowBitEvaluator Tests
# =============================================================================


class TestLowBitEvaluatorDPS:
    """Tests for LowBitEvaluator with DPS style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_matched_ratio_included_dps(self, tmp_path: Path):
        definition = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_dps_mock(ref_tensor)

        evaluation = LowBitEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
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
    def test_lowbit_matched_ratio_on_failure_dps(self, tmp_path: Path):
        definition = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        wrong_result = torch.tensor([1.0, 2.0, 3.0, 6.0], device=dev)
        runnable = _make_dps_mock(wrong_result)

        evaluation = LowBitEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert evaluation.correctness is not None
        assert evaluation.correctness.extra is not None
        assert evaluation.correctness.extra["matched_ratio"] == pytest.approx(3.0 / 4.0)


class TestLowBitEvaluatorVR:
    """Tests for LowBitEvaluator with VR style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_matched_ratio_included_vr(self, tmp_path: Path):
        definition = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_vr_mock(ref_tensor)

        evaluation = LowBitEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
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
    def test_lowbit_matched_ratio_on_failure_vr(self, tmp_path: Path):
        definition = _lowbit_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        wrong_result = torch.tensor([1.0, 2.0, 3.0, 6.0], device=dev)
        runnable = _make_vr_mock(wrong_result)

        evaluation = LowBitEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert evaluation.correctness is not None
        assert evaluation.correctness.extra is not None
        assert evaluation.correctness.extra["matched_ratio"] == pytest.approx(3.0 / 4.0)


# =============================================================================
# Evaluator Resolution Tests
# =============================================================================


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
