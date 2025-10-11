import sys
from unittest.mock import MagicMock

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.evaluator import (
    DefaultValidator,
    SamplingValidator,
    SolutionEvaluator,
)
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import AxisConst, Definition, EvaluationStatus, TensorSpec


def _simple_def():
    """Simple definition for testing."""
    return Definition(
        name="simple_op",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )


def _sampling_def():
    """Sampling definition for testing."""
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


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestDefaultValidator:
    """Tests for DefaultValidator correctness checking."""

    def test_validate_correct_output(self):
        """Test validation passes for correct outputs."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns correct output
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = ref_out["B"]

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is None  # No error
        assert max_abs < 1e-6  # Very small error
        assert not numerical_incorrect

    def test_validate_shape_mismatch(self):
        """Test validation detects shape mismatches."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns wrong shape
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = torch.tensor([1.0, 2.0], device=device)  # Wrong shape

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.INCORRECT_SHAPE

    def test_validate_dtype_mismatch(self):
        """Test validation detects dtype mismatches."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns wrong dtype
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)}

        runnable.return_value = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device)

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.INCORRECT_DTYPE

    def test_validate_numerical_error(self):
        """Test validation detects numerical errors exceeding tolerance."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-5, rtol=1e-5)

        device = "cuda:0"
        # Create mock runnable that returns incorrect numerical result
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        # Large error exceeding tolerance
        runnable.return_value = torch.tensor([1.0, 2.0, 3.0, 10.0], device=device)

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is None  # No early return for numerical errors
        assert numerical_incorrect is True
        assert max_abs > 1.0  # Error is significant

    def test_validate_inf_output(self):
        """Test validation detects infinite values."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns inf
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = torch.tensor([1.0, 2.0, float("inf"), 4.0], device=device)

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert eval_result.correctness is not None
        assert eval_result.correctness.max_absolute_error == float("inf")

    def test_validate_nan_output(self):
        """Test validation detects NaN values."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns NaN
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = torch.tensor([1.0, float("nan"), 3.0, 4.0], device=device)

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device=device,
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert eval_result.correctness is not None

    def test_validate_runtime_error(self):
        """Test validation handles runtime errors gracefully."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        # Create mock runnable that raises exception
        runnable = MagicMock(spec=Runnable)
        runnable.side_effect = RuntimeError("Test error")

        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        eval_result, max_abs, max_rel, numerical_incorrect, matched_ratio = (
            DefaultValidator.validate_correctness(
                runnable_sol=runnable,
                inputs=[inp],
                ref_outputs_bl=[ref_out],
                cfg=cfg,
                device="cpu",
                log_path="/tmp/test.log",
                defn=defn,
            )
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.RUNTIME_ERROR
        assert "RuntimeError" in eval_result.error


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestSamplingValidator:
    """Tests for SamplingValidator correctness checking."""

    def test_validate_out_of_vocab(self):
        """Test validation detects samples outside vocabulary range."""
        defn = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )

        device = "cuda:0"
        # Create mock runnable that returns invalid samples
        runnable = MagicMock(spec=Runnable)
        probs = torch.softmax(torch.randn(2, 100), dim=-1).to(device)
        inp = {"probs": probs, "top_k": 10}

        # Samples outside vocab range [0, 100)
        runnable.return_value = torch.tensor([50, 150], device=device)  # 150 is out of range

        ref_freq = torch.zeros(100, device=device)
        ref_out = {"frequency_distribution": ref_freq}

        eval_result, max_abs, max_rel, numerical_incorrect = SamplingValidator.validate_correctness(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs_bl=[ref_out],
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert "out of vocabulary range" in eval_result.error

    def test_validate_runtime_error_sampling(self):
        """Test sampling validation handles runtime errors."""
        defn = _sampling_def()
        cfg = BenchmarkConfig(
            num_trials=1, warmup_runs=0, iterations=1, sampling_validation_trials=1
        )

        device = "cuda:0"
        # Create mock runnable that raises exception
        runnable = MagicMock(spec=Runnable)
        runnable.side_effect = RuntimeError("Sampling error")

        probs = torch.softmax(torch.randn(2, 100), dim=-1).to(device)
        inp = {"probs": probs, "top_k": 10}
        ref_freq = torch.zeros(100, device=device)
        ref_out = {"frequency_distribution": ref_freq}

        eval_result, max_abs, max_rel, numerical_incorrect = SamplingValidator.validate_correctness(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs_bl=[ref_out],
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert eval_result is not None
        assert eval_result.status == EvaluationStatus.RUNTIME_ERROR
        assert "RuntimeError" in eval_result.error


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestSolutionEvaluator:
    """Tests for SolutionEvaluator.evaluate method."""

    def test_evaluate_empty_ref_outputs(self):
        """Test evaluation with empty reference outputs."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        runnable = MagicMock(spec=Runnable)

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[{"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}],
            ref_outputs=[],  # Empty
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR
        assert evaluation.log and "No reference outputs provided" in evaluation.log

    def test_evaluate_correct_solution(self):
        """Test evaluation of a correct solution."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that returns correct output
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = ref_out["B"]

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.performance is not None
        assert evaluation.performance.reference_latency_ms == 1.0
        assert evaluation.performance.speedup_factor > 0

    def test_evaluate_incorrect_numerical(self):
        """Test evaluation of numerically incorrect solution."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-5, rtol=1e-5)

        device = "cuda:0"
        # Create mock runnable with large error
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = torch.tensor([1.0, 2.0, 3.0, 100.0], device=device)  # Large error

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert evaluation.correctness is not None
        assert evaluation.correctness.max_absolute_error > 10.0

    def test_evaluate_shape_error(self):
        """Test evaluation detects shape errors."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable with wrong shape
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = torch.tensor([1.0, 2.0], device=device)  # Wrong shape

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_SHAPE

    def test_evaluate_performance_measurement_error(self):
        """Test evaluation handles performance measurement errors."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        # Create mock runnable that works for validation but fails for perf
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        # First call for validation succeeds, subsequent calls for perf fail
        runnable.side_effect = [
            ref_out["B"],  # First call for validation
            RuntimeError("Performance measurement failed"),  # Second call for perf
        ]

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.RUNTIME_ERROR
        assert evaluation.log and "Performance measurement failed" in evaluation.log


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestEvaluatorWithGPU:
    """GPU-specific tests for the evaluator."""

    def test_evaluate_on_gpu(self):
        """Test evaluation on GPU device."""
        defn = _simple_def()
        cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

        device = "cuda:0"
        runnable = MagicMock(spec=Runnable)
        inp = {"A": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}
        ref_out = {"B": torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)}

        runnable.return_value = ref_out["B"]

        evaluation = SolutionEvaluator.evaluate(
            runnable_sol=runnable,
            inputs=[inp],
            ref_outputs=[ref_out],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            device=device,
            log_path="/tmp/test.log",
            defn=defn,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.environment is not None


if __name__ == "__main__":
    pytest.main(sys.argv)
