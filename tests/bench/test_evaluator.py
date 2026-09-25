import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig, ResolvedEvalConfig
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


def _lowbit_def(n: int = 4) -> Definition:
    return Definition(
        name="moe_fp8_block_scale",
        op_type="moe",
        axes={"N": AxisConst(value=n)},
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


class TestDefaultEvaluatorSetupHook:
    """Tests that DefaultEvaluator invokes setup_for_workload() once per workload (ab70330)."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_setup_for_workload_called_in_correctness_only(self, tmp_path: Path):
        """A passing workload triggers setup_for_workload once, in the correctness
        check. The perf path no longer pre-calls it outside the timer: for real
        setup-hook solutions setup runs INSIDE the timed callable (so latency_ms
        keeps full-call semantics), and mocks don't pass the has_setup_hook gate."""
        definition = _simple_def()
        # NOTE: use ResolvedEvalConfig (atol/rtol have float defaults).
        # BenchmarkConfig leaves atol/rtol as None which only get filled in by
        # resolve_eval_config() — passing BenchmarkConfig straight to evaluator
        # causes compute_error_stats to raise TypeError on `tensor > None`.
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        # Single workload (mock writes the same ref tensor across all workloads,
        # so multi-workload tests would fail correctness on the 2nd onward).
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = inp[0].clone()
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
        assert runnable.setup_for_workload.call_count == 1

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_default_latency_times_setup_inside_for_setup_hook_solutions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """For a real setup-hook Runnable the single-metric path must time
        setup + run together — the timed callable invokes setup_for_workload
        on every call, exactly like a plain solution planning inline."""
        from flashinfer_bench.compile import Runnable, RunnableMetadata

        calls = {"setup": 0, "run": 0}

        def _setup(A):
            calls["setup"] += 1
            return {"bias": 0.0}

        def _run(A, *, bias):
            calls["run"] += 1
            return A + bias

        runnable = Runnable(
            callable=_run,
            metadata=RunnableMetadata(
                build_type="python",
                definition_name="simple_op",
                solution_name="setup_hook_sol",
                destination_passing_style=False,
            ),
            setup_callable=_setup,
        )

        timed_iters = 3

        def _fake_time_runnable(fn, args, warmup, iters, device):
            for _ in range(timed_iters):
                fn(*args)
            return 1.0

        monkeypatch.setattr(default_eval_module, "time_runnable", _fake_time_runnable)

        definition = _simple_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]

        performance, evaluation = DefaultEvaluator.eval_performance(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_mean_latency_ms=2.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation is None
        # Every timed invocation ran setup + run as one unit.
        assert calls["setup"] == timed_iters
        assert calls["run"] == timed_iters
        assert performance.latency_ms == pytest.approx(1.0)
        assert performance.speedup_factor == pytest.approx(2.0)

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_setup_for_workload_called_with_dps_args(self, tmp_path: Path):
        """In DPS mode, setup_for_workload receives (*inputs, *outputs) — same signature as run()."""
        definition = _simple_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)]
        ref_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
        runnable = _make_dps_mock(ref_tensor)

        DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[[ref_tensor]],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        # The DPS evaluator calls setup_for_workload(*inp, *output_tensors).
        # Definition has 1 input + 1 output, so each call receives 2 positional args.
        for call in runnable.setup_for_workload.call_args_list:
            args, kwargs = call
            assert len(args) == 2
            assert all(isinstance(a, torch.Tensor) for a in args)
            assert kwargs == {}


class TestDefaultEvaluatorDPS:
    """Tests for DefaultEvaluator with destination-passing style (DPS) runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_evaluate_pass_dps(self, tmp_path: Path):
        definition = _simple_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
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
# Split-timing aggregation (mechanism-aware kernel_gpu_ms; latency semantics)
# =============================================================================


def _split_metrics(e2e=2.0, kernel=0.5, kgpu=0.4, kgpu_status="ok"):
    from flashinfer_bench.bench.timing import SplitTimingMetrics

    return SplitTimingMetrics(
        e2e_ms=e2e,
        kernel_ms=kernel,
        kernel_gpu_ms=kgpu,
        kernel_ms_status="ok",
        kernel_gpu_ms_status=kgpu_status,
    )


class TestSplitTimingAggregation:
    """eval_performance(split_timing=True) — CPU-only, timers mocked out."""

    def _run(self, monkeypatch, metrics_per_trial):
        it = iter(metrics_per_trial)
        monkeypatch.setattr(
            default_eval_module, "time_runnable_split_timing", lambda *a, **k: next(it)
        )
        definition = _simple_def()
        cfg = ResolvedEvalConfig(
            num_trials=len(metrics_per_trial), warmup_runs=0, iterations=1, split_timing=True
        )
        runnable = _make_vr_mock(torch.tensor([0.0]))
        inputs = [[torch.tensor([1.0, 2.0, 3.0, 4.0])] for _ in metrics_per_trial]
        performance, evaluation = DefaultEvaluator.eval_performance(
            definition=definition,
            sol_runnable=runnable,
            inputs=inputs,
            ref_mean_latency_ms=3.0,
            cfg=cfg,
            log_path="/tmp/log",
            device="cuda:0",
        )
        assert evaluation is None
        return performance

    def test_latency_keeps_single_metric_semantics_and_e2e_is_separate(self, monkeypatch):
        """latency_ms comes from time_runnable (mocked to 1.0 by the autouse
        fixture) — NOT from e2e — and speedup uses it; e2e_ms is its own field."""
        perf = self._run(monkeypatch, [_split_metrics(e2e=2.0), _split_metrics(e2e=4.0)])
        assert perf.latency_ms == pytest.approx(1.0)
        assert perf.speedup_factor == pytest.approx(3.0)
        assert perf.e2e_ms == pytest.approx(3.0)  # mean of 2.0 and 4.0
        assert perf.kernel_ms == pytest.approx(0.5)

    def test_kernel_gpu_all_ok_averages_all(self, monkeypatch):
        perf = self._run(
            monkeypatch,
            [
                _split_metrics(kgpu=0.4, kgpu_status="ok"),
                _split_metrics(kgpu=0.6, kgpu_status="ok"),
            ],
        )
        assert perf.kernel_gpu_ms == pytest.approx(0.5)
        assert perf.kernel_gpu_ms_status == "ok"

    def test_kernel_gpu_partial_uses_cupti_trials_only(self, monkeypatch):
        """A None (unavailable) trial must not drag the mean toward zero."""
        perf = self._run(
            monkeypatch,
            [
                _split_metrics(kgpu=0.4, kgpu_status="ok"),
                _split_metrics(kgpu=None, kgpu_status="no_cupti:RuntimeError"),
            ],
        )
        assert perf.kernel_gpu_ms == pytest.approx(0.4)
        assert perf.kernel_gpu_ms_status == "ok_partial:1/2"

    def test_kernel_gpu_never_mixes_cupti_with_fallback(self, monkeypatch):
        """CUDA-event fallback numbers are a different mechanism; when real
        CUPTI trials exist, only those are aggregated."""
        perf = self._run(
            monkeypatch,
            [
                _split_metrics(kgpu=0.4, kgpu_status="ok"),
                _split_metrics(kgpu=9.9, kgpu_status="cupti_fallback:cuda_events"),
            ],
        )
        assert perf.kernel_gpu_ms == pytest.approx(0.4)
        assert perf.kernel_gpu_ms_status == "ok_partial:1/2"

    def test_kernel_gpu_all_fallback_reports_fallback(self, monkeypatch):
        perf = self._run(
            monkeypatch,
            [
                _split_metrics(kgpu=0.5, kgpu_status="cupti_fallback:cuda_events"),
                _split_metrics(kgpu=0.7, kgpu_status="cupti_fallback:cuda_events"),
            ],
        )
        assert perf.kernel_gpu_ms == pytest.approx(0.6)
        assert perf.kernel_gpu_ms_status == "cupti_fallback:cuda_events"

    def test_kernel_gpu_all_unavailable_is_none(self, monkeypatch):
        perf = self._run(
            monkeypatch,
            [
                _split_metrics(kgpu=None, kgpu_status="no_cupti:ModuleNotFoundError"),
                _split_metrics(kgpu=None, kgpu_status="cupti_no_samples"),
            ],
        )
        assert perf.kernel_gpu_ms is None
        assert perf.kernel_gpu_ms_status == "no_cupti:ModuleNotFoundError"


# =============================================================================
# Setup-hook payload-independence enforcement (anti-gaming)
# =============================================================================


def _matrix_def() -> Definition:
    return Definition(
        name="matrix_identity",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N", "N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A.clone()\n",
    )


def _make_setup_hook_runnable(run_fn, setup_fn):
    from flashinfer_bench.compile import Runnable, RunnableMetadata

    return Runnable(
        callable=run_fn,
        metadata=RunnableMetadata(
            build_type="python",
            definition_name="matrix_identity",
            solution_name="setup_hook_sol",
            destination_passing_style=False,
        ),
        setup_callable=setup_fn,
    )


class TestSetupPayloadIndependence:
    """setup() must not smuggle payload-derived results past the timers.

    A solution that computes its answer inside setup() and replays it from
    run() passes plain correctness (same inputs) but must fail the
    payload-independence re-check: the evaluator re-randomizes the floating
    payload tensors, reuses the stale cached state, and compares run() against
    a fresh reference."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_cached_output_gaming_is_caught(self, tmp_path: Path):
        def _setup(A):
            return {"cached": A.clone()}  # full result precomputed in setup()

        def _run(A, *, cached):
            return cached

        runnable = _make_setup_hook_runnable(_run, _setup)
        definition = _matrix_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        inp = [torch.randn(4, 4, device=device)]
        ref = [inp[0].clone()]

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_metadata_only_setup_state_passes(self, tmp_path: Path):
        def _setup(A):
            return {"n_rows": A.shape[0]}  # shape-derived state is legal

        def _run(A, *, n_rows):
            assert n_rows == A.shape[0]
            return A.clone()

        runnable = _make_setup_hook_runnable(_run, _setup)
        definition = _matrix_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
        device = "cuda:0"
        inp = [torch.randn(4, 4, device=device)]
        original_payload = inp[0].clone()
        ref = [inp[0].clone()]

        evaluation = DefaultEvaluator.evaluate(
            definition=definition,
            sol_runnable=runnable,
            inputs=[inp],
            ref_outputs=[ref],
            ref_mean_latency_ms=1.0,
            cfg=cfg,
            log_path=str(tmp_path / "log"),
            device=device,
        )

        assert evaluation.status == EvaluationStatus.PASSED
        # The payload-independence check must restore the original payload so
        # timing runs on the true workload values (value-dependent kernels,
        # safetensors-captured data).
        assert torch.equal(inp[0], original_payload)


# =============================================================================
# SamplingEvaluator Tests
# =============================================================================


class TestSamplingEvaluatorDPS:
    """Tests for SamplingEvaluator with DPS style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_detects_out_of_vocab_dps(self, tmp_path: Path):
        definition = _sampling_def()
        cfg = ResolvedEvalConfig(
            num_trials=1, warmup_runs=0, iterations=1, extra={"sampling_validation_trials": 1}
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
        cfg = ResolvedEvalConfig(
            num_trials=1, warmup_runs=0, iterations=1, extra={"sampling_validation_trials": 1}
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
        cfg = ResolvedEvalConfig(
            num_trials=1, warmup_runs=0, iterations=1, extra={"sampling_validation_trials": 1}
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
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

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_uses_default_required_matched_ratio_dps(self, tmp_path: Path):
        definition = _lowbit_def(n=20)
        cfg = ResolvedEvalConfig(
            num_trials=1,
            warmup_runs=0,
            iterations=1,
            atol=1e-6,
            rtol=1e-6,
            required_matched_ratio=None,
        )
        device = "cuda:0"
        dev = torch.device(device)
        inp = [torch.arange(20, dtype=torch.float32, device=dev)]
        ref_tensor = torch.arange(20, dtype=torch.float32, device=dev)
        wrong_result = ref_tensor.clone()
        wrong_result[-1] += 1.0
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

        assert evaluation.status == EvaluationStatus.PASSED
        assert evaluation.correctness is not None
        assert evaluation.correctness.extra is not None
        assert evaluation.correctness.extra["matched_ratio"] == pytest.approx(19.0 / 20.0)


class TestLowBitEvaluatorVR:
    """Tests for LowBitEvaluator with VR style runnables."""

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
    def test_lowbit_matched_ratio_included_vr(self, tmp_path: Path):
        definition = _lowbit_def()
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1)
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
        cfg = ResolvedEvalConfig(num_trials=1, warmup_runs=0, iterations=1, atol=1e-6, rtol=1e-6)
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
