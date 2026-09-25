"""Unit tests for the kernel-agnostic split timing engine.

Covers ``SplitTimingMetrics`` schema, the internal ``_measure_*`` helpers, and the
top-level ``time_runnable_split_timing`` API. CPU-only tests use mocks; GPU-required
tests are guarded with ``pytest.mark.skipif(torch.cuda.device_count() == 0)``.
"""

from __future__ import annotations

import contextlib
import warnings
from unittest.mock import patch

import pytest
import torch

from flashinfer_bench.bench.timing import SplitTimingMetrics, time_runnable_split_timing
from flashinfer_bench.bench.timing.split_timing import (
    _PHASE_SEQUENCE,
    _maybe_clone,
    _measure_e2e,
    _measure_kernel_cudaevent,
    _measure_kernel_gpu_cupti,
    _rotated_phase_order,
)
from flashinfer_bench.compile import Runnable, RunnableMetadata

# -----------------------------------------------------------------------------
# Helpers — build a small mock Runnable for tests
# -----------------------------------------------------------------------------


def _make_runnable(run_fn, setup_fn=None) -> Runnable:
    return Runnable(
        callable=run_fn,
        metadata=RunnableMetadata(
            build_type="python",
            definition_name="test_def",
            solution_name="test_sol",
            destination_passing_style=False,
        ),
        setup_callable=setup_fn,
    )


# -----------------------------------------------------------------------------
# SplitTimingMetrics dataclass — schema, frozenness, defaults
# -----------------------------------------------------------------------------


class TestSplitTimingMetrics:
    def test_construct_with_all_fields(self):
        m = SplitTimingMetrics(
            e2e_ms=1.5,
            kernel_ms=0.3,
            kernel_gpu_ms=0.32,
            kernel_ms_status="ok",
            kernel_gpu_ms_status="ok",
        )
        assert m.e2e_ms == 1.5
        assert m.kernel_ms == 0.3
        assert m.kernel_gpu_ms == 0.32
        assert m.kernel_ms_status == "ok"
        assert m.kernel_gpu_ms_status == "ok"

    def test_frozen(self):
        m = SplitTimingMetrics(1.0, 0.1, 0.1, "ok", "ok")
        with pytest.raises((AttributeError, TypeError)):
            m.e2e_ms = 2.0


# -----------------------------------------------------------------------------
# _maybe_clone — tensor clones, non-tensor passes through
# -----------------------------------------------------------------------------


class TestMaybeClone:
    def test_tensor_is_cloned(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        c = _maybe_clone(t)
        assert c is not t
        assert torch.equal(c, t)

    def test_tensor_clone_independent(self):
        t = torch.tensor([1.0, 2.0])
        c = _maybe_clone(t)
        c[0] = 99.0
        assert t[0].item() == 1.0  # original unaffected

    @pytest.mark.parametrize("value", [42, 3.14, "string", None, (1, 2), [1, 2, 3]])
    def test_non_tensor_passes_through(self, value):
        assert _maybe_clone(value) is value


# -----------------------------------------------------------------------------
# _measure_kernel_gpu_cupti — CPU-mockable: exception + fallback + ok paths
# -----------------------------------------------------------------------------


class TestMeasureKernelGpuCupti:
    """All branches of _measure_kernel_gpu_cupti — no GPU required, the
    `bench_gpu_time_with_cupti` call is mocked out."""

    def _build_runnable(self):
        # The Runnable's setup_for_workload is called before the timer; mock its
        # callable so we don't accidentally exercise real code.
        called = {"setup": 0, "run": 0}

        def _setup(*a):
            called["setup"] += 1
            return {}

        def _run(*a):
            called["run"] += 1

        return _make_runnable(_run, _setup), called

    def test_returns_no_cupti_when_import_fails(self):
        runnable, called = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            side_effect=ModuleNotFoundError("No module named 'cupti'"),
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms is None  # unavailable is None, never a 0.0 sentinel
        assert status.startswith("no_cupti:")
        assert "ModuleNotFoundError" in status
        # setup is still called before the timer — verifies the contract
        assert called["setup"] == 1

    def test_returns_no_cupti_when_runtime_error(self):
        runnable, _ = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            side_effect=RuntimeError("Incompatible CUPTI Library"),
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms is None
        assert status == "no_cupti:RuntimeError"

    def test_returns_cupti_no_samples_on_empty(self):
        runnable, _ = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti", return_value=[]
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms is None
        assert status == "cupti_no_samples"

    def test_returns_ok_with_median_when_clean(self):
        runnable, _ = self._build_runnable()
        # Three samples; median is the middle one.
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            return_value=[0.1, 0.2, 0.3],
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=3, device="cuda:0"
            )
        assert ms == pytest.approx(0.2)
        assert status == "ok"

    @pytest.mark.parametrize("cold_l2_cache", [True, False])
    def test_passes_l2_policy_to_cupti(self, cold_l2_cache):
        runnable, _ = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            return_value=[0.2],
        ) as bench:
            _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0", cold_l2_cache=cold_l2_cache
            )
        assert bench.call_args.kwargs["cold_l2_cache"] is cold_l2_cache

    def test_detects_silent_cupti_fallback_via_warning(self):
        """Flashinfer's `bench_gpu_time_with_cupti` emits a UserWarning when it
        can't reach a real CUPTI runtime and silently falls back to CUDA events.
        Without the warning-capture logic, we'd return status="ok" — hiding
        that the numbers are CUDA-event timings, not CUPTI activity sums. The
        new code surfaces this as `cupti_fallback:cuda_events`."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn(
                "CUPTI is not installed. Try 'pip install -U cupti-python'. "
                "Falling back to CUDA events for benchmarking.",
                UserWarning,
                stacklevel=1,
            )
            return [0.1, 0.2, 0.3]

        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=3, device="cuda:0"
            )
        assert ms == pytest.approx(0.2)
        assert status == "cupti_fallback:cuda_events"

    def test_detects_version_mismatch_fallback(self):
        """Same as above but with the "needs to be >= 13.0.0" wording flashinfer
        uses for ABI-version mismatches."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn(
                "CUPTI needs to be >= 13.0.0. Falling back to CUDA events.",
                UserWarning,
                stacklevel=1,
            )
            return [1.0]

        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            _, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert status == "cupti_fallback:cuda_events"

    def test_unrelated_warning_does_not_trigger_fallback_status(self):
        """A non-CUPTI warning shouldn't be misclassified as a CUPTI fallback."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn("Some unrelated deprecation notice.", DeprecationWarning, stacklevel=1)
            return [0.5]

        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            _, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert status == "ok"


class TestCudaEventL2Policy:
    @pytest.mark.parametrize("cold_l2_cache", [True, False])
    def test_kernel_measurement_passes_l2_policy(self, cold_l2_cache):
        runnable = _make_runnable(lambda *_args: None, lambda *_args: {})
        with patch(
            "flashinfer_bench.bench.timing.split_timing.bench_gpu_time_with_cuda_event",
            return_value=[0.4],
        ) as bench:
            ms, status = _measure_kernel_cudaevent(
                runnable, [42], warmup=1, iters=1, device="cuda:0", cold_l2_cache=cold_l2_cache
            )
        assert ms == pytest.approx(0.4)
        assert status == "ok"
        assert bench.call_args.kwargs["cold_l2_cache"] is cold_l2_cache

    @pytest.mark.parametrize("cold_l2_cache", [True, False])
    def test_e2e_applies_l2_policy_per_iteration(self, cold_l2_cache, monkeypatch):
        """e2e is host wall-clock — no flashinfer helper involved. The L2 flush
        runs once per invocation (warmup + timed, mirroring flashinfer's
        dry-run flush behavior) when cold, never when warm."""
        import flashinfer_bench.bench.timing.split_timing as st

        calls = {"flush": 0, "setup": 0, "run": 0}

        def _fake_flusher(_dev):
            return lambda: calls.__setitem__("flush", calls["flush"] + 1)

        monkeypatch.setattr(st, "_make_l2_flusher", _fake_flusher)
        monkeypatch.setattr(st.torch.cuda, "synchronize", lambda *_a, **_k: None)

        def _setup(*_args):
            calls["setup"] += 1
            return {}

        def _run(*_args):
            calls["run"] += 1

        runnable = _make_runnable(_run, _setup)
        WARMUP, ITERS = 2, 3
        ms = _measure_e2e(
            runnable, [42], warmup=WARMUP, iters=ITERS, device="cuda:0", cold_l2_cache=cold_l2_cache
        )
        assert ms >= 0.0
        assert calls["flush"] == (WARMUP + ITERS if cold_l2_cache else 0)
        # setup + run stay paired inside the timed region, once per invocation
        assert calls["setup"] == calls["run"] == WARMUP + ITERS


# -----------------------------------------------------------------------------
# GPU-required tests below
# -----------------------------------------------------------------------------

cuda_available = pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA devices not available"
)


@cuda_available
class TestMeasureE2E:
    def test_setup_called_once_per_iter(self):
        """e2e re-runs setup_for_workload inside the timing region.
        Verify setup and run stay paired for every full invocation."""
        called = {"setup": 0, "run": 0}

        def _setup(t):
            called["setup"] += 1
            return {}

        def _run(t):
            called["run"] += 1
            _ = t.sum()

        runnable = _make_runnable(_run, _setup)
        x = torch.randn(64, device="cuda")
        WARMUP, ITERS = 3, 5
        ms = _measure_e2e(runnable, [x], warmup=WARMUP, iters=ITERS, device="cuda:0")
        # Host wall-clock loop runs exactly warmup + iters invocations.
        assert called["setup"] == called["run"] == WARMUP + ITERS
        assert ms > 0.0

    def test_tensors_are_cloned_per_iter(self):
        """e2e clones tensor args each iter. Mutating cloned input inside run()
        should not leak back to the caller's original tensor."""
        original = torch.zeros(8, device="cuda")

        def _setup(t):
            return {}

        def _run(t):
            t.add_(1.0)

        runnable = _make_runnable(_run, _setup)
        _measure_e2e(runnable, [original], warmup=1, iters=2, device="cuda:0")
        # If clones leaked, original would have been incremented 3 times.
        assert torch.equal(original, torch.zeros(8, device="cuda"))


@cuda_available
class TestMeasureKernelCudaEvent:
    def test_happy_path_returns_ok(self):
        """A simple eager kernel should produce a CUDA Event median > 0."""
        a = torch.randn(128, device="cuda")
        b = torch.randn(128, device="cuda")
        out = torch.empty(128, device="cuda")

        def _setup(a, b):
            return {}

        def _run(a, b):
            torch.add(a, b, out=out)

        runnable = _make_runnable(_run, _setup)
        ms, status = _measure_kernel_cudaevent(runnable, [a, b], warmup=3, iters=5, device="cuda:0")
        assert status == "ok"
        assert ms > 0.0

    def test_setup_runs_once_outside_measurement(self):
        a = torch.randn(64, device="cuda")
        called = {"setup": 0, "run": 0}

        def _setup(t):
            called["setup"] += 1
            return {}

        def _run(t):
            called["run"] += 1
            _ = t.sum()

        runnable = _make_runnable(_run, _setup)
        ms, status = _measure_kernel_cudaevent(runnable, [a], warmup=2, iters=3, device="cuda:0")
        assert called["setup"] == 1
        assert called["run"] >= 5
        assert status == "ok"
        assert ms > 0.0


@cuda_available
class TestTimeRunnableSplitTiming:
    """End-to-end: invoke the public API on a mock Runnable, verify all three
    metrics + both status strings come back populated and finite."""

    def test_three_metrics_populated(self):
        a = torch.randn(256, device="cuda")
        b = torch.randn(256, device="cuda")
        out = torch.empty(256, device="cuda")

        def _setup(a, b):
            return {}

        def _run(a, b):
            torch.mul(a, b, out=out)

        runnable = _make_runnable(_run, _setup)
        m = time_runnable_split_timing(runnable, [a, b], warmup=3, iters=5, device="cuda:0")

        assert isinstance(m, SplitTimingMetrics)
        assert m.e2e_ms > 0.0
        assert m.kernel_ms > 0.0
        # kernel_gpu_ms is None if cupti is unavailable; the status string then
        # documents the reason. Either way, status must be one of the known
        # values.
        assert m.kernel_gpu_ms is None or m.kernel_gpu_ms > 0.0
        assert m.kernel_ms_status == "ok"
        assert m.kernel_gpu_ms_status in (
            "ok",
            "cupti_no_samples",
            "cupti_fallback:cuda_events",
        ) or m.kernel_gpu_ms_status.startswith("no_cupti:")

    def test_e2e_at_least_kernel(self):
        """e2e_ms >= kernel_ms (modulo run-to-run noise — give it some slack).
        Rationale: e2e includes everything kernel_ms does, plus setup and a
        per-iteration device sync inside the timed window."""
        a = torch.randn(512, device="cuda")
        out = torch.empty(512, device="cuda")

        def _setup(t):
            return {}

        def _run(t):
            # add(t,t) supports out=; relu(t, out=…) is missing on some torch builds.
            torch.add(t, t, out=out)

        runnable = _make_runnable(_run, _setup)
        m = time_runnable_split_timing(runnable, [a], warmup=5, iters=10, device="cuda:0")
        # 2x slack to absorb noise on extremely small kernels; the inequality
        # is robust on any non-trivial workload.
        assert m.e2e_ms >= m.kernel_ms * 0.5


# -----------------------------------------------------------------------------
# Cross-trial phase rotation
# -----------------------------------------------------------------------------

_SPLIT_MOD = "flashinfer_bench.bench.timing.split_timing"


class TestPhaseRotation:
    """Rotation must be deterministic, cover every slot, and stay opt-out-able.

    CPU-only: the ordering rule is a pure function, and the dispatch is checked
    with the three measurement helpers mocked out."""

    def test_full_cycle_puts_every_phase_in_every_slot(self):
        orders = [_rotated_phase_order(i, True) for i in range(3)]
        assert orders == [
            ("kernel", "kernel_gpu", "e2e"),
            ("kernel_gpu", "e2e", "kernel"),
            ("e2e", "kernel", "kernel_gpu"),
        ]
        # The property that makes the cross-trial mean position-unbiased: over a
        # full cycle each slot is occupied by each phase exactly once.
        for slot in range(len(_PHASE_SEQUENCE)):
            assert {o[slot] for o in orders} == set(_PHASE_SEQUENCE)

    def test_rotation_is_deterministic_and_wraps(self):
        # Same trial index always yields the same order — no randomization, so
        # a run stays reproducible phase-for-phase.
        for i in range(10):
            assert _rotated_phase_order(i, True) == _rotated_phase_order(i, True)
            assert _rotated_phase_order(i, True) == _rotated_phase_order(
                i + len(_PHASE_SEQUENCE), True
            )

    def test_disabled_pins_legacy_fixed_order(self):
        for i in range(5):
            assert _rotated_phase_order(i, False) == _PHASE_SEQUENCE

    def test_trial_zero_matches_pre_rotation_behavior(self):
        # Guards backward compatibility: the first trial runs the same order the
        # engine used before rotation existed.
        assert _rotated_phase_order(0, True) == _PHASE_SEQUENCE

    def _dispatch(self, trial_index, rotate=True):
        """Run the public API with all three measurements mocked; return the
        observed execution order and the metrics object."""
        runnable = _make_runnable(lambda *a: None)
        calls = []

        def _fake_kernel(*a, **k):
            calls.append("kernel")
            return 1.0, "ok"

        def _fake_gpu(*a, **k):
            calls.append("kernel_gpu")
            return 2.0, "ok"

        def _fake_e2e(*a, **k):
            calls.append("e2e")
            return 3.0

        with patch(f"{_SPLIT_MOD}._measure_kernel_cudaevent", _fake_kernel), patch(
            f"{_SPLIT_MOD}._measure_kernel_gpu_cupti", _fake_gpu
        ), patch(f"{_SPLIT_MOD}._measure_e2e", _fake_e2e), patch(
            f"{_SPLIT_MOD}._cool_down", lambda *a, **k: None
        ), patch(
            # No GPU needed once the measurements are mocked.
            "torch.cuda.device",
            lambda d: contextlib.nullcontext(),
        ):
            m = time_runnable_split_timing(
                runnable,
                [],
                warmup=1,
                iters=1,
                device="cuda:0",
                trial_index=trial_index,
                rotate_phases=rotate,
            )
        return tuple(calls), m

    def test_dispatch_executes_in_rotated_order(self):
        calls, m = self._dispatch(trial_index=1)
        assert calls == ("kernel_gpu", "e2e", "kernel")
        assert m.phase_order == ("kernel_gpu", "e2e", "kernel")

    def test_dispatch_routes_values_to_correct_fields(self):
        # Run order must not change which metric a value lands in.
        for trial_index in range(3):
            _, m = self._dispatch(trial_index=trial_index)
            assert (m.kernel_ms, m.kernel_gpu_ms, m.e2e_ms) == (1.0, 2.0, 3.0)
            assert m.kernel_ms_status == "ok"
            assert m.kernel_gpu_ms_status == "ok"

    def test_dispatch_honors_rotation_opt_out(self):
        calls, m = self._dispatch(trial_index=2, rotate=False)
        assert calls == _PHASE_SEQUENCE
        assert m.phase_order == _PHASE_SEQUENCE
