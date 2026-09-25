"""Default evaluator for general kernel correctness and performance."""

import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
from flashinfer_bench.bench.evaluators.evaluator import Evaluator
from flashinfer_bench.bench.runner.runner import BaselineHandle, DeviceBaseline
from flashinfer_bench.bench.timing import (
    SplitTimingMetrics,
    time_runnable,
    time_runnable_split_timing,
)
from flashinfer_bench.bench.utils import (
    compute_error_stats,
    gen_inputs,
    is_sampling_operation,
    load_safetensors,
    make_eval,
)
from flashinfer_bench.compile import BuilderRegistry, Runnable
from flashinfer_bench.data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)

from .utils import allocate_outputs, normalize_result

# Floating tensors with at least this many dims count as "payload" (q/kv/activation
# data) for the setup-hook contract; lower-rank float tensors (per-channel scales,
# scalar knobs) and all integer/bool tensors count as metadata.
_PAYLOAD_MIN_NDIM = 2


def _has_setup_hook(runnable: Runnable) -> bool:
    """True only for real Runnables that declare a setup hook (mock-safe)."""
    return getattr(runnable, "has_setup_hook", False) is True


def _full_call_target(sol_runnable: Runnable):
    """Return the callable the single-metric latency path should time.

    For setup-hook solutions, setup() must run INSIDE the timed region every
    iteration — identical semantics to a plain solution that does its planning
    inline — so ``latency_ms`` stays comparable across solution styles and
    with pre-setup-hook traces. Solutions without a setup hook are passed
    through untouched (bit-identical to the original path).
    """
    if not _has_setup_hook(sol_runnable):
        return sol_runnable

    def _setup_and_run(*args: Any) -> Any:
        sol_runnable.setup_for_workload(*args)
        return sol_runnable(*args)

    return _setup_and_run


class DefaultEvaluator(Evaluator):
    @classmethod
    def can_evaluate(cls, definition: Definition) -> bool:
        return True

    @classmethod
    def build_baseline(
        cls,
        definition: Definition,
        workload: Workload,
        cfg: ResolvedEvalConfig,
        device: str,
        trace_set_root: Optional[Path] = None,
    ) -> DeviceBaseline:
        # Reference is always value-returning style
        ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
        loaded_safe_tensors = (
            load_safetensors(definition, workload, trace_set_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: List[List[Any]] = []
        outputs: List[List[torch.Tensor]] = []

        for _ in range(cfg.num_trials):
            inp = gen_inputs(definition, workload, device=device, safe_tensors=loaded_safe_tensors)
            inputs.append(inp)

            with torch.no_grad():
                result = ref_runnable(*inp)
            torch.cuda.synchronize(device)
            outputs.append(normalize_result(definition, result, device))

        if cfg.profile_baseline:
            latencies: List[float] = []
            for inp in inputs:
                ms = time_runnable(ref_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
                latencies.append(ms)

            mean_latency_ms = sum(latencies) / float(len(latencies))
        else:
            mean_latency_ms = 0.0

        handle = BaselineHandle(uuid.uuid4().hex)

        return DeviceBaseline(
            handle=handle,
            definition=definition,
            device=device,
            inputs=inputs,
            outputs=outputs,
            mean_latency_ms=mean_latency_ms,
        )

    @classmethod
    def check_correctness(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_outputs: List[List[torch.Tensor]],
        cfg: ResolvedEvalConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        max_abs = 0.0
        max_rel = 0.0
        numerical_incorrect = False
        is_dps = sol_runnable.metadata.destination_passing_style

        for trial, inp in enumerate(inputs):
            try:
                if is_dps:
                    # DPS style: allocate outputs and call with them
                    out = allocate_outputs(definition, inp, device)
                    # Per-workload setup (no-op if the solution defines no setup hook).
                    sol_runnable.setup_for_workload(*inp, *out)
                    with torch.no_grad():
                        sol_runnable(*inp, *out)
                    torch.cuda.synchronize(device)
                else:
                    # Value-returning style: call and normalize result
                    sol_runnable.setup_for_workload(*inp)
                    with torch.no_grad():
                        result = sol_runnable(*inp)
                    torch.cuda.synchronize(device)
                    out = normalize_result(definition, result, device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            ref_out = ref_outputs[trial]

            for sol_tensor, ref_tensor in zip(out, ref_out):
                # Shape validation
                if tuple(sol_tensor.shape) != tuple(ref_tensor.shape):
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_SHAPE, device=device, log_path=log_path
                    )

                # Dtype validation
                if sol_tensor.dtype != ref_tensor.dtype:
                    return None, make_eval(
                        status=EvaluationStatus.INCORRECT_DTYPE, device=device, log_path=log_path
                    )

                # Non-finite values check
                non_finite_err_val = None
                if torch.isinf(sol_tensor).any().item():
                    non_finite_err_val = float("inf")
                elif torch.isnan(sol_tensor).any().item():
                    non_finite_err_val = float("nan")

                if non_finite_err_val is not None:
                    correctness = Correctness(
                        max_relative_error=non_finite_err_val, max_absolute_error=non_finite_err_val
                    )
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                    )

                # Compute error statistics
                abs_err, rel_err, exceeds_tol, _ = compute_error_stats(sol_tensor, ref_tensor, cfg)

                if exceeds_tol:
                    numerical_incorrect = True

                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)

        correctness = Correctness(max_relative_error=max_rel, max_absolute_error=max_abs)

        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        if _has_setup_hook(sol_runnable):
            evaluation = cls._check_setup_payload_independence(
                definition=definition,
                sol_runnable=sol_runnable,
                inputs=inputs,
                cfg=cfg,
                log_path=log_path,
                device=device,
            )
            if evaluation is not None:
                return None, evaluation

        return correctness, None

    @classmethod
    def _check_setup_payload_independence(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        cfg: ResolvedEvalConfig,
        log_path: str,
        device: str,
    ) -> Optional[Evaluation]:
        """Enforce the setup-hook contract: state must not depend on payload values.

        ``setup()`` receives the real input tensors, so a solution could compute
        its full result there — outside every timed region — and have ``run()``
        replay the cached answer. Correctness on the original inputs cannot catch
        this. Enforcement: bind the cached state to the first trial's inputs,
        re-randomize the floating-point payload tensors (ndim >= 2) IN PLACE
        while leaving metadata (integer/bool tensors, scalars, low-rank float
        tensors) untouched so legitimately cached plans stay valid, then re-run
        ``run()`` with the stale state and compare against a freshly computed
        reference on the mutated inputs.

        Returns None when the check passes (or is vacuous), otherwise a failed
        Evaluation. The original payload values are restored before returning,
        so subsequent timing runs on the true workload data (value-dependent
        kernels, safetensors-captured payloads).
        """
        inp = inputs[0]
        is_dps = sol_runnable.metadata.destination_passing_style

        # 1. Bind cached state to the ORIGINAL payload values.
        if is_dps:
            out_tensors = allocate_outputs(definition, inp, device)
            sol_runnable.setup_for_workload(*inp, *out_tensors)
        else:
            sol_runnable.setup_for_workload(*inp)

        names = list(definition.inputs.keys())
        saved: List[Tuple[torch.Tensor, torch.Tensor]] = []
        failure: Optional[Correctness] = None
        try:
            # 2. Re-randomize floating payload tensors in place (same generator
            #    family as gen_inputs; sampling "probs" keeps its simplex
            #    property), remembering originals for restoration.
            with torch.no_grad():
                for name, arg in zip(names, inp):
                    if not isinstance(arg, torch.Tensor):
                        continue
                    if not arg.is_floating_point() or arg.ndim < _PAYLOAD_MIN_NDIM:
                        continue
                    saved.append((arg, arg.clone()))
                    fresh = torch.randn(arg.shape, dtype=torch.float32, device=arg.device)
                    if is_sampling_operation(definition) and name == "probs":
                        fresh = torch.softmax(fresh, dim=-1)
                    elif arg.element_size() == 1:
                        # low-precision floats (fp8/fp4): clamp like gen_inputs does
                        fresh = fresh.clamp_(-2.0, 2.0)
                    arg.copy_(fresh.to(arg.dtype))
            if not saved:
                return None  # no payload tensors — nothing to enforce

            # 3. Fresh reference on the mutated inputs.
            ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
            with torch.no_grad():
                ref_result = ref_runnable(*inp)
            torch.cuda.synchronize(device)
            ref_out = normalize_result(definition, ref_result, device)

            # 4. Solution re-run with the STALE cached state (no setup re-run).
            if is_dps:
                out_tensors = allocate_outputs(definition, inp, device)
                with torch.no_grad():
                    sol_runnable(*inp, *out_tensors)
                torch.cuda.synchronize(device)
                out = out_tensors
            else:
                with torch.no_grad():
                    result = sol_runnable(*inp)
                torch.cuda.synchronize(device)
                out = normalize_result(definition, result, device)

            # 5. Compare, with a non-finite screen mirroring the main loop
            #    (compute_error_stats treats NaN as within tolerance).
            for sol_tensor, ref_tensor in zip(out, ref_out):
                if not torch.isfinite(ref_tensor.to(torch.float32)).all().item():
                    # The reference degenerated on randomized payload
                    # (domain-constrained op) — comparison is meaningless here.
                    continue
                if not torch.isfinite(sol_tensor.to(torch.float32)).all().item():
                    failure = Correctness(
                        max_relative_error=float("nan"), max_absolute_error=float("nan")
                    )
                    break
                abs_err, rel_err, exceeds_tol, _ = compute_error_stats(sol_tensor, ref_tensor, cfg)
                if exceeds_tol:
                    failure = Correctness(max_relative_error=rel_err, max_absolute_error=abs_err)
                    break
        except Exception:
            traceback.print_exc()
            return make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )
        finally:
            # Restore original payload values regardless of outcome.
            with torch.no_grad():
                for arg, original in saved:
                    arg.copy_(original)

        if failure is not None:
            print(
                "setup-payload-independence check FAILED: run() with the cached "
                "setup() state no longer matches the reference after payload "
                "re-randomization. setup() state must not depend on floating-point "
                f"payload values (max_abs={failure.max_absolute_error:.3e}, "
                f"max_rel={failure.max_relative_error:.3e}).",
                file=sys.stderr,
            )
            return make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=failure,
                extra_msg=(
                    "setup_payload_dependence: run() with cached setup() state does not "
                    "match a fresh reference after payload re-randomization"
                ),
            )
        return None

    @classmethod
    def eval_performance(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_mean_latency_ms: float,
        cfg: ResolvedEvalConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Performance, Optional[Evaluation]]:
        is_dps = sol_runnable.metadata.destination_passing_style

        def _args_for(inp: List[Any]) -> List[Any]:
            if is_dps:
                return list(inp) + allocate_outputs(definition, inp, device)
            return list(inp)

        if cfg.split_timing:
            try:
                trial_metrics: List[SplitTimingMetrics] = []
                trial_latencies: List[float] = []
                for trial_index, inp in enumerate(inputs):
                    args = _args_for(inp)
                    # time_runnable_split_timing handles setup invocation internally
                    # (once-outside for kernel_ms / kernel_gpu_ms; per-iter for
                    # e2e_ms). Do NOT call setup_for_workload here.
                    #
                    # trial_index drives the phase rotation: over num_trials the
                    # three split phases each occupy every slot, so the means
                    # below cannot inherit a fixed-position bias from the
                    # measurement schedule.
                    metrics = time_runnable_split_timing(
                        sol_runnable,
                        args,
                        cfg.warmup_runs,
                        cfg.iterations,
                        device,
                        cold_l2_cache=cfg.cold_l2_cache,
                        trial_index=trial_index,
                        rotate_phases=cfg.split_phase_rotation,
                    )
                    trial_metrics.append(metrics)
                    # latency_ms keeps its single-metric semantics under split
                    # timing: the full solution call (setup inside the timed
                    # region for setup-hook solutions) measured by the same
                    # mechanism as reference_latency_ms — so speedup_factor
                    # stays apples-to-apples. Deliberately kept OUT of the phase
                    # rotation and always measured last, behind the trailing
                    # cool-down: its protocol must stay identical to non-split
                    # runs, since it is the number historical traces and the
                    # leaderboard compare against.
                    lat_ms = time_runnable(
                        _full_call_target(sol_runnable),
                        args,
                        cfg.warmup_runs,
                        cfg.iterations,
                        device,
                    )
                    trial_latencies.append(lat_ms)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            if not trial_metrics:
                print("Failed to collect solution latencies", file=sys.stderr)
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            n = float(len(trial_metrics))
            lat_mean = sum(trial_latencies) / n
            e2e_mean = sum(m.e2e_ms for m in trial_metrics) / n
            kernel_mean = sum(m.kernel_ms for m in trial_metrics) / n
            # Status: report "ok" only if every trial succeeded; else surface
            # the first non-ok value so the user can tell why fallback fired.
            kernel_status = next(
                (m.kernel_ms_status for m in trial_metrics if m.kernel_ms_status != "ok"), "ok"
            )
            # kernel_gpu_ms: never average across mechanisms or unavailable
            # trials. Real CUPTI activity sums and CUDA-event fallback numbers
            # are different measurements; None marks "unavailable" (no 0.0
            # sentinels contaminating a mean).
            cupti_vals = [
                m.kernel_gpu_ms
                for m in trial_metrics
                if m.kernel_gpu_ms_status == "ok" and m.kernel_gpu_ms is not None
            ]
            fallback_vals = [
                m.kernel_gpu_ms
                for m in trial_metrics
                if m.kernel_gpu_ms_status == "cupti_fallback:cuda_events"
                and m.kernel_gpu_ms is not None
            ]
            if cupti_vals:
                kernel_gpu_mean = sum(cupti_vals) / float(len(cupti_vals))
                kernel_gpu_status = (
                    "ok"
                    if len(cupti_vals) == len(trial_metrics)
                    else f"ok_partial:{len(cupti_vals)}/{len(trial_metrics)}"
                )
            elif fallback_vals:
                kernel_gpu_mean = sum(fallback_vals) / float(len(fallback_vals))
                kernel_gpu_status = "cupti_fallback:cuda_events"
            else:
                kernel_gpu_mean = None
                kernel_gpu_status = next(
                    (
                        m.kernel_gpu_ms_status
                        for m in trial_metrics
                        if m.kernel_gpu_ms_status != "ok"
                    ),
                    "cupti_no_samples",
                )
            performance = Performance(
                latency_ms=lat_mean,
                reference_latency_ms=ref_mean_latency_ms,
                speedup_factor=(ref_mean_latency_ms / lat_mean) if lat_mean > 0 else 0.0,
                e2e_ms=e2e_mean,
                kernel_ms=kernel_mean,
                kernel_gpu_ms=kernel_gpu_mean,
                kernel_ms_status=kernel_status,
                kernel_gpu_ms_status=kernel_gpu_status,
                l2_cache_mode="cold" if cfg.cold_l2_cache else "warm",
            )
            return performance, None

        sol_latencies: List[float] = []
        try:
            for inp in inputs:
                args = _args_for(inp)
                # For setup-hook solutions, _full_call_target times setup + run
                # together every iteration — identical semantics to a plain
                # solution doing its planning inline, so latency_ms stays
                # comparable with existing traces. Solutions without a setup
                # hook go through bit-identical to the pre-setup-hook path.
                ms = time_runnable(
                    _full_call_target(sol_runnable), args, cfg.warmup_runs, cfg.iterations, device
                )
                sol_latencies.append(ms)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        if not sol_latencies:
            print("Failed to collect solution latencies", file=sys.stderr)
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        sol_mean_latency_ms = sum(sol_latencies) / float(len(sol_latencies))
        performance = Performance(
            latency_ms=sol_mean_latency_ms,
            reference_latency_ms=ref_mean_latency_ms,
            speedup_factor=(ref_mean_latency_ms / sol_mean_latency_ms),
        )

        return performance, None
