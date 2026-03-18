"""Benchmark one trace solution against the definition reference on the same workload."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashinfer_bench.apply.trace_filter import hardware_matches, solution_language_matches
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.bench.utils import compute_error_stats, gen_inputs
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one solution from a trace set against the definition reference on a "
            "single matched workload."
        )
    )
    parser.add_argument("--trace-set-path", required=True, help="Trace-set root path")
    parser.add_argument("--definition", required=True, help="Definition name, e.g. fused_add_rmsnorm_h4096")
    parser.add_argument("--solution-name", help="Optional exact solution name to benchmark")
    parser.add_argument(
        "--trace-hardware-contains",
        action="append",
        default=[],
        help="Retain only traces whose hardware string contains this token. Can be passed multiple times.",
    )
    parser.add_argument(
        "--solution-language",
        action="append",
        default=[],
        help="Retain only solutions whose spec.language matches this token, e.g. cuda or triton.",
    )
    parser.add_argument("--batch-size", type=int, help="Optional batch_size axis filter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--output-json", help="Optional path to write the summary JSON")
    return parser


def allocate_outputs(definition, inputs: List[Any], device: str) -> List[torch.Tensor]:
    var_values = definition.get_axes_values_from_inputs(inputs)
    output_shapes = definition.get_output_shapes(var_values)
    return [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(output_shapes, definition.torch_output_dtypes)
    ]


def normalize_result(definition, result: Any, device: str) -> List[torch.Tensor]:
    def to_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device) if str(value.device) != device else value
        return torch.tensor(value, dtype=dtype, device=device)

    if isinstance(result, (tuple, list)):
        if len(result) != len(definition.torch_output_dtypes):
            raise ValueError("Returned output count does not match definition outputs")
        return [to_tensor(value, definition.torch_output_dtypes[idx]) for idx, value in enumerate(result)]

    if len(definition.torch_output_dtypes) != 1:
        raise ValueError("Single value returned but definition expects multiple outputs")
    return [to_tensor(result, definition.torch_output_dtypes[0])]


def _run_runnable(runnable, definition, inputs: List[Any], device: str):
    if runnable.metadata.destination_passing_style:
        outputs = allocate_outputs(definition, inputs, device)
        runnable(*inputs, *outputs)
        return outputs
    return normalize_result(definition, runnable(*inputs), device)


def _timing_args(runnable, definition, inputs: List[Any], device: str) -> List[Any]:
    if runnable.metadata.destination_passing_style:
        return list(inputs) + allocate_outputs(definition, inputs, device)
    return list(inputs)


def _pick_trace(trace_set: TraceSet, args: argparse.Namespace) -> Trace:
    hardware_filters = [value.strip() for value in args.trace_hardware_contains if value.strip()]
    language_filters = [value.strip() for value in args.solution_language if value.strip()]

    candidates: list[Trace] = []
    for trace in trace_set.traces.get(args.definition, []):
        if trace.evaluation is None or trace.evaluation.status != EvaluationStatus.PASSED:
            continue
        if args.batch_size is not None and int(trace.workload.axes.get("batch_size", -1)) != args.batch_size:
            continue
        if hardware_filters and not hardware_matches(trace.evaluation.environment.hardware, hardware_filters):
            continue
        if args.solution_name and trace.solution != args.solution_name:
            continue
        solution = trace_set.get_solution(trace.solution) if trace.solution else None
        if language_filters:
            if solution is None:
                continue
            language = str(getattr(solution.spec.language, "value", solution.spec.language))
            if not solution_language_matches(language, language_filters):
                continue
        candidates.append(trace)

    if not candidates:
        raise ValueError("No matching PASSED trace found for the requested filters")

    return max(candidates, key=lambda trace: trace.evaluation.performance.speedup_factor)


def main() -> None:
    args = build_parser().parse_args()
    trace_set = TraceSet.from_path(args.trace_set_path)
    definition = trace_set.definitions.get(args.definition)
    if definition is None:
        raise ValueError(f"Unknown definition: {args.definition}")

    trace = _pick_trace(trace_set, args)
    solution = trace_set.get_solution(trace.solution)
    if solution is None:
        raise ValueError(f"Missing solution object for trace solution: {trace.solution}")

    builder = BuilderRegistry.get_instance()
    ref_runnable = builder.build_reference(definition)
    sol_runnable = builder.build(definition, solution)

    inputs = gen_inputs(definition, trace.workload, device=args.device)

    ref_outputs = _run_runnable(ref_runnable, definition, inputs, args.device)
    sol_outputs = _run_runnable(sol_runnable, definition, inputs, args.device)

    max_abs_error = 0.0
    max_rel_error = 0.0
    for sol_tensor, ref_tensor in zip(sol_outputs, ref_outputs, strict=True):
        abs_err, rel_err, _, _ = compute_error_stats(
            sol_tensor,
            ref_tensor,
            type("Cfg", (), {"atol": args.atol, "rtol": args.rtol, "required_matched_ratio": 1.0})(),
        )
        max_abs_error = max(max_abs_error, abs_err)
        max_rel_error = max(max_rel_error, rel_err)

    ref_latency_ms = time_runnable(
        ref_runnable,
        _timing_args(ref_runnable, definition, inputs, args.device),
        args.warmup_runs,
        args.benchmark_runs,
        args.device,
    )
    sol_latency_ms = time_runnable(
        sol_runnable,
        _timing_args(sol_runnable, definition, inputs, args.device),
        args.warmup_runs,
        args.benchmark_runs,
        args.device,
    )

    summary = {
        "definition": args.definition,
        "device": args.device,
        "selected_trace_solution": trace.solution,
        "selected_trace_hardware": trace.evaluation.environment.hardware,
        "selected_trace_workload_axes": dict(trace.workload.axes),
        "selected_trace_latency_ms": trace.evaluation.performance.latency_ms,
        "selected_trace_reference_latency_ms": trace.evaluation.performance.reference_latency_ms,
        "selected_trace_speedup_factor": trace.evaluation.performance.speedup_factor,
        "measured_reference_latency_ms": ref_latency_ms,
        "measured_solution_latency_ms": sol_latency_ms,
        "measured_speedup_factor": ref_latency_ms / sol_latency_ms if sol_latency_ms > 0 else math.inf,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
