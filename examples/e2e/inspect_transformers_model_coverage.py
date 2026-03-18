"""Inspect a Transformers model for FlashInfer-Bench-replaceable operators."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIB_E2E_PYTHON = "/data/workspace/airulan/conda_envs/fib_e2e/bin/python"


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_gemm_dimensions(def_name: str) -> Optional[tuple[int, int]]:
    match = re.fullmatch(r"gemm_n(\d+)_k(\d+)", def_name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _axis_as_int(axes: Dict[str, Any], name: str) -> Optional[int]:
    value = axes.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _estimate_gemm_tflops(m: int, n: int, k: int, latency_ms: float) -> Optional[float]:
    if latency_ms <= 0.0:
        return None
    flop_count = 2.0 * float(m) * float(n) * float(k)
    return flop_count / (latency_ms * 1.0e-3) / 1.0e12


def _best_trace_summary(traces: List[Any], trace_set: Any) -> Dict[str, Any]:
    if not traces:
        return {}

    best_trace = max(
        traces,
        key=lambda trace: trace.evaluation.performance.speedup_factor,
    )
    solution = trace_set.get_solution(best_trace.solution) if best_trace.solution else None
    perf = best_trace.evaluation.performance
    return {
        "solution": best_trace.solution,
        "author": solution.author if solution else "",
        "target_hardware": list(solution.spec.target_hardware) if solution else [],
        "hardware": best_trace.evaluation.environment.hardware,
        "speedup_factor": perf.speedup_factor,
        "latency_ms": perf.latency_ms,
        "reference_latency_ms": perf.reference_latency_ms,
        "workload_axes": dict(best_trace.workload.axes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Hugging Face model and report FlashInfer-Bench definition coverage "
            "for replaceable GEMM/RMSNorm operators."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-mode",
        choices=["empty", "pretrained"],
        default="empty",
        help="Use init_empty_weights skeleton or load full pretrained weights",
    )
    parser.add_argument("--trace-set-path", help="Optional trace-set root for coverage checks")
    parser.add_argument(
        "--trace-hardware-contains",
        action="append",
        default=[],
        help=(
            "Retain only traces whose evaluation.environment.hardware contains this substring. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--max-atol", type=float, default=1e-2)
    parser.add_argument("--max-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--sample-module-names",
        type=int,
        default=8,
        help="How many module names to keep per matched definition",
    )
    parser.add_argument(
        "--include-modules",
        action="store_true",
        help="Include the full named module list in the JSON output",
    )
    parser.add_argument("--output-json", help="Optional path to write the inspection summary JSON")
    parser.add_argument(
        "--include-solution-stats",
        action="store_true",
        help="Include per-solution isolated-kernel performance summaries for matched definitions",
    )
    parser.add_argument(
        "--max-solutions-per-definition",
        type=int,
        default=8,
        help="Maximum number of solution summaries to emit per definition",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit verbose module_type_counts from the JSON output",
    )
    return parser


def _require_module(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{name}' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc


def _load_model_skeleton(args: argparse.Namespace):
    _require_module("torch")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    if args.load_mode == "pretrained":
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )

    try:
        from accelerate import init_empty_weights
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'accelerate' required for --load-mode empty.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)


def _module_type_name(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}"


def _linear_shape(module: Any) -> Optional[tuple[int, int]]:
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return None

    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.out_features), int(module.in_features)

    cls_name = type(module).__name__
    if cls_name == "Conv1D":
        return int(weight.shape[1]), int(weight.shape[0])

    if "Linear" in cls_name:
        return int(weight.shape[0]), int(weight.shape[1])

    return None


def _rmsnorm_hidden_size(module: Any) -> Optional[int]:
    cls_name = type(module).__name__.lower()
    if "rmsnorm" not in cls_name and "rms_norm" not in cls_name:
        return None
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 1:
        return None
    return int(weight.shape[0])


def _summarize_definition(
    def_name: str,
    op_type: str,
    occurrences: int,
    module_names: List[str],
    trace_set: Any,
    max_atol: float,
    max_rtol: float,
    include_solution_stats: bool,
    max_solutions_per_definition: int,
) -> Dict[str, Any]:
    from flashinfer_bench.apply.trace_filter import is_baseline_solution

    summary: Dict[str, Any] = {
        "definition": def_name,
        "op_type": op_type,
        "occurrences": occurrences,
        "module_names": module_names,
    }
    if trace_set is None:
        return summary

    definition = trace_set.definitions.get(def_name)
    solutions = list(trace_set.solutions.get(def_name, []))
    eligible_traces = trace_set.filter_traces(def_name, atol=max_atol, rtol=max_rtol)
    best_trace = trace_set.get_best_trace(
        def_name,
        max_abs_error=max_atol,
        max_rel_error=max_rtol,
    )
    generated_traces = [
        trace
        for trace in eligible_traces
        if trace.solution
        and (solution := trace_set.get_solution(trace.solution)) is not None
        and not is_baseline_solution(solution)
    ]
    baseline_traces = [
        trace
        for trace in eligible_traces
        if trace.solution
        and (solution := trace_set.get_solution(trace.solution)) is not None
        and is_baseline_solution(solution)
    ]
    a800_targeted = [
        sol.name for sol in solutions if any("A800" in hw.upper() for hw in sol.spec.target_hardware)
    ]

    summary.update(
        {
            "definition_present": definition is not None,
            "solution_count": len(solutions),
            "eligible_trace_count": len(eligible_traces),
            "best_trace_solution": best_trace.solution if best_trace else "",
            "best_trace_hardware": (
                best_trace.evaluation.environment.hardware
                if best_trace and best_trace.evaluation is not None
                else ""
            ),
            "best_trace_speedup_factor": (
                best_trace.evaluation.performance.speedup_factor
                if best_trace and best_trace.evaluation is not None and best_trace.evaluation.performance
                else 0.0
            ),
            "best_solution_target_hardware": (
                trace_set.get_solution(best_trace.solution).spec.target_hardware
                if best_trace and best_trace.solution and trace_set.get_solution(best_trace.solution)
                else []
            ),
            "best_trace_by_pool": {
                "all": _best_trace_summary(eligible_traces, trace_set),
                "generated_only": _best_trace_summary(generated_traces, trace_set),
                "baseline_only": _best_trace_summary(baseline_traces, trace_set),
            },
            "a800_targeted_solution_count": len(a800_targeted),
            "a800_targeted_solutions": a800_targeted,
        }
    )
    if include_solution_stats:
        gemm_dims = _parse_gemm_dimensions(def_name) if op_type == "gemm" else None
        solution_summaries: List[Dict[str, Any]] = []
        for solution in solutions:
            solution_traces = [
                trace for trace in eligible_traces if trace.solution == solution.name and trace.evaluation
            ]
            if not solution_traces:
                continue
            latencies = [trace.evaluation.performance.latency_ms for trace in solution_traces]
            reference_latencies = [
                trace.evaluation.performance.reference_latency_ms for trace in solution_traces
            ]
            speedups = [trace.evaluation.performance.speedup_factor for trace in solution_traces]
            best_solution_trace = max(
                solution_traces,
                key=lambda trace: trace.evaluation.performance.speedup_factor,
            )
            solution_summary = {
                "solution": solution.name,
                "author": solution.author,
                "is_baseline": is_baseline_solution(solution),
                "target_hardware": list(solution.spec.target_hardware),
                "eligible_trace_count": len(solution_traces),
                "latency_ms_min": min(latencies),
                "latency_ms_avg": _safe_mean(latencies),
                "latency_ms_max": max(latencies),
                "reference_latency_ms_min": min(reference_latencies),
                "reference_latency_ms_avg": _safe_mean(reference_latencies),
                "reference_latency_ms_max": max(reference_latencies),
                "speedup_min": min(speedups),
                "speedup_avg": _safe_mean(speedups),
                "speedup_max": max(speedups),
                "best_trace_axes": dict(best_solution_trace.workload.axes),
                "best_trace_hardware": best_solution_trace.evaluation.environment.hardware,
            }
            if gemm_dims is not None:
                n_dim, k_dim = gemm_dims
                solution_tflops: List[float] = []
                reference_tflops: List[float] = []
                for trace in solution_traces:
                    m_dim = _axis_as_int(trace.workload.axes, "M")
                    if m_dim is None:
                        continue
                    perf = trace.evaluation.performance
                    sol_tflops = _estimate_gemm_tflops(m_dim, n_dim, k_dim, perf.latency_ms)
                    ref_tflops = _estimate_gemm_tflops(
                        m_dim,
                        n_dim,
                        k_dim,
                        perf.reference_latency_ms,
                    )
                    if sol_tflops is not None:
                        solution_tflops.append(sol_tflops)
                    if ref_tflops is not None:
                        reference_tflops.append(ref_tflops)
                if solution_tflops:
                    solution_summary.update(
                        {
                            "estimated_tflops_min": min(solution_tflops),
                            "estimated_tflops_avg": _safe_mean(solution_tflops),
                            "estimated_tflops_max": max(solution_tflops),
                        }
                    )
                if reference_tflops:
                    solution_summary.update(
                        {
                            "reference_estimated_tflops_min": min(reference_tflops),
                            "reference_estimated_tflops_avg": _safe_mean(reference_tflops),
                            "reference_estimated_tflops_max": max(reference_tflops),
                        }
                    )
            solution_summaries.append(solution_summary)
        solution_summaries.sort(key=lambda item: item["speedup_max"], reverse=True)
        summary["solution_stats"] = solution_summaries[: max(0, max_solutions_per_definition)]
    return summary


def inspect_model(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model_skeleton(args)

    trace_set = None
    if args.trace_set_path:
        from flashinfer_bench.data import TraceSet
        from flashinfer_bench.apply.trace_filter import build_filtered_trace_set

        trace_set = TraceSet.from_path(str(Path(args.trace_set_path).resolve()))
        hardware_filters = [value.strip() for value in args.trace_hardware_contains if value.strip()]
        if hardware_filters:
            trace_set = build_filtered_trace_set(
                trace_set,
                definition_names=list(trace_set.definitions.keys()),
                trace_hardware_filters=hardware_filters,
            )

    module_type_counts: Counter[str] = Counter()
    all_modules: List[Dict[str, str]] = []
    gemm_counts: Counter[str] = Counter()
    gemm_modules: Dict[str, List[str]] = defaultdict(list)
    rmsnorm_counts: Counter[str] = Counter()
    rmsnorm_modules: Dict[str, List[str]] = defaultdict(list)

    for name, module in model.named_modules():
        if not name:
            continue
        module_type = _module_type_name(module)
        module_type_counts[module_type] += 1
        if args.include_modules:
            all_modules.append({"name": name, "type": module_type})

        linear_shape = _linear_shape(module)
        if linear_shape is not None:
            out_features, in_features = linear_shape
            def_name = f"gemm_n{out_features}_k{in_features}"
            gemm_counts[def_name] += 1
            if len(gemm_modules[def_name]) < args.sample_module_names:
                gemm_modules[def_name].append(name)

        hidden_size = _rmsnorm_hidden_size(module)
        if hidden_size is not None:
            def_name = f"rmsnorm_h{hidden_size}"
            rmsnorm_counts[def_name] += 1
            if len(rmsnorm_modules[def_name]) < args.sample_module_names:
                rmsnorm_modules[def_name].append(name)

    gemm_summaries = [
        _summarize_definition(
            def_name,
            "gemm",
            gemm_counts[def_name],
            gemm_modules[def_name],
            trace_set,
            args.max_atol,
            args.max_rtol,
            args.include_solution_stats,
            args.max_solutions_per_definition,
        )
        for def_name in sorted(gemm_modules)
    ]
    rmsnorm_summaries = [
        _summarize_definition(
            def_name,
            "rmsnorm",
            rmsnorm_counts[def_name],
            rmsnorm_modules[def_name],
            trace_set,
            args.max_atol,
            args.max_rtol,
            args.include_solution_stats,
            args.max_solutions_per_definition,
        )
        for def_name in sorted(rmsnorm_modules)
    ]

    summary: Dict[str, Any] = {
        "model": args.model,
        "load_mode": args.load_mode,
        "trace_set_path": str(Path(args.trace_set_path).resolve()) if args.trace_set_path else "",
        "trace_hardware_filters": [value.strip() for value in args.trace_hardware_contains if value.strip()],
        "gemm_definitions": gemm_summaries,
        "rmsnorm_definitions": rmsnorm_summaries,
    }
    if not args.summary_only:
        summary["module_type_counts"] = dict(module_type_counts.most_common())
    if args.include_modules:
        summary["modules"] = all_modules
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = inspect_model(args)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[output] wrote inspection summary to {output_path}")


if __name__ == "__main__":
    main()
