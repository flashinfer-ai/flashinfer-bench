"""Enable FlashInfer-Bench apply before handing control to another Python entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashinfer_bench.apply import ApplyConfig, disable_apply, enable_apply
from flashinfer_bench.apply.trace_filter import (
    build_filtered_trace_set,
    collect_solution_names_by_pool,
    count_eligible_traces_by_solution,
)
from flashinfer_bench.data import TraceSet


def _trace_count(trace_set: TraceSet) -> int:
    return sum(len(traces) for traces in trace_set.traces.values())


def _collect_selected_solutions(dispatch_stats: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    if not dispatch_stats:
        return []

    aggregated: Dict[str, int] = {}
    for definition_bucket in dispatch_stats.get("definitions", []):
        for solution_name, count in definition_bucket.get("selected_solutions", {}).items():
            aggregated[solution_name] = aggregated.get(solution_name, 0) + int(count)

    return [
        {"solution": solution_name, "calls": count}
        for solution_name, count in sorted(
            aggregated.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _collect_replaced_definitions(dispatch_stats: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    if not dispatch_stats:
        return []

    replaced = []
    for definition_bucket in dispatch_stats.get("definitions", []):
        if not definition_bucket.get("selected_solutions"):
            continue
        replaced.append(
            {
                "definition": definition_bucket["definition"],
                "total_calls": definition_bucket["total_calls"],
                "table_hit_calls": definition_bucket["table_hit_calls"],
                "def_best_calls": definition_bucket["def_best_calls"],
                "fallback_calls": definition_bucket["fallback_calls"],
                "selected_solutions": definition_bucket["selected_solutions"],
            }
        )
    return replaced


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start a target Python module or script after enabling FlashInfer-Bench apply "
            "in the same process."
        )
    )
    parser.add_argument("--trace-set-path", required=True, help="Trace-set root path")
    parser.add_argument(
        "--adapter-scope",
        choices=["all", "gemm_only", "rmsnorm_only", "attention_only"],
        default="all",
        help="Which integration adapters to install before launching the target process.",
    )
    parser.add_argument(
        "--trace-hardware-contains",
        action="append",
        default=[],
        help=(
            "Retain only traces whose evaluation.environment.hardware contains this substring. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--solution-language",
        action="append",
        default=[],
        help=(
            "Retain only solutions whose spec.language matches this token, e.g. cuda or triton. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--only-definition",
        action="append",
        default=[],
        help="Restrict apply to the given definition name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pin-solution",
        action="append",
        default=[],
        help=(
            "Force apply to consider only the given solution name(s). "
            "Pinned solutions are registered with use_def_best."
        ),
    )
    parser.add_argument(
        "--solution-pool",
        choices=["all", "generated_only", "baseline_only"],
        default="all",
        help="Which solution pool apply may choose from when not pinning a specific solution",
    )
    parser.add_argument(
        "--on-miss-policy",
        choices=["fallback_only", "use_def_best"],
        default="fallback_only",
    )
    parser.add_argument("--max-atol", type=float, default=1e-2)
    parser.add_argument("--max-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--aot-ratio",
        type=float,
        default=0.0,
        help="Fraction of top solutions to AOT-build per definition before launching the target",
    )
    parser.add_argument("--fib-cache-path", help="Optional override for FIB_CACHE_PATH")
    parser.add_argument(
        "--flashinfer-workspace-base",
        help=(
            "Optional override for FLASHINFER_WORKSPACE_BASE. Use this to isolate FlashInfer "
            "JIT cache from other Python environments."
        ),
    )
    parser.add_argument(
        "--apply-summary-json",
        help="Optional path to write apply dispatch summary JSON after the target exits",
    )
    parser.add_argument("--chdir", help="Optional working directory before launching the target")
    parser.add_argument(
        "--prepend-pythonpath",
        action="append",
        default=[],
        help="Additional path prepended to sys.path before launching the target",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module", help="Python module executed like `python -m module`")
    target.add_argument("--script", help="Python script path executed like `python script.py`")

    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target. Prefix with `--` to stop launcher parsing.",
    )
    return parser


def _normalize_target_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _prepare_environment(args: argparse.Namespace) -> str:
    trace_root = str(Path(args.trace_set_path).resolve())
    os.environ["FIB_ENABLE_APPLY"] = "1"
    os.environ["FIB_DATASET_PATH"] = trace_root
    os.environ["FIB_APPLY_ADAPTER_SCOPE"] = args.adapter_scope
    if args.fib_cache_path:
        os.environ["FIB_CACHE_PATH"] = str(Path(args.fib_cache_path).resolve())
    if args.flashinfer_workspace_base:
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(Path(args.flashinfer_workspace_base).resolve())

    if args.chdir:
        os.chdir(Path(args.chdir).resolve())

    for path in reversed(args.prepend_pythonpath):
        sys.path.insert(0, str(Path(path).resolve()))

    return trace_root


def _build_apply_target(
    args: argparse.Namespace,
) -> tuple[str | TraceSet | None, ApplyConfig, str]:
    trace_root = str(Path(args.trace_set_path).resolve())
    config = ApplyConfig(
        max_atol=args.max_atol,
        max_rtol=args.max_rtol,
        aot_ratio=args.aot_ratio,
        on_miss_policy=args.on_miss_policy,
    )
    trace_hardware_filters = [value.strip() for value in args.trace_hardware_contains if value.strip()]
    solution_language_filters = [value.strip() for value in args.solution_language if value.strip()]
    if not (
        args.pin_solution
        or args.only_definition
        or args.solution_pool != "all"
        or trace_hardware_filters
        or solution_language_filters
    ):
        return trace_root, config, ""

    trace_set = TraceSet.from_path(trace_root)
    if trace_hardware_filters:
        trace_set = build_filtered_trace_set(
            trace_set,
            definition_names=list(trace_set.definitions.keys()),
            trace_hardware_filters=trace_hardware_filters,
            solution_language_filters=solution_language_filters,
        )
    if args.pin_solution:
        eligible_counts = count_eligible_traces_by_solution(
            trace_set,
            solution_names=args.pin_solution,
            max_atol=args.max_atol,
            max_rtol=args.max_rtol,
            trace_hardware_filters=trace_hardware_filters,
        )
        ineligible = [name for name, count in eligible_counts.items() if count == 0]
        if ineligible:
            raise ValueError(
                "Pinned solution(s) have no PASSED traces under the requested tolerances: "
                f"{sorted(ineligible)}"
            )
        trace_set = build_filtered_trace_set(
            trace_set,
            solution_names=args.pin_solution,
            definition_names=args.only_definition or None,
            trace_hardware_filters=trace_hardware_filters,
            solution_language_filters=solution_language_filters,
        )
        return trace_set, config.model_copy(update={"on_miss_policy": "use_def_best"}), ""

    if args.solution_pool != "all":
        pool_solutions = collect_solution_names_by_pool(
            trace_set,
            pool=args.solution_pool,
            definition_names=args.only_definition or None,
            solution_language_filters=solution_language_filters,
        )
        if not pool_solutions:
            print(
                "[apply] no solutions found in pool "
                f"'{args.solution_pool}' for definitions "
                f"{sorted(set(args.only_definition)) if args.only_definition else 'ALL'}; "
                "skipping apply"
            )
            return None, config, "no_solutions_in_requested_pool"
        trace_set = build_filtered_trace_set(
            trace_set,
            solution_names=pool_solutions,
            definition_names=args.only_definition or None,
            trace_hardware_filters=trace_hardware_filters,
            solution_language_filters=solution_language_filters,
        )
        if _trace_count(trace_set) == 0:
            print("[apply] no traces matched the requested filters; skipping apply")
            return None, config, "no_traces_matched_filters"
        return trace_set, config, ""

    trace_set = build_filtered_trace_set(
        trace_set,
        definition_names=args.only_definition,
        trace_hardware_filters=trace_hardware_filters,
        solution_language_filters=solution_language_filters,
    )
    if _trace_count(trace_set) == 0:
        print("[apply] no traces matched the requested filters; skipping apply")
        return None, config, "no_traces_matched_filters"
    return trace_set, config, ""


def _run_target(args: argparse.Namespace, target_args: list[str]) -> None:
    if args.module:
        sys.argv = [args.module, *target_args]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        return

    script_path = str(Path(args.script).resolve())
    sys.argv = [script_path, *target_args]
    runpy.run_path(script_path, run_name="__main__")


def main() -> None:
    args = build_parser().parse_args()
    target_args = _normalize_target_args(args.target_args)
    trace_root = _prepare_environment(args)
    apply_target, config, skip_reason = _build_apply_target(args)
    target_desc = args.module if args.module else args.script
    runtime = None
    if apply_target is not None:
        runtime = enable_apply(apply_target, config)
        print(f"[apply] enabled with trace set: {trace_root}")
        if args.only_definition:
            print(f"[apply] restricted definitions: {sorted(set(args.only_definition))}")
        if args.trace_hardware_contains:
            print(
                "[apply] trace hardware filters: "
                f"{[value.strip() for value in args.trace_hardware_contains if value.strip()]}"
            )
        if args.solution_language:
            print(
                "[apply] solution language filters: "
                f"{[value.strip() for value in args.solution_language if value.strip()]}"
            )
        if args.adapter_scope != "all":
            print(f"[apply] adapter scope: {args.adapter_scope}")
        if args.solution_pool != "all" and not args.pin_solution:
            print(f"[apply] solution pool: {args.solution_pool}")
        if args.pin_solution:
            print(f"[apply] pinned solutions: {sorted(set(args.pin_solution))}")
    print(f"[apply] launching target: {target_desc}")

    try:
        _run_target(args, target_args)
    finally:
        if args.apply_summary_json:
            dispatch_stats = runtime.snapshot_stats() if runtime is not None else None
            summary_payload = {
                "requested": True,
                "active": runtime is not None,
                "status": "enabled" if runtime is not None else "skipped",
                "skip_reason": skip_reason,
                "trace_set_path": trace_root,
                "adapter_scope": args.adapter_scope,
                "trace_hardware_filters": [value.strip() for value in args.trace_hardware_contains if value.strip()],
                "solution_language_filters": [value.strip() for value in args.solution_language if value.strip()],
                "dispatch_stats": dispatch_stats,
                "selected_solutions": _collect_selected_solutions(dispatch_stats),
                "replaced_definitions": _collect_replaced_definitions(dispatch_stats),
            }
            output_path = Path(args.apply_summary_json).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[apply] wrote summary to {output_path}")
        if runtime is not None:
            disable_apply()


if __name__ == "__main__":
    main()
