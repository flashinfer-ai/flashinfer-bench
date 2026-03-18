"""Run a mode matrix for the Transformers E2E benchmark and aggregate the results."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

from _compare_utils import (
    load_json_file,
    render_templates,
    run_command,
    slugify,
    summarize_named_counts,
    write_csv,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAUNCHER = SCRIPT_DIR / "transformers_generate_benchmark.py"
MODE_CHOICES = ["torch", "baseline_only", "generated_only", "pin_solution", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple apply modes against transformers_generate_benchmark.py and "
            "write a CSV/JSON summary."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory for logs, JSON, and CSV outputs")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess runs")
    parser.add_argument(
        "--launcher",
        default=str(DEFAULT_LAUNCHER),
        help="Path to transformers_generate_benchmark.py",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODE_CHOICES,
        default=["torch", "baseline_only", "generated_only", "pin_solution"],
        help="Mode matrix to execute",
    )
    parser.add_argument("--trace-set-path", help="Trace-set root required for apply-enabled modes")
    parser.add_argument(
        "--trace-hardware-contains",
        action="append",
        default=[],
        help="Apply-mode trace hardware substring filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--solution-language",
        action="append",
        default=[],
        help="Apply-mode solution language filter, e.g. cuda or triton. Can be passed multiple times.",
    )
    parser.add_argument(
        "--only-definition",
        action="append",
        default=[],
        help="Restrict apply modes to the given definition name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pin-solution",
        help="Required when `pin_solution` is present in --modes",
    )
    parser.add_argument(
        "--apply-scope",
        choices=["gemm_only", "all"],
        default="gemm_only",
        help="Forwarded to transformers_generate_benchmark.py for apply-enabled modes",
    )
    parser.add_argument(
        "--on-miss-policy",
        choices=["fallback_only", "use_def_best"],
        default="fallback_only",
    )
    parser.add_argument("--max-atol", type=float, default=1e-2)
    parser.add_argument("--max-rtol", type=float, default=1e-5)
    parser.add_argument("--aot-ratio", type=float, default=0.0)
    parser.add_argument("--fib-cache-path", help="Optional override for FIB_CACHE_PATH")
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to transformers_generate_benchmark.py. Prefix with `--`.",
    )
    return parser


def _normalize_forwarded_args(values: List[str]) -> List[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _validate_args(args: argparse.Namespace, forwarded_args: List[str]) -> None:
    reserved_flags = {
        "--enable-apply",
        "--trace-set-path",
        "--trace-hardware-contains",
        "--solution-language",
        "--only-definition",
        "--pin-solution",
        "--solution-pool",
        "--apply-scope",
        "--on-miss-policy",
        "--max-atol",
        "--max-rtol",
        "--aot-ratio",
        "--fib-cache-path",
        "--output-json",
    }
    conflicts = [flag for flag in forwarded_args if flag in reserved_flags]
    if conflicts:
        raise ValueError(
            "benchmark_args already contain compare-managed flags: "
            f"{sorted(set(conflicts))}"
        )

    apply_modes = [mode for mode in args.modes if mode != "torch"]
    if apply_modes and not args.trace_set_path:
        raise ValueError("--trace-set-path is required when apply-enabled modes are requested")
    if "pin_solution" in args.modes and not args.pin_solution:
        raise ValueError("--pin-solution is required when `pin_solution` is present in --modes")
    if not forwarded_args:
        raise ValueError("benchmark_args are required after `--`")


def _mode_dir_name(mode: str, pin_solution: str | None) -> str:
    if mode != "pin_solution":
        return mode
    return f"pin_solution_{slugify(pin_solution or 'solution')}"


def _build_apply_args(args: argparse.Namespace, mode: str) -> List[str]:
    if mode == "torch":
        return []

    apply_args = [
        "--enable-apply",
        "--trace-set-path",
        str(Path(args.trace_set_path).resolve()),
        "--apply-scope",
        args.apply_scope,
        "--on-miss-policy",
        args.on_miss_policy,
        "--max-atol",
        str(args.max_atol),
        "--max-rtol",
        str(args.max_rtol),
        "--aot-ratio",
        str(args.aot_ratio),
    ]
    if args.fib_cache_path:
        apply_args.extend(["--fib-cache-path", str(Path(args.fib_cache_path).resolve())])
    for value in args.trace_hardware_contains:
        if value.strip():
            apply_args.extend(["--trace-hardware-contains", value.strip()])
    for value in args.solution_language:
        if value.strip():
            apply_args.extend(["--solution-language", value.strip()])
    for definition in args.only_definition:
        if definition.strip():
            apply_args.extend(["--only-definition", definition.strip()])

    if mode == "baseline_only":
        apply_args.extend(["--solution-pool", "baseline_only"])
    elif mode == "generated_only":
        apply_args.extend(["--solution-pool", "generated_only"])
    elif mode == "pin_solution":
        apply_args.extend(["--pin-solution", args.pin_solution])
    elif mode == "all":
        apply_args.extend(["--solution-pool", "all"])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return apply_args


def _summarize_mode(
    mode: str,
    run_info: Dict[str, Any],
    *,
    result_json_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    command: List[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "mode": mode,
        "status": run_info["status"],
        "exit_code": run_info["exit_code"],
        "duration_s": run_info["duration_s"],
        "command": shlex.join(command),
        "result_json": str(result_json_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if not result_json_path.exists():
        return row

    summary = load_json_file(result_json_path)
    apply_status = summary.get("apply_status", "")
    apply_skip_reason = summary.get("apply_skip_reason", "")
    comparison_valid = mode == "torch" or apply_status == "enabled"
    row.update(
        {
            "apply_requested": summary.get("apply_requested"),
            "apply_active": summary.get("apply_active"),
            "apply_status": apply_status,
            "apply_skip_reason": apply_skip_reason,
            "comparison_valid": comparison_valid,
            "latency_ms_avg": summary.get("latency_ms_avg"),
            "latency_ms_p50": summary.get("latency_ms_p50"),
            "latency_ms_p90": summary.get("latency_ms_p90"),
            "latency_ms_min": summary.get("latency_ms_min"),
            "latency_ms_max": summary.get("latency_ms_max"),
            "generated_tokens_per_second": summary.get("generated_tokens_per_second"),
            "prompt_tokens_per_second": summary.get("prompt_tokens_per_second"),
            "selected_solutions": summarize_named_counts(
                summary.get("apply_selected_solutions", []),
                key_name="solution",
            ),
            "replaced_definitions": summarize_named_counts(
                summary.get("apply_replaced_definitions", []),
                key_name="definition",
            ),
        }
    )
    if mode == "torch":
        row["effective_mode"] = mode
        row["comparison_note"] = "pure torch baseline"
    elif apply_status == "skipped":
        row["effective_mode"] = "torch_equivalent"
        row["comparison_note"] = (
            "apply was requested but skipped; this is not a valid routed-apply comparison"
        )
    else:
        row["effective_mode"] = mode
        row["comparison_note"] = "apply enabled"
    return row


def main() -> None:
    args = build_parser().parse_args()
    forwarded_args = _normalize_forwarded_args(list(args.benchmark_args))
    _validate_args(args, forwarded_args)

    launcher_path = Path(args.launcher).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "launcher": str(launcher_path),
        "python": args.python,
        "modes": list(args.modes),
        "trace_set_path": str(Path(args.trace_set_path).resolve()) if args.trace_set_path else "",
        "trace_hardware_filters": [value.strip() for value in args.trace_hardware_contains if value.strip()],
        "only_definitions": [value.strip() for value in args.only_definition if value.strip()],
        "pin_solution": args.pin_solution or "",
        "apply_scope": args.apply_scope,
        "on_miss_policy": args.on_miss_policy,
        "max_atol": args.max_atol,
        "max_rtol": args.max_rtol,
        "aot_ratio": args.aot_ratio,
        "fib_cache_path": args.fib_cache_path or "",
        "forwarded_args": forwarded_args,
    }
    write_json(output_dir / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    total_modes = len(args.modes)
    for index, mode in enumerate(args.modes, start=1):
        run_name = _mode_dir_name(mode, args.pin_solution)
        run_dir = output_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        result_json_path = run_dir / "result.json"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        command = [args.python, str(launcher_path), *forwarded_args]
        command.extend(_build_apply_args(args, mode))
        command.extend(["--output-json", str(result_json_path)])

        print(f"[mode {index}/{total_modes}] starting {mode}")

        run_info = run_command(
            command,
            cwd=Path.cwd(),
            env=dict(os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        row = _summarize_mode(
            mode,
            run_info,
            result_json_path=result_json_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command,
        )
        rows.append(row)

        detailed_result = dict(row)
        if result_json_path.exists():
            detailed_result["summary"] = load_json_file(result_json_path)
        detailed_results.append(detailed_result)

        write_csv(output_dir / "summary.csv", rows)
        write_json(output_dir / "summary.json", detailed_results)
        print(
            f"[mode {index}/{total_modes}] finished {mode}: "
            f"status={row['status']} summary={output_dir / 'summary.csv'}"
        )

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", detailed_results)
    print(f"[output] wrote manifest to {output_dir / 'manifest.json'}")
    print(f"[output] wrote summary CSV to {output_dir / 'summary.csv'}")
    print(f"[output] wrote summary JSON to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
