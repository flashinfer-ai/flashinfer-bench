"""Run a mode matrix for bootstrap_apply_runner.py and aggregate results."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

from _compare_utils import (
    flatten_scalars,
    load_json_artifact,
    load_json_file,
    render_template,
    render_templates,
    run_command,
    slugify,
    write_csv,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOOTSTRAP = SCRIPT_DIR / "bootstrap_apply_runner.py"
MODE_CHOICES = ["torch", "baseline_only", "generated_only", "pin_solution", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple apply modes against a Python module/script via bootstrap_apply_runner.py "
            "and write a CSV/JSON summary."
        )
    )
    parser.add_argument("--output-dir", required=True, help="Directory for logs, artifacts, and summaries")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess runs")
    parser.add_argument(
        "--bootstrap-launcher",
        default=str(DEFAULT_BOOTSTRAP),
        help="Path to bootstrap_apply_runner.py",
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
        "--adapter-scope",
        choices=["all", "gemm_only", "rmsnorm_only", "attention_only"],
        default="all",
        help="Which integration adapters the bootstrap runner should install.",
    )
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
        "--on-miss-policy",
        choices=["fallback_only", "use_def_best"],
        default="fallback_only",
    )
    parser.add_argument("--max-atol", type=float, default=1e-2)
    parser.add_argument("--max-rtol", type=float, default=1e-5)
    parser.add_argument("--aot-ratio", type=float, default=0.0)
    parser.add_argument("--fib-cache-path", help="Optional override for FIB_CACHE_PATH")
    parser.add_argument("--chdir", help="Optional working directory before launching the target")
    parser.add_argument(
        "--prepend-pythonpath",
        action="append",
        default=[],
        help="Additional path prepended to sys.path before launching the target",
    )
    parser.add_argument(
        "--target-result-path",
        help=(
            "Optional JSON/JSONL artifact path produced by the target benchmark. "
            "Supports {mode}, {mode_output_dir}, and {output_dir} placeholders."
        ),
    )
    parser.add_argument(
        "--target-result-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module", help="Python module executed like `python -m module`")
    target.add_argument("--script", help="Python script path executed like `python script.py`")

    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target. Prefix with `--`.",
    )
    return parser


def _normalize_target_args(values: List[str]) -> List[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _validate_args(args: argparse.Namespace) -> None:
    apply_modes = [mode for mode in args.modes if mode != "torch"]
    if apply_modes and not args.trace_set_path:
        raise ValueError("--trace-set-path is required when apply-enabled modes are requested")
    if "pin_solution" in args.modes and not args.pin_solution:
        raise ValueError("--pin-solution is required when `pin_solution` is present in --modes")


def _mode_dir_name(mode: str, pin_solution: str | None) -> str:
    if mode != "pin_solution":
        return mode
    return f"pin_solution_{slugify(pin_solution or 'solution')}"


def _target_spec(args: argparse.Namespace) -> List[str]:
    if args.module:
        return ["--module", args.module]
    return ["--script", str(Path(args.script).resolve())]


def _build_direct_target_command(
    args: argparse.Namespace,
    rendered_target_args: List[str],
) -> List[str]:
    if args.module:
        return [args.python, "-m", args.module, *rendered_target_args]
    return [args.python, str(Path(args.script).resolve()), *rendered_target_args]


def _build_bootstrap_command(
    args: argparse.Namespace,
    *,
    mode: str,
    rendered_target_args: List[str],
    apply_summary_path: Path,
) -> List[str]:
    command = [
        args.python,
        str(Path(args.bootstrap_launcher).resolve()),
        "--trace-set-path",
        str(Path(args.trace_set_path).resolve()),
        "--adapter-scope",
        args.adapter_scope,
        "--on-miss-policy",
        args.on_miss_policy,
        "--max-atol",
        str(args.max_atol),
        "--max-rtol",
        str(args.max_rtol),
        "--aot-ratio",
        str(args.aot_ratio),
        "--apply-summary-json",
        str(apply_summary_path),
    ]
    if args.fib_cache_path:
        command.extend(["--fib-cache-path", str(Path(args.fib_cache_path).resolve())])
    if args.chdir:
        command.extend(["--chdir", str(Path(args.chdir).resolve())])
    for path in args.prepend_pythonpath:
        command.extend(["--prepend-pythonpath", str(Path(path).resolve())])
    for value in args.trace_hardware_contains:
        if value.strip():
            command.extend(["--trace-hardware-contains", value.strip()])
    for value in args.solution_language:
        if value.strip():
            command.extend(["--solution-language", value.strip()])
    for definition in args.only_definition:
        if definition.strip():
            command.extend(["--only-definition", definition.strip()])

    if mode == "baseline_only":
        command.extend(["--solution-pool", "baseline_only"])
    elif mode == "generated_only":
        command.extend(["--solution-pool", "generated_only"])
    elif mode == "pin_solution":
        command.extend(["--pin-solution", args.pin_solution])
    elif mode == "all":
        command.extend(["--solution-pool", "all"])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    command.extend(_target_spec(args))
    command.append("--")
    command.extend(rendered_target_args)
    return command


def _build_mode_mapping(output_dir: Path, run_dir: Path, mode: str) -> Dict[str, str]:
    return {
        "mode": mode,
        "output_dir": str(output_dir),
        "mode_output_dir": str(run_dir),
        "mode_slug": slugify(mode),
    }


def _summarize_mode(
    mode: str,
    run_info: Dict[str, Any],
    *,
    result_artifact_path: Path | None,
    result_artifact_format: str,
    apply_summary_path: Path | None,
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
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "target_result_path": str(result_artifact_path) if result_artifact_path else "",
        "target_result_format": result_artifact_format if result_artifact_path else "",
        "apply_summary_json": str(apply_summary_path) if apply_summary_path else "",
    }

    if apply_summary_path is not None and apply_summary_path.exists():
        apply_summary = load_json_file(apply_summary_path)
        row.update(
            {
                "apply_status": apply_summary.get("status", ""),
                "apply_skip_reason": apply_summary.get("skip_reason", ""),
                "apply_selected_solutions": ";".join(
                    f"{item.get('solution')}:{item.get('calls')}"
                    for item in apply_summary.get("selected_solutions", [])
                    if item.get("solution")
                ),
                "apply_replaced_definitions": ";".join(
                    str(item.get("definition"))
                    for item in apply_summary.get("replaced_definitions", [])
                    if item.get("definition")
                ),
            }
        )
        row.update(flatten_scalars(apply_summary.get("dispatch_stats"), prefix="apply", max_depth=2))

    if result_artifact_path is None:
        return row

    payload = load_json_artifact(result_artifact_path, fmt=result_artifact_format)
    row["target_result_found"] = bool(result_artifact_path.exists())
    if payload is None:
        return row

    row.update(flatten_scalars(payload, prefix="artifact", max_depth=2))
    return row


def main() -> None:
    args = build_parser().parse_args()
    args.target_args = _normalize_target_args(list(args.target_args))
    _validate_args(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path(args.chdir).resolve() if args.chdir else Path.cwd()

    manifest = {
        "bootstrap_launcher": str(Path(args.bootstrap_launcher).resolve()),
        "python": args.python,
        "modes": list(args.modes),
        "trace_set_path": str(Path(args.trace_set_path).resolve()) if args.trace_set_path else "",
        "adapter_scope": args.adapter_scope,
        "trace_hardware_filters": [value.strip() for value in args.trace_hardware_contains if value.strip()],
        "only_definitions": [value.strip() for value in args.only_definition if value.strip()],
        "pin_solution": args.pin_solution or "",
        "on_miss_policy": args.on_miss_policy,
        "max_atol": args.max_atol,
        "max_rtol": args.max_rtol,
        "aot_ratio": args.aot_ratio,
        "fib_cache_path": args.fib_cache_path or "",
        "chdir": str(cwd),
        "prepend_pythonpath": [str(Path(path).resolve()) for path in args.prepend_pythonpath],
        "target": {"module": args.module or "", "script": args.script or ""},
        "target_args": args.target_args,
        "target_result_path": args.target_result_path or "",
        "target_result_format": args.target_result_format,
    }
    write_json(output_dir / "manifest.json", manifest)

    rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    total_modes = len(args.modes)
    for index, mode in enumerate(args.modes, start=1):
        run_name = _mode_dir_name(mode, args.pin_solution)
        run_dir = output_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        mapping = _build_mode_mapping(output_dir, run_dir, mode)
        rendered_target_args = render_templates(args.target_args, mapping)
        result_artifact_path = (
            Path(render_template(args.target_result_path, mapping)).resolve()
            if args.target_result_path
            else None
        )
        apply_summary_path = run_dir / "apply_summary.json" if mode != "torch" else None
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        if mode == "torch":
            command = _build_direct_target_command(args, rendered_target_args)
        else:
            command = _build_bootstrap_command(
                args,
                mode=mode,
                rendered_target_args=rendered_target_args,
                apply_summary_path=apply_summary_path,
            )

        print(f"[mode {index}/{total_modes}] starting {mode}")

        env = dict(os.environ)
        if args.prepend_pythonpath:
            extra_paths = [str(Path(path).resolve()) for path in args.prepend_pythonpath]
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [*extra_paths, current_pythonpath] if current_pythonpath else extra_paths
            )

        run_info = run_command(
            command,
            cwd=cwd,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        row = _summarize_mode(
            mode,
            run_info,
            result_artifact_path=result_artifact_path,
            result_artifact_format=args.target_result_format,
            apply_summary_path=apply_summary_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command,
        )
        rows.append(row)

        detailed_result = dict(row)
        if apply_summary_path is not None and apply_summary_path.exists():
            detailed_result["apply_summary"] = load_json_file(apply_summary_path)
        if result_artifact_path is not None and result_artifact_path.exists():
            detailed_result["artifact_payload"] = load_json_artifact(
                result_artifact_path,
                fmt=args.target_result_format,
            )
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
