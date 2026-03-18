"""Run a mode matrix for server/client benchmarks and aggregate the results."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

from _compare_utils import (
    flatten_scalars,
    launch_command,
    load_json_artifact,
    load_json_file,
    render_template,
    render_templates,
    slugify,
    stop_process,
    wait_for_http_ready,
    write_csv,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOOTSTRAP = SCRIPT_DIR / "bootstrap_apply_runner.py"
MODE_CHOICES = ["torch", "baseline_only", "generated_only", "pin_solution", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a framework server under multiple apply modes, wait for it to become "
            "ready, run a client benchmark command, and aggregate results."
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
    parser.add_argument("--pin-solution", help="Required when `pin_solution` is present in --modes")
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
        help="Additional path prepended to sys.path and PYTHONPATH before launching processes",
    )
    parser.add_argument("--ready-url", required=True, help="HTTP URL polled until the server is ready")
    parser.add_argument(
        "--ready-substring",
        default="",
        help="Optional response-body substring required for readiness",
    )
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--ready-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--client-command",
        required=True,
        help=(
            "Shell-style client benchmark command template. Supports placeholders "
            "{mode}, {mode_slug}, {output_dir}, and {mode_output_dir}."
        ),
    )
    parser.add_argument(
        "--client-result-path",
        help=(
            "Optional JSON/JSONL artifact path produced by the client benchmark. "
            "Supports the same placeholders as --client-command."
        ),
    )
    parser.add_argument(
        "--client-result-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
    )

    server = parser.add_mutually_exclusive_group(required=True)
    server.add_argument("--server-module", help="Python module executed like `python -m module`")
    server.add_argument("--server-script", help="Python script path executed like `python script.py`")

    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the server target. Prefix with `--`.",
    )
    return parser


def _normalize_forwarded_args(values: List[str]) -> List[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _validate_args(args: argparse.Namespace) -> None:
    apply_modes = [mode for mode in args.modes if mode != "torch"]
    if apply_modes and not args.trace_set_path:
        raise ValueError("--trace-set-path is required when apply-enabled modes are requested")
    if "pin_solution" in args.modes and not args.pin_solution:
        raise ValueError("--pin-solution is required when `pin_solution` is present in --modes")
    if not args.server_args:
        raise ValueError("server_args are required after `--`")


def _mode_dir_name(mode: str, pin_solution: str | None) -> str:
    if mode != "pin_solution":
        return mode
    return f"pin_solution_{slugify(pin_solution or 'solution')}"


def _target_spec(args: argparse.Namespace) -> List[str]:
    if args.server_module:
        return ["--module", args.server_module]
    return ["--script", str(Path(args.server_script).resolve())]


def _build_direct_server_command(
    args: argparse.Namespace,
    rendered_server_args: List[str],
) -> List[str]:
    if args.server_module:
        return [args.python, "-m", args.server_module, *rendered_server_args]
    return [args.python, str(Path(args.server_script).resolve()), *rendered_server_args]


def _build_bootstrap_server_command(
    args: argparse.Namespace,
    *,
    mode: str,
    rendered_server_args: List[str],
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
    command.extend(rendered_server_args)
    return command


def _build_mode_mapping(output_dir: Path, run_dir: Path, mode: str) -> Dict[str, str]:
    return {
        "mode": mode,
        "output_dir": str(output_dir),
        "mode_output_dir": str(run_dir),
        "mode_slug": slugify(mode),
    }


def _apply_extra_pythonpath(env: Dict[str, str], args: argparse.Namespace) -> Dict[str, str]:
    if not args.prepend_pythonpath:
        return env
    extra_paths = [str(Path(path).resolve()) for path in args.prepend_pythonpath]
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [*extra_paths, current_pythonpath] if current_pythonpath else extra_paths
    )
    return env


def _ensure_local_no_proxy(env: Dict[str, str]) -> Dict[str, str]:
    no_proxy_hosts = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        current = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        merged = []
        for value in [*current, *no_proxy_hosts]:
            if value not in merged:
                merged.append(value)
        env[key] = ",".join(merged)
    return env


def _client_command(command_template: str) -> List[str]:
    return shlex.split(command_template)


def _summarize_mode(
    mode: str,
    *,
    server_command: List[str],
    client_command: List[str],
    ready_result: Dict[str, Any],
    client_run_info: Dict[str, Any] | None,
    client_result_path: Path | None,
    client_result_format: str,
    apply_summary_path: Path | None,
    server_stdout_path: Path,
    server_stderr_path: Path,
    client_stdout_path: Path,
    client_stderr_path: Path,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "mode": mode,
        "server_command": shlex.join(server_command),
        "client_command": shlex.join(client_command),
        "server_stdout_log": str(server_stdout_path),
        "server_stderr_log": str(server_stderr_path),
        "client_stdout_log": str(client_stdout_path),
        "client_stderr_log": str(client_stderr_path),
        "client_result_path": str(client_result_path) if client_result_path else "",
        "client_result_format": client_result_format if client_result_path else "",
        "ready_ok": ready_result.get("ok", False),
        "ready_attempts": ready_result.get("attempts"),
        "ready_error": ready_result.get("error", ""),
        "apply_summary_json": str(apply_summary_path) if apply_summary_path else "",
    }

    if client_run_info is not None:
        row.update(
            {
                "status": client_run_info["status"] if ready_result.get("ok") else "failed",
                "exit_code": client_run_info["exit_code"],
                "duration_s": client_run_info["duration_s"],
            }
        )
    else:
        row.update({"status": "failed", "exit_code": None, "duration_s": None})

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

    if client_result_path is None:
        return row

    payload = load_json_artifact(client_result_path, fmt=client_result_format)
    row["client_result_found"] = bool(client_result_path.exists())
    if payload is None:
        return row

    row.update(flatten_scalars(payload, prefix="artifact", max_depth=2))
    return row


def main() -> None:
    args = build_parser().parse_args()
    args.server_args = _normalize_forwarded_args(list(args.server_args))
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
        "ready_url": args.ready_url,
        "ready_substring": args.ready_substring,
        "ready_timeout_s": args.ready_timeout_s,
        "ready_interval_s": args.ready_interval_s,
        "server": {"module": args.server_module or "", "script": args.server_script or ""},
        "server_args": args.server_args,
        "client_command": args.client_command,
        "client_result_path": args.client_result_path or "",
        "client_result_format": args.client_result_format,
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
        rendered_server_args = render_templates(args.server_args, mapping)
        rendered_client_command = render_template(args.client_command, mapping)
        client_result_path = (
            Path(render_template(args.client_result_path, mapping)).resolve()
            if args.client_result_path
            else None
        )
        apply_summary_path = run_dir / "apply_summary.json" if mode != "torch" else None

        server_stdout_path = run_dir / "server_stdout.log"
        server_stderr_path = run_dir / "server_stderr.log"
        client_stdout_path = run_dir / "client_stdout.log"
        client_stderr_path = run_dir / "client_stderr.log"

        if mode == "torch":
            server_command = _build_direct_server_command(args, rendered_server_args)
        else:
            server_command = _build_bootstrap_server_command(
                args,
                mode=mode,
                rendered_server_args=rendered_server_args,
                apply_summary_path=apply_summary_path,
            )
        client_command = _client_command(rendered_client_command)

        print(f"[mode {index}/{total_modes}] starting {mode}")

        env = _ensure_local_no_proxy(_apply_extra_pythonpath(dict(os.environ), args))
        server_state = launch_command(
            server_command,
            cwd=cwd,
            env=env,
            stdout_path=server_stdout_path,
            stderr_path=server_stderr_path,
        )

        client_run_info: Dict[str, Any] | None = None
        ready_result = wait_for_http_ready(
            render_template(args.ready_url, mapping),
            timeout_s=args.ready_timeout_s,
            interval_s=args.ready_interval_s,
            required_substring=args.ready_substring,
            process=server_state["process"],
        )
        try:
            if ready_result.get("ok"):
                from _compare_utils import run_command

                client_run_info = run_command(
                    client_command,
                    cwd=cwd,
                    env=env,
                    stdout_path=client_stdout_path,
                    stderr_path=client_stderr_path,
                )
        finally:
            stop_process(server_state)

        row = _summarize_mode(
            mode,
            server_command=server_command,
            client_command=client_command,
            ready_result=ready_result,
            client_run_info=client_run_info,
            client_result_path=client_result_path,
            client_result_format=args.client_result_format,
            apply_summary_path=apply_summary_path,
            server_stdout_path=server_stdout_path,
            server_stderr_path=server_stderr_path,
            client_stdout_path=client_stdout_path,
            client_stderr_path=client_stderr_path,
        )
        rows.append(row)

        detailed_result = dict(row)
        if apply_summary_path is not None and apply_summary_path.exists():
            detailed_result["apply_summary"] = load_json_file(apply_summary_path)
        if client_result_path is not None and client_result_path.exists():
            detailed_result["artifact_payload"] = load_json_artifact(
                client_result_path,
                fmt=args.client_result_format,
            )
        detailed_results.append(detailed_result)

        write_csv(output_dir / "summary.csv", rows)
        write_json(output_dir / "summary.json", detailed_results)
        print(
            f"[mode {index}/{total_modes}] finished {mode}: "
            f"ready={row.get('ready_ok')} status={row.get('status', '')} "
            f"summary={output_dir / 'summary.csv'}"
        )

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", detailed_results)
    print(f"[output] wrote manifest to {output_dir / 'manifest.json'}")
    print(f"[output] wrote summary CSV to {output_dir / 'summary.csv'}")
    print(f"[output] wrote summary JSON to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
