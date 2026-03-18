"""Compare SGLang fused_add_rmsnorm end-to-end paths under bench_one_batch."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

from _compare_utils import load_json_artifact, load_json_file, run_command, write_csv, write_json

SCRIPT_DIR = Path(__file__).resolve().parent
APPLY_BOOTSTRAP = SCRIPT_DIR / "bootstrap_apply_runner.py"
FALLBACK_BOOTSTRAP = SCRIPT_DIR / "bootstrap_flashinfer_fallback_runner.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four SGLang bench_one_batch paths: original, fallback substitution, "
            "apply pinned to FlashInfer baseline, and apply pinned to the generated CUDA solution."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-set-path", required=True)
    parser.add_argument("--definition", default="fused_add_rmsnorm_h4096")
    parser.add_argument("--generated-solution", required=True)
    parser.add_argument("--baseline-solution", default="flashinfer_wrapper_0ff432")
    parser.add_argument("--trace-hardware-contains", default="A800")
    parser.add_argument("--flashinfer-workspace-base", default="/tmp/flashinfer_ws_fib_e2e")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--sampling-backend", default="pytorch")
    parser.add_argument("--max-atol", type=float, default=0.05)
    parser.add_argument("--max-rtol", type=float, default=0.01)
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)
    return parser


def _run_env(args: argparse.Namespace) -> Dict[str, str]:
    import os

    env = os.environ.copy()
    env["FLASHINFER_WORKSPACE_BASE"] = str(Path(args.flashinfer_workspace_base).resolve())
    return env


def _bench_args(args: argparse.Namespace, batch_size: int, result_path: Path) -> List[str]:
    out = [
        "--model-path",
        str(Path(args.model_path).resolve()),
        "--dtype",
        args.dtype,
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--attention-backend",
        args.attention_backend,
        "--prefill-attention-backend",
        args.attention_backend,
        "--decode-attention-backend",
        args.attention_backend,
        "--sampling-backend",
        args.sampling_backend,
        "--result-filename",
        str(result_path),
    ]
    if args.disable_cuda_graph:
        out.append("--disable-cuda-graph")
    return out


def _run_case(args: argparse.Namespace, *, batch_size: int, mode: str, output_dir: Path) -> Dict[str, Any]:
    run_dir = output_dir / f"bs{batch_size}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    result_path = run_dir / "result.jsonl"
    apply_summary_path = run_dir / "apply_summary.json"
    fallback_summary_path = run_dir / "fallback_summary.json"
    bench_args = _bench_args(args, batch_size, result_path)

    if mode == "original":
        command = [args.python, "-m", "sglang.bench_one_batch", *bench_args]
        apply_summary_path = None
        fallback_summary_path = None
    elif mode == "fallback_substitution":
        command = [
            args.python,
            str(FALLBACK_BOOTSTRAP),
            "--summary-json",
            str(fallback_summary_path),
            "--flashinfer-workspace-base",
            str(Path(args.flashinfer_workspace_base).resolve()),
            "--module",
            "sglang.bench_one_batch",
            "--",
            *bench_args,
        ]
        apply_summary_path = None
    elif mode == "apply_flashinfer_wrapper":
        command = [
            args.python,
            str(APPLY_BOOTSTRAP),
            "--trace-set-path",
            str(Path(args.trace_set_path).resolve()),
            "--flashinfer-workspace-base",
            str(Path(args.flashinfer_workspace_base).resolve()),
            "--adapter-scope",
            "rmsnorm_only",
            "--only-definition",
            args.definition,
            "--pin-solution",
            args.baseline_solution,
            "--apply-summary-json",
            str(apply_summary_path),
            "--module",
            "sglang.bench_one_batch",
            "--",
            *bench_args,
        ]
        fallback_summary_path = None
    elif mode == "apply_generated_solution":
        command = [
            args.python,
            str(APPLY_BOOTSTRAP),
            "--trace-set-path",
            str(Path(args.trace_set_path).resolve()),
            "--flashinfer-workspace-base",
            str(Path(args.flashinfer_workspace_base).resolve()),
            "--trace-hardware-contains",
            args.trace_hardware_contains,
            "--solution-language",
            "cuda",
            "--adapter-scope",
            "rmsnorm_only",
            "--only-definition",
            args.definition,
            "--pin-solution",
            args.generated_solution,
            "--max-atol",
            str(args.max_atol),
            "--max-rtol",
            str(args.max_rtol),
            "--apply-summary-json",
            str(apply_summary_path),
            "--module",
            "sglang.bench_one_batch",
            "--",
            *bench_args,
        ]
        fallback_summary_path = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    run_info = run_command(
        command,
        cwd=SCRIPT_DIR,
        env=_run_env(args),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    row: Dict[str, Any] = {
        "batch_size": batch_size,
        "mode": mode,
        "status": run_info["status"],
        "exit_code": run_info["exit_code"],
        "duration_s": run_info["duration_s"],
        "command": shlex.join(command),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "result_path": str(result_path),
        "apply_summary_json": str(apply_summary_path) if apply_summary_path else "",
        "fallback_summary_json": str(fallback_summary_path) if fallback_summary_path else "",
    }
    if result_path.exists():
        payload = load_json_artifact(result_path, fmt="jsonl") or {}
        row.update(
            {
                "prefill_latency_s": payload.get("prefill_latency"),
                "total_latency_s": payload.get("total_latency"),
                "prefill_throughput": payload.get("prefill_throughput"),
                "overall_throughput": payload.get("overall_throughput"),
                "input_len": payload.get("input_len"),
                "output_len": payload.get("output_len"),
            }
        )
    if apply_summary_path and apply_summary_path.exists():
        payload = load_json_file(apply_summary_path)
        stats = payload.get("dispatch_stats") or {}
        row["selected_solutions"] = ";".join(
            f"{item['solution']}:{item['calls']}" for item in payload.get("selected_solutions", [])
        )
        row["table_hit_calls"] = stats.get("table_hit_calls")
        row["def_best_calls"] = stats.get("def_best_calls")
        row["fallback_calls"] = stats.get("fallback_calls")
        definitions = stats.get("definitions") or []
        if definitions:
            row["observed_axes"] = ";".join(
                f"{item['axes']}:{item['calls']}" for item in definitions[0].get("observed_axes", [])
            )
    if fallback_summary_path and fallback_summary_path.exists():
        payload = load_json_file(fallback_summary_path)
        row["fallback_mode"] = payload.get("mode")
        row["replacement_entrypoint"] = payload.get("replacement_entrypoint")
    return row


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        "original",
        "fallback_substitution",
        "apply_flashinfer_wrapper",
        "apply_generated_solution",
    ]
    rows: List[Dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for mode in modes:
            rows.append(_run_case(args, batch_size=batch_size, mode=mode, output_dir=output_dir))

    manifest = {
        "model_path": str(Path(args.model_path).resolve()),
        "trace_set_path": str(Path(args.trace_set_path).resolve()),
        "definition": args.definition,
        "generated_solution": args.generated_solution,
        "baseline_solution": args.baseline_solution,
        "trace_hardware_contains": args.trace_hardware_contains,
        "flashinfer_workspace_base": str(Path(args.flashinfer_workspace_base).resolve()),
        "batch_sizes": args.batch_sizes,
        "attention_backend": args.attention_backend,
        "modes": modes,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)


if __name__ == "__main__":
    main()
