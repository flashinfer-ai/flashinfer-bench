"""Probe SGLang's fused_add_rmsnorm entrypoint with optional FlashInfer-Bench apply."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Call sgl_kernel.fused_add_rmsnorm directly, validate correctness against a "
            "PyTorch reference, and report timing. This is useful for checking whether "
            "bootstrap_apply_runner actually replaces the intended SGLang RMSNorm kernel."
        )
    )
    parser.add_argument(
        "--entrypoint",
        choices=["sglang", "flashinfer", "generated"],
        default="sglang",
        help="Which fused_add_rmsnorm implementation to benchmark directly.",
    )
    parser.add_argument("--trace-set-path", help="Required when --entrypoint=generated")
    parser.add_argument("--definition", default="fused_add_rmsnorm_h4096")
    parser.add_argument("--solution-name", help="Required when --entrypoint=generated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", help="Optional path to write the summary JSON")
    return parser


def _reference(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    residual_out = residual.to(torch.float32) + hidden_states.to(torch.float32)
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    output = residual_out * torch.rsqrt(variance + eps)
    output = output * weight.to(torch.float32)
    return output.to(hidden_states.dtype), residual_out.to(hidden_states.dtype)


def _run_once(
    fn,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Dict[str, Any]:
    x = hidden_states.clone()
    r = residual.clone()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    fn(x, r, weight, eps)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"latency_ms": latency_ms, "output": x, "residual": r}


def _make_generated_fn(args: argparse.Namespace):
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import TraceSet

    if not args.trace_set_path:
        raise ValueError("--trace-set-path is required when --entrypoint=generated")
    if not args.solution_name:
        raise ValueError("--solution-name is required when --entrypoint=generated")

    trace_set = TraceSet.from_path(args.trace_set_path)
    definition = trace_set.definitions.get(args.definition)
    if definition is None:
        raise ValueError(f"Unknown definition: {args.definition}")
    solution = trace_set.get_solution(args.solution_name)
    if solution is None:
        raise ValueError(f"Unknown solution: {args.solution_name}")

    runnable = BuilderRegistry.get_instance().build(definition, solution)
    if runnable is None:
        raise RuntimeError(f"Failed to build solution runnable: {args.solution_name}")

    def _run_generated(input_tensor: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
        if abs(float(eps) - 1e-5) > 1e-12:
            raise ValueError("Generated fused_add_rmsnorm traces currently assume eps=1e-5")

        if runnable.metadata.destination_passing_style:
            output = torch.empty_like(input_tensor)
            runnable.call_destination_passing(input_tensor, residual, weight, output)
        else:
            output = runnable.call_value_returning(input_tensor, residual, weight)
        residual.add_(input_tensor)
        input_tensor.copy_(output)

    return _run_generated


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    if args.entrypoint == "flashinfer":
        import flashinfer

        fn = flashinfer.norm.fused_add_rmsnorm
        entrypoint_name = "flashinfer.norm.fused_add_rmsnorm"
    elif args.entrypoint == "generated":
        fn = _make_generated_fn(args)
        entrypoint_name = args.solution_name
    else:
        import sgl_kernel

        fn = sgl_kernel.fused_add_rmsnorm
        entrypoint_name = "sgl_kernel.fused_add_rmsnorm"

    hidden_states = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    ref_output, ref_residual = _reference(hidden_states, residual, weight, args.eps)

    for _ in range(args.warmup_runs):
        _run_once(fn, hidden_states, residual, weight, args.eps)

    latencies_ms: List[float] = []
    final_run: Dict[str, Any] | None = None
    for _ in range(args.benchmark_runs):
        final_run = _run_once(fn, hidden_states, residual, weight, args.eps)
        latencies_ms.append(final_run["latency_ms"])

    assert final_run is not None

    output = final_run["output"]
    residual_out = final_run["residual"]
    output_abs_err = (output.float() - ref_output.float()).abs()
    residual_abs_err = (residual_out.float() - ref_residual.float()).abs()

    summary = {
        "entrypoint": entrypoint_name,
        "definition": args.definition if args.entrypoint == "generated" else None,
        "trace_set_path": args.trace_set_path if args.entrypoint == "generated" else None,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "eps": args.eps,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "latency_ms_avg": statistics.fmean(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "raw_latencies_ms": latencies_ms,
        "max_abs_error_output": float(output_abs_err.max().item()),
        "max_abs_error_residual": float(residual_abs_err.max().item()),
        "correct_output_atol_1e-2": bool(torch.allclose(output.float(), ref_output.float(), atol=1e-2, rtol=1e-2)),
        "correct_residual_atol_1e-2": bool(
            torch.allclose(residual_out.float(), ref_residual.float(), atol=1e-2, rtol=1e-2)
        ),
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
