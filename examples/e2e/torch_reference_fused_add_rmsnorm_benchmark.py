"""Benchmark a pure PyTorch fused-add-RMSNorm reference implementation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a pure PyTorch fused_add_rmsnorm reference implementation."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", help="Optional path to write the summary JSON")
    return parser


def _run_once(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> float:
    x = hidden_states.clone()
    r = residual.clone()
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    start = time.perf_counter()
    residual_out = r.float() + x.float()
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    output = residual_out * torch.rsqrt(variance + eps)
    output = (output * weight.float()).to(x.dtype)
    r.copy_(residual_out.to(x.dtype))
    x.copy_(output)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    hidden_states = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)

    for _ in range(args.warmup_runs):
        _run_once(hidden_states, residual, weight, args.eps)

    latencies_ms = [
        _run_once(hidden_states, residual, weight, args.eps) for _ in range(args.benchmark_runs)
    ]
    summary = {
        "entrypoint": "torch_reference_fused_add_rmsnorm",
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
