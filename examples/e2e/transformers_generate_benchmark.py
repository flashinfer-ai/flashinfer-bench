"""Minimal end-to-end benchmark launcher for Hugging Face Transformers."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIB_E2E_PYTHON = "/data/workspace/airulan/conda_envs/fib_e2e/bin/python"


def _trace_count(trace_set) -> int:
    return sum(len(traces) for traces in trace_set.traces.values())


def _apply_state(
    *,
    requested: bool,
    active: bool,
    status: str,
    runtime: Any = None,
    skip_reason: str = "",
) -> Dict[str, Any]:
    return {
        "requested": requested,
        "active": active,
        "status": status,
        "runtime": runtime,
        "skip_reason": skip_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Transformers model.generate() end-to-end latency with optional "
            "FlashInfer-Bench apply enabled."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--tokenizer", help="Tokenizer name or local path; defaults to --model")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype passed to from_pretrained",
    )
    parser.add_argument(
        "--attn-implementation",
        help="Optional Transformers attention backend, e.g. eager, sdpa, flash_attention_2",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", help="Explicit prompt text. If omitted, a synthetic prompt is used.")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Synthetic prompt length in tokens when --prompt is not provided",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Wrap model with torch.compile")
    parser.add_argument("--enable-apply", action="store_true")
    parser.add_argument("--trace-set-path", help="Trace-set root used by FlashInfer-Bench apply")
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
            "Pinned solutions are registered with use_def_best for E2E testing."
        ),
    )
    parser.add_argument(
        "--solution-pool",
        choices=["all", "generated_only", "baseline_only"],
        default="all",
        help="Which solution pool apply may choose from when not pinning a specific solution",
    )
    parser.add_argument(
        "--apply-scope",
        choices=["gemm_only", "all"],
        default="gemm_only",
        help="Which definitions to register with apply when benchmarking Transformers",
    )
    parser.add_argument(
        "--on-miss-policy",
        choices=["fallback_only", "use_def_best"],
        default="fallback_only",
        help="Apply miss policy",
    )
    parser.add_argument("--max-atol", type=float, default=1e-2)
    parser.add_argument("--max-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--aot-ratio",
        type=float,
        default=0.0,
        help="Fraction of top solutions to AOT-build per definition before the benchmark starts",
    )
    parser.add_argument("--fib-cache-path", help="Optional override for FIB_CACHE_PATH")
    parser.add_argument("--output-json", help="Optional path to write the benchmark summary JSON")
    return parser


def _resolve_dtype(name: str):
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'torch' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _maybe_enable_apply(args: argparse.Namespace):
    if not args.enable_apply:
        return _apply_state(requested=False, active=False, status="disabled")
    if not args.trace_set_path:
        raise ValueError("--trace-set-path is required when --enable-apply is set")

    from flashinfer_bench.apply import ApplyConfig, ApplyConfigRegistry, enable_apply
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.apply.trace_filter import (
        build_filtered_trace_set,
        collect_solution_names_by_pool,
        collect_definition_names_from_solutions,
        count_eligible_traces_by_solution,
    )

    trace_root = str(Path(args.trace_set_path).resolve())
    os.environ["FIB_ENABLE_APPLY"] = "1"
    os.environ["FIB_DATASET_PATH"] = trace_root
    os.environ["FIB_APPLY_ADAPTER_SCOPE"] = "gemm_only" if args.apply_scope == "gemm_only" else "all"
    if args.fib_cache_path:
        os.environ["FIB_CACHE_PATH"] = str(Path(args.fib_cache_path).resolve())

    config = ApplyConfig(
        max_atol=args.max_atol,
        max_rtol=args.max_rtol,
        aot_ratio=args.aot_ratio,
        on_miss_policy=args.on_miss_policy,
    )
    apply_config = config
    trace_hardware_filters = [value.strip() for value in args.trace_hardware_contains if value.strip()]
    solution_language_filters = [value.strip() for value in args.solution_language if value.strip()]

    use_in_memory_trace_set = bool(
        args.pin_solution
        or args.only_definition
        or args.apply_scope == "gemm_only"
        or args.solution_pool != "all"
        or trace_hardware_filters
        or solution_language_filters
    )
    if use_in_memory_trace_set:
        trace_set = TraceSet.from_path(trace_root)
        if trace_hardware_filters or solution_language_filters:
            trace_set = build_filtered_trace_set(
                trace_set,
                definition_names=list(trace_set.definitions.keys()),
                trace_hardware_filters=trace_hardware_filters,
                solution_language_filters=solution_language_filters,
            )
        selected_definitions = list(args.only_definition)
        if not selected_definitions and args.apply_scope == "gemm_only":
            selected_definitions = [
                def_name
                for def_name, definition in trace_set.definitions.items()
                if definition.op_type == "gemm" or def_name.startswith("gemm_")
            ]
        if args.pin_solution:
            pinned_mapping = collect_definition_names_from_solutions(trace_set, args.pin_solution)
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
            selected_definitions.extend(sorted(set(pinned_mapping.values())))
            trace_set = build_filtered_trace_set(
                trace_set,
                solution_names=args.pin_solution,
                definition_names=selected_definitions or None,
                trace_hardware_filters=trace_hardware_filters,
                solution_language_filters=solution_language_filters,
            )
        elif args.solution_pool != "all":
            pool_solutions = collect_solution_names_by_pool(
                trace_set,
                pool=args.solution_pool,
                definition_names=selected_definitions or None,
                solution_language_filters=solution_language_filters,
            )
            if not pool_solutions:
                print(
                    "[apply] no solutions found in pool "
                    f"'{args.solution_pool}' for definitions "
                    f"{sorted(set(selected_definitions)) if selected_definitions else 'ALL'}; "
                    "skipping apply"
                )
                return _apply_state(
                    requested=True,
                    active=False,
                    status="skipped",
                    skip_reason="no_solutions_in_requested_pool",
                )
            trace_set = build_filtered_trace_set(
                trace_set,
                solution_names=pool_solutions,
                definition_names=selected_definitions or None,
                trace_hardware_filters=trace_hardware_filters,
                solution_language_filters=solution_language_filters,
            )
        elif selected_definitions:
            trace_set = build_filtered_trace_set(
                trace_set,
                definition_names=selected_definitions,
                trace_hardware_filters=trace_hardware_filters,
                solution_language_filters=solution_language_filters,
            )

        if _trace_count(trace_set) == 0:
            print("[apply] no traces matched the requested filters; skipping apply")
            return _apply_state(
                requested=True,
                active=False,
                status="skipped",
                skip_reason="no_traces_matched_filters",
            )

        registry = ApplyConfigRegistry()
        for def_name, definition in trace_set.definitions.items():
            if args.pin_solution:
                pinned_config = config.model_copy(update={"on_miss_policy": "use_def_best"})
                registry.register(def_name, pinned_config)
                continue
            if args.only_definition:
                registry.register(def_name, config)
                continue
            if args.apply_scope == "all":
                registry.register(def_name, config)
                continue
            if definition.op_type == "gemm" or def_name.startswith("gemm_"):
                registry.register(def_name, config)
        apply_config = registry
        runtime = enable_apply(trace_set, apply_config)
    else:
        runtime = enable_apply(trace_root, apply_config)
    print(f"[apply] enabled with trace set: {trace_root}")
    if args.only_definition:
        print(f"[apply] restricted definitions: {sorted(set(args.only_definition))}")
    if trace_hardware_filters:
        print(f"[apply] trace hardware filters: {trace_hardware_filters}")
    if solution_language_filters:
        print(f"[apply] solution language filters: {solution_language_filters}")
    if args.solution_pool != "all" and not args.pin_solution:
        print(f"[apply] solution pool: {args.solution_pool}")
    if args.pin_solution:
        print(f"[apply] pinned solutions: {sorted(set(args.pin_solution))}")
    return _apply_state(requested=True, active=True, status="enabled", runtime=runtime)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _collect_selected_solutions(dispatch_stats: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _collect_replaced_definitions(dispatch_stats: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _build_synthetic_inputs(tokenizer, prompt_length: int, batch_size: int):
    import torch

    piece = tokenizer.encode(" hello", add_special_tokens=False)
    if not piece:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer has no reusable token for synthetic prompt generation")
        piece = [eos_id]

    token_ids = (piece * math.ceil(prompt_length / len(piece)))[:prompt_length]
    input_ids = torch.tensor([token_ids] * batch_size, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_inputs(tokenizer, args: argparse.Namespace):
    if args.prompt:
        encoded = tokenizer(
            [args.prompt] * args.batch_size,
            return_tensors="pt",
            padding=True,
        )
        return {key: value for key, value in encoded.items()}
    return _build_synthetic_inputs(tokenizer, args.prompt_length, args.batch_size)


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _synchronize_if_needed(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _load_model_and_tokenizer(args: argparse.Namespace):
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'torch' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'transformers' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    dtype = _resolve_dtype(args.dtype)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(args.device)
    model.eval()
    if args.compile:
        model = torch.compile(model)

    return model, tokenizer


def _run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    from flashinfer_bench.apply import disable_apply

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'torch' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    try:
        from transformers import set_seed
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing dependency 'transformers' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {FIB_E2E_PYTHON}"
        ) from exc

    set_seed(args.seed)
    apply_state = _maybe_enable_apply(args)
    runtime = apply_state["runtime"]

    try:
        model, tokenizer = _load_model_and_tokenizer(args)
        batch = _to_device(_build_inputs(tokenizer, args), args.device)
        input_len = int(batch["input_ids"].shape[1])

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }

        latencies_s: List[float] = []
        generated_tokens_per_run: List[int] = []

        total_runs = args.warmup_runs + args.benchmark_runs
        if runtime is not None:
            runtime.reset_stats()
        with torch.inference_mode():
            for run_idx in range(total_runs):
                _synchronize_if_needed(args.device)
                started = time.perf_counter()
                outputs = model.generate(**batch, **generate_kwargs)
                _synchronize_if_needed(args.device)
                elapsed = time.perf_counter() - started

                generated_tokens = int(outputs.shape[1] - batch["input_ids"].shape[1])
                if run_idx >= args.warmup_runs:
                    latencies_s.append(elapsed)
                    generated_tokens_per_run.append(generated_tokens * args.batch_size)
                elif runtime is not None and run_idx + 1 == args.warmup_runs:
                    runtime.reset_stats()

        avg_latency_s = statistics.fmean(latencies_s)
        avg_generated_tokens = statistics.fmean(generated_tokens_per_run)
        prompt_tokens_total = input_len * args.batch_size
        dispatch_stats = (
            runtime.snapshot_stats()
            if runtime is not None and apply_state["active"]
            else None
        )

        summary = {
            "model": args.model,
            "tokenizer": args.tokenizer or args.model,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation or "",
            "enable_apply": args.enable_apply,
            "apply_requested": apply_state["requested"],
            "apply_active": apply_state["active"],
            "apply_status": apply_state["status"],
            "apply_skip_reason": apply_state["skip_reason"],
            "trace_set_path": (
                str(Path(args.trace_set_path).resolve()) if args.trace_set_path else ""
            ),
            "trace_hardware_filters": [value.strip() for value in args.trace_hardware_contains if value.strip()],
            "solution_pool": args.solution_pool,
            "only_definitions": list(args.only_definition),
            "pinned_solutions": list(args.pin_solution),
            "batch_size": args.batch_size,
            "input_tokens_per_sequence": input_len,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "latency_ms_avg": avg_latency_s * 1000.0,
            "latency_ms_p50": _quantile(latencies_s, 0.50) * 1000.0,
            "latency_ms_p90": _quantile(latencies_s, 0.90) * 1000.0,
            "latency_ms_min": min(latencies_s) * 1000.0,
            "latency_ms_max": max(latencies_s) * 1000.0,
            "generated_tokens_avg": avg_generated_tokens,
            "generated_tokens_per_second": avg_generated_tokens / avg_latency_s,
            "prompt_tokens_per_second": prompt_tokens_total / avg_latency_s,
            "raw_latencies_ms": [value * 1000.0 for value in latencies_s],
            "apply_dispatch_stats": dispatch_stats,
            "apply_replaced_definitions": _collect_replaced_definitions(dispatch_stats),
            "apply_selected_solutions": _collect_selected_solutions(dispatch_stats),
        }
        return summary
    finally:
        if runtime is not None:
            disable_apply()


def main() -> None:
    args = build_parser().parse_args()
    summary = _run_benchmark(args)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[output] wrote benchmark summary to {output_path}")


if __name__ == "__main__":
    main()
