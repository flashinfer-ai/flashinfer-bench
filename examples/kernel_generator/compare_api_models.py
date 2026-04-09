"""Batch-compare OpenAI-compatible API models on FlashInfer-Bench kernel generation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_generator import KernelGenerator

from flashinfer_bench.bench.error_taxonomy import classify_trace
from flashinfer_bench.data import EvaluationStatus, Trace, TraceSet, save_json_file

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local convenience
    load_dotenv = None

if TYPE_CHECKING:
    from flashinfer_bench.bench.config import BenchmarkConfig


def _safe_path_segment(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    value = value.strip("._")
    return value or "unknown"


def _natural_sort_key(text: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_api_key(env_name: str) -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable '{env_name}'")
    return api_key


def _build_feedback_config(args: argparse.Namespace) -> BenchmarkConfig:
    from flashinfer_bench.bench.config import BenchmarkConfig

    return BenchmarkConfig(
        warmup_runs=args.feedback_warmup_runs,
        iterations=args.feedback_iterations,
        num_trials=args.feedback_num_trials,
        rtol=args.rtol,
        atol=args.atol,
        timeout_seconds=args.feedback_timeout,
        use_isolated_runner=args.use_isolated_runner,
    )


def _build_final_config(args: argparse.Namespace) -> BenchmarkConfig:
    from flashinfer_bench.bench.config import BenchmarkConfig

    return BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=args.rtol,
        atol=args.atol,
        timeout_seconds=args.timeout,
        use_isolated_runner=args.use_isolated_runner,
    )


def _pick_feedback_workload(
    definition_name: str,
    workloads: List[Trace],
    workload_mode: str,
    random_seed: int,
    workload_uuid: Optional[str],
) -> Trace:
    if not workloads:
        raise ValueError(f"No workloads found for definition '{definition_name}'")

    if workload_uuid is not None:
        selected = next((trace for trace in workloads if trace.workload.uuid == workload_uuid), None)
        if selected is None:
            raise ValueError(
                f"Workload UUID '{workload_uuid}' not found for definition '{definition_name}'"
            )
        return selected

    if workload_mode == "first":
        return workloads[0]

    import random

    chooser = random.Random(random_seed)
    return chooser.choice(workloads)


def _save_solution(trace_root: Path, definition_name: str, op_type: str, solution) -> Path:
    author_dir = _safe_path_segment(solution.author)
    filename = f"{_safe_path_segment(solution.name)}.json"
    solution_path = trace_root / "solutions" / author_dir / op_type / definition_name / filename
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(solution, solution_path)
    return solution_path


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _model_sort_key(model: Dict[str, Any]) -> Any:
    return (int(model.get("created") or 0), _natural_sort_key(model.get("id", "")))


def _discover_models(
    api_key: str,
    base_url: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    probe = KernelGenerator(
        model_name="model-discovery",
        language=args.language,
        target_gpu=args.target_gpu,
        api_key=api_key,
        base_url=base_url,
        api_mode=args.api_mode,
        temperature=args.temperature,
        use_ffi=args.use_ffi,
        feedback_benchmark_config=_build_feedback_config(args),
    )
    return probe.list_models()


def _select_models(args: argparse.Namespace, discovered_models: List[Dict[str, Any]]) -> List[str]:
    if args.models:
        return list(dict.fromkeys(args.models))

    if not args.model_prefixes:
        raise ValueError("Provide --models or --model-prefixes, or use --list-models")

    selected: List[str] = []
    for prefix in args.model_prefixes:
        matches = [model for model in discovered_models if model.get("id", "").startswith(prefix)]
        if not matches:
            print(f"[warn] no models matched prefix '{prefix}'")
            continue

        if args.all_matching_models:
            matches.sort(key=_model_sort_key)
            selected.extend(model["id"] for model in matches)
            continue

        latest = sorted(matches, key=_model_sort_key)[-1]
        selected.append(latest["id"])

    return list(dict.fromkeys(selected))


def _benchmark_solution(
    trace_set: TraceSet,
    definition_name: str,
    solution,
    config: BenchmarkConfig,
    save_traces: bool,
) -> List[Trace]:
    from flashinfer_bench.bench.benchmark import Benchmark

    definition = trace_set.definitions[definition_name]
    benchmark_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition_name: definition},
        solutions={definition_name: [solution]},
        workloads={definition_name: trace_set.workloads.get(definition_name, [])},
        traces={definition_name: []},
    )

    benchmark = Benchmark(benchmark_trace_set, config)
    try:
        result_trace_set = benchmark.run_all(dump_traces=save_traces, resume=False)
    finally:
        benchmark.close()

    return result_trace_set.traces.get(definition_name, [])


def _make_trace_row(
    model_name: str,
    definition_name: str,
    solution_name: str,
    feedback_workload_uuid: str,
    trace: Trace,
) -> Dict[str, Any]:
    evaluation = trace.evaluation
    taxonomy = classify_trace(trace)
    correctness = evaluation.correctness if evaluation else None
    performance = evaluation.performance if evaluation else None

    return {
        "model": model_name,
        "definition": definition_name,
        "solution": solution_name,
        "feedback_workload_uuid": feedback_workload_uuid,
        "benchmark_workload_uuid": trace.workload.uuid,
        "status": evaluation.status.value if evaluation else "MISSING",
        "status_family": taxonomy.status_family,
        "secondary_bucket": taxonomy.secondary_bucket,
        "efficiency_bucket": taxonomy.efficiency_bucket or "",
        "max_absolute_error": (
            correctness.max_absolute_error if correctness is not None else ""
        ),
        "max_relative_error": (
            correctness.max_relative_error if correctness is not None else ""
        ),
        "latency_ms": performance.latency_ms if performance is not None else "",
        "reference_latency_ms": (
            performance.reference_latency_ms if performance is not None else ""
        ),
        "speedup_factor": performance.speedup_factor if performance is not None else "",
        "log_excerpt": ((evaluation.log or "")[:240].replace("\n", "\\n") if evaluation else ""),
    }


def _make_experiment_row(
    model_name: str,
    definition_name: str,
    solution_name: str,
    feedback_workload_uuid: str,
    traces: List[Trace],
    *,
    experiment_status: str,
    error_message: str = "",
) -> Dict[str, Any]:
    status_counts = Counter(
        trace.evaluation.status.value for trace in traces if trace.evaluation is not None
    )
    taxonomy_counts = Counter(classify_trace(trace).secondary_bucket for trace in traces)
    passed_traces = [
        trace
        for trace in traces
        if trace.evaluation is not None and trace.evaluation.status == EvaluationStatus.PASSED
    ]

    best_speedup = (
        max(trace.evaluation.performance.speedup_factor for trace in passed_traces)
        if passed_traces
        else ""
    )
    avg_speedup = (
        sum(trace.evaluation.performance.speedup_factor for trace in passed_traces)
        / len(passed_traces)
        if passed_traces
        else ""
    )
    top_error_bucket = taxonomy_counts.most_common(1)[0][0] if taxonomy_counts else ""

    return {
        "model": model_name,
        "definition": definition_name,
        "solution": solution_name,
        "feedback_workload_uuid": feedback_workload_uuid,
        "experiment_status": experiment_status,
        "total_traces": len(traces),
        "passed_traces": len(passed_traces),
        "compile_errors": status_counts.get(EvaluationStatus.COMPILE_ERROR.value, 0),
        "runtime_errors": status_counts.get(EvaluationStatus.RUNTIME_ERROR.value, 0),
        "correctness_errors": (
            status_counts.get(EvaluationStatus.INCORRECT_SHAPE.value, 0)
            + status_counts.get(EvaluationStatus.INCORRECT_DTYPE.value, 0)
            + status_counts.get(EvaluationStatus.INCORRECT_NUMERICAL.value, 0)
        ),
        "timeouts": status_counts.get(EvaluationStatus.TIMEOUT.value, 0),
        "pass_rate": (len(passed_traces) / len(traces)) if traces else "",
        "best_speedup": best_speedup,
        "avg_speedup": avg_speedup,
        "top_error_bucket": top_error_bucket,
        "error_message": error_message,
    }


def _summarize_errors(trace_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter(
        (row["model"], row["definition"], row["secondary_bucket"]) for row in trace_rows
    )
    rows = []
    for (model_name, definition_name, secondary_bucket), count in sorted(counter.items()):
        rows.append(
            {
                "model": model_name,
                "definition": definition_name,
                "secondary_bucket": secondary_bucket,
                "count": count,
            }
        )
    return rows


def _print_summary(experiment_rows: List[Dict[str, Any]]) -> None:
    print("\nExperiment summary")
    print("=" * 80)
    for row in experiment_rows:
        best_speedup = row["best_speedup"]
        best_speedup_text = f"{best_speedup:.2f}" if isinstance(best_speedup, float) else "n/a"
        pass_rate = row["pass_rate"]
        pass_rate_text = f"{pass_rate:.2%}" if isinstance(pass_rate, float) else "n/a"
        print(
            f"{row['model']} | {row['definition']} | {row['experiment_status']} | "
            f"pass_rate={pass_rate_text} | best_speedup={best_speedup_text} | "
            f"top_bucket={row['top_error_bucket'] or 'n/a'}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare OpenAI-compatible API models on FlashInfer-Bench CUDA code generation."
    )
    parser.add_argument(
        "--trace-set-path",
        required=True,
        help="Path to the local flashinfer-trace dataset",
    )
    parser.add_argument(
        "--base-url",
        default="https://aigc.x-see.cn/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="Environment variable that stores the API key",
    )
    parser.add_argument("--models", nargs="*", default=[], help="Explicit model IDs to benchmark")
    parser.add_argument(
        "--model-prefixes",
        nargs="*",
        default=[],
        help="Discover models via /models and select the latest match for each prefix",
    )
    parser.add_argument(
        "--all-matching-models",
        action="store_true",
        help="When using --model-prefixes, keep all matching models instead of only the latest",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List provider models from /models and exit",
    )
    parser.add_argument("--definitions", nargs="*", default=[], help="Definitions to benchmark")
    parser.add_argument("--language", default="cuda", choices=["cuda", "triton", "python"])
    parser.add_argument("--target-gpu", default="A800")
    parser.add_argument("--gen-rounds", type=int, default=4)
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument(
        "--workload-mode",
        choices=["first", "random"],
        default="first",
        help="How to select the single feedback workload used during iterative generation",
    )
    parser.add_argument(
        "--workload-uuid",
        help="Optional explicit workload UUID for the generation feedback loop",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ffi", action="store_true", help="Generate TVM-FFI CUDA bindings")
    parser.add_argument(
        "--api-mode",
        choices=["auto", "chat", "responses"],
        default="auto",
        help="Force the API surface used for generation",
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--feedback-warmup-runs", type=int, default=2)
    parser.add_argument("--feedback-iterations", type=int, default=10)
    parser.add_argument("--feedback-num-trials", type=int, default=1)
    parser.add_argument("--feedback-timeout", type=int, default=180)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--use-isolated-runner", action="store_true")
    parser.add_argument(
        "--no-save-traces",
        action="store_true",
        help="Do not persist final benchmark traces back to flashinfer-trace",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for run manifests and CSV summaries",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    args = build_parser().parse_args()
    api_key = _load_api_key(args.api_key_env)
    trace_root = Path(args.trace_set_path).resolve()
    trace_set = TraceSet.from_path(str(trace_root))

    discovered_models: List[Dict[str, Any]] = []
    if args.list_models or args.model_prefixes:
        discovered_models = _discover_models(api_key, args.base_url, args)

    if args.list_models:
        for model in discovered_models:
            created = model.get("created", "")
            owned_by = model.get("owned_by", "")
            print(f"{model.get('id', '')}\tcreated={created}\towned_by={owned_by}")
        return

    selected_models = _select_models(args, discovered_models)
    if not selected_models:
        raise RuntimeError("No models selected")

    definition_names = list(args.definitions) if args.definitions else sorted(trace_set.definitions)
    missing_definitions = [name for name in definition_names if name not in trace_set.definitions]
    if missing_definitions:
        raise ValueError(f"Definitions not found: {missing_definitions}")

    feedback_workloads: Dict[str, Trace] = {}
    for idx, definition_name in enumerate(definition_names):
        feedback_workloads[definition_name] = _pick_feedback_workload(
            definition_name,
            trace_set.workloads.get(definition_name, []),
            workload_mode=args.workload_mode,
            random_seed=args.seed + idx,
            workload_uuid=args.workload_uuid,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "trace_set_path": str(trace_root),
        "base_url": args.base_url,
        "api_mode": args.api_mode,
        "language": args.language,
        "target_gpu": args.target_gpu,
        "models": selected_models,
        "definitions": definition_names,
        "workload_mode": args.workload_mode,
        "workload_uuid": args.workload_uuid,
        "save_traces": not args.no_save_traces,
        "feedback_benchmark": _build_feedback_config(args).__dict__,
        "final_benchmark": _build_final_config(args).__dict__,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if discovered_models:
        with open(run_dir / "available_models.json", "w", encoding="utf-8") as f:
            json.dump(discovered_models, f, indent=2, ensure_ascii=False)

    experiment_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for model_name in selected_models:
        generator = KernelGenerator(
            model_name=model_name,
            language=args.language,
            target_gpu=args.target_gpu,
            api_key=api_key,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            api_mode=args.api_mode,
            temperature=args.temperature,
            use_ffi=args.use_ffi,
            feedback_benchmark_config=_build_feedback_config(args),
        )

        for definition_name in definition_names:
            definition = trace_set.definitions[definition_name]
            feedback_workload = feedback_workloads[definition_name]
            print(
                f"\n[{model_name}] generating {definition_name} "
                f"using feedback workload {feedback_workload.workload.uuid}"
            )

            try:
                solution = generator.generate(
                    trace_set=trace_set,
                    definition=definition,
                    gen_rounds=args.gen_rounds,
                    beam=args.beam,
                    beam_width=args.beam_width,
                    selected_workload=feedback_workload,
                    random_seed=args.seed,
                )
                solution_path = _save_solution(
                    trace_root,
                    definition_name,
                    definition.op_type,
                    solution,
                )
                traces = _benchmark_solution(
                    trace_set=trace_set,
                    definition_name=definition_name,
                    solution=solution,
                    config=_build_final_config(args),
                    save_traces=not args.no_save_traces,
                )
                experiment_rows.append(
                    _make_experiment_row(
                        model_name=model_name,
                        definition_name=definition_name,
                        solution_name=solution.name,
                        feedback_workload_uuid=feedback_workload.workload.uuid,
                        traces=traces,
                        experiment_status="OK",
                    )
                    | {"solution_path": str(solution_path)}
                )
                for trace in traces:
                    trace_rows.append(
                        _make_trace_row(
                            model_name=model_name,
                            definition_name=definition_name,
                            solution_name=solution.name,
                            feedback_workload_uuid=feedback_workload.workload.uuid,
                            trace=trace,
                        )
                    )
            except Exception as exc:
                error_message = str(exc)
                print(f"[error] {model_name} / {definition_name}: {error_message}")
                experiment_rows.append(
                    _make_experiment_row(
                        model_name=model_name,
                        definition_name=definition_name,
                        solution_name="",
                        feedback_workload_uuid=feedback_workload.workload.uuid,
                        traces=[],
                        experiment_status="FAILED",
                        error_message=error_message,
                    )
                )
                if args.fail_fast:
                    raise

    error_rows = _summarize_errors(trace_rows)

    _write_csv(
        run_dir / "experiment_summary.csv",
        experiment_rows,
        fieldnames=[
            "model",
            "definition",
            "solution",
            "solution_path",
            "feedback_workload_uuid",
            "experiment_status",
            "total_traces",
            "passed_traces",
            "compile_errors",
            "runtime_errors",
            "correctness_errors",
            "timeouts",
            "pass_rate",
            "best_speedup",
            "avg_speedup",
            "top_error_bucket",
            "error_message",
        ],
    )
    _write_csv(
        run_dir / "trace_records.csv",
        trace_rows,
        fieldnames=[
            "model",
            "definition",
            "solution",
            "feedback_workload_uuid",
            "benchmark_workload_uuid",
            "status",
            "status_family",
            "secondary_bucket",
            "efficiency_bucket",
            "max_absolute_error",
            "max_relative_error",
            "latency_ms",
            "reference_latency_ms",
            "speedup_factor",
            "log_excerpt",
        ],
    )
    _write_csv(
        run_dir / "error_summary.csv",
        error_rows,
        fieldnames=["model", "definition", "secondary_bucket", "count"],
    )

    _print_summary(experiment_rows)
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
