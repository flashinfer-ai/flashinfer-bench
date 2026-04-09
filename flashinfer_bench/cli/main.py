"""Main CLI entry point and command implementations."""

import argparse
import logging
from pathlib import Path
from typing import List

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import TraceSet, save_json_file, save_jsonl_file

logger = logging.getLogger(__name__)
pkg_name = __name__.split(".")[0]


def cli_config_logging(args: argparse.Namespace):
    """Configure package-level logging from CLI args."""
    log_level = getattr(args, "log_level", "WARNING")
    pkg_logger = logging.getLogger(pkg_name)
    pkg_logger.setLevel(log_level)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
            )
        )
        pkg_logger.addHandler(handler)
    pkg_logger.propagate = False


def best(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for trace_set in trace_sets:
        definitions = trace_set.definitions.keys()
        for definition in definitions:
            trace = trace_set.get_best_trace(definition)
            if not trace:
                logger.warning(f"No valid solution found for {definition}.")
                continue
            logger.info(f"Best solution for {definition}:")
            logger.info(f"- Solution: {trace.solution}")
            logger.info(f"- Speedup vs. ref: {trace.evaluation.performance.speedup_factor:.2f}×")
            logger.info(
                f"- Errors:   abs={trace.evaluation.correctness.max_absolute_error:.2e}, "
                f"rel={trace.evaluation.correctness.max_relative_error:.2e}"
            )


def summary(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for i, trace_set in enumerate(trace_sets):
        s = trace_set.summary()

        if len(trace_sets) > 1:
            logger.info("Dataset %d:", i + 1)

        logger.info("Traces: %d total, %d passed, %d failed", s.total, s.passed, s.failed)
        if s.rankings:
            logger.info("")
            logger.info("Author Rankings (by average speedup):")
            logger.info(
                "  %-4s  %-24s  %-10s  %-14s  %-14s  %-10s",
                "Rank",
                "Author",
                "Avg Speedup",
                "Success Rate",
                "Win Rate",
                "Workloads",
            )
            logger.info("  " + "-" * 92)
            for rank, (author, score) in enumerate(s.rankings, start=1):
                logger.info(
                    "  %-4d  %-24s  %-10.4f  %-14.1f  %-14.1f  %-10d",
                    rank,
                    author,
                    score.avg_speedup,
                    score.success_rate * 100,
                    score.win_rate * 100,
                    score.workloads,
                )
            logger.info("")
            logger.info("  Avg Speedup: mean of baseline_latency / author_latency")
            logger.info("  Success Rate: fraction of workloads with successful execution")
            logger.info("  Win Rate: fraction of workloads where this author beats the baseline")
        else:
            logger.info("(No author ranking data available — run with multiple solutions)")


def merge_trace_sets(trace_sets):
    """Merge multiple TraceSets into one, raising on definition conflicts."""
    if not trace_sets:
        raise ValueError("No TraceSets to merge.")
    # Start with a deep copy of the first TraceSet
    from copy import deepcopy

    merged = deepcopy(trace_sets[0])
    for trace_set in trace_sets[1:]:
        # Merge definitions
        for name, definition in trace_set.definitions.items():
            if name in merged.definitions:
                if merged.definitions[name] != definition:
                    raise ValueError(f"Definition conflict for '{name}' during merge.")
            else:
                merged.definitions[name] = definition
        # Merge solutions
        for def_name, solutions in trace_set.solutions.items():
            if def_name not in merged.solutions:
                merged.solutions[def_name] = []
            merged.solutions[def_name].extend(solutions)
        # Merge workloads
        for def_name, workloads in trace_set.workload.items():
            if def_name not in merged.workload:
                merged.workload[def_name] = []
            merged.workload[def_name].extend(workloads)
        # Merge traces
        for def_name, traces in trace_set.traces.items():
            if def_name not in merged.traces:
                merged.traces[def_name] = []
            merged.traces[def_name].extend(traces)
    return merged


def _safe_path_segment(segment: str) -> str:
    """Validate that a string is safe to use as a single path segment."""
    if not segment or "/" in segment or "\\" in segment or segment in (".", ".."):
        raise ValueError(f"Invalid path segment: {segment!r}")
    return segment


def export_trace_set(trace_set, output_dir):
    """Export a TraceSet to a directory in the expected structure."""
    output_dir = Path(output_dir)
    (output_dir / "definitions").mkdir(parents=True, exist_ok=True)
    (output_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    # Save definitions
    for definition in trace_set.definitions.values():
        out_path = output_dir / "definitions" / f"{_safe_path_segment(definition.name)}.json"
        save_json_file(definition, out_path)
    # Save solutions
    for def_name, solutions in trace_set.solutions.items():
        definition = trace_set.definitions[def_name]
        for solution in solutions:
            out_path = (
                output_dir
                / "solutions"
                / _safe_path_segment(solution.author)
                / _safe_path_segment(definition.op_type)
                / _safe_path_segment(def_name)
                / f"{_safe_path_segment(solution.name)}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_json_file(solution, out_path)
    # Save workload traces
    for def_name, workloads in trace_set.workload.items():
        if workloads:
            out_path = output_dir / "traces" / f"{_safe_path_segment(def_name)}_workloads.jsonl"
            save_jsonl_file(workloads, out_path)
    # Save regular traces
    for def_name, traces in trace_set.traces.items():
        if traces:
            out_path = output_dir / "traces" / f"{_safe_path_segment(def_name)}.jsonl"
            save_jsonl_file(traces, out_path)


def merge(args: argparse.Namespace):
    """Merge multiple TraceSets into a single one and export to output directory."""
    if not args.output:
        raise ValueError("--output <MERGED_PATH> is required for merge.")
    trace_sets = _load_traces(args)
    merged = merge_trace_sets(trace_sets)
    export_trace_set(merged, args.output)
    logger.info(f"Merged {len(trace_sets)} TraceSets and exported to {args.output}")


def visualize(args: argparse.Namespace):
    """Visualize benchmark results as a console table."""
    trace_sets = _load_traces(args)

    logger.info("FlashInfer Bench Results Visualization")
    logger.info("=" * 80)

    for i, trace_set in enumerate(trace_sets):
        if len(trace_sets) > 1:
            logger.info(f"\nDataset {i+1}:")
            logger.info("-" * 40)

        # Print summary statistics
        summary = trace_set.summary()
        logger.info(f"Summary: {summary.passed}/{summary.total} traces passed")

        # Print detailed results table
        logger.info("\nDetailed Results:")
        logger.info("-" * 80)
        logger.info(
            f"{'Definition':<15} {'Solution':<25} {'Status':<10} {'Speedup vs. ref':<16} {'Latency(ms)':<12} {'Max Error':<15}"
        )
        logger.info("-" * 80)

        for def_name, traces in trace_set.traces.items():
            for trace in traces:
                status = trace.evaluation.get("status", "UNKNOWN")
                perf = trace.evaluation.get("performance", {})
                corr = trace.evaluation.get("correctness", {})

                speedup = perf.get("speedup_factor", "N/A")
                if isinstance(speedup, (int, float)):
                    speedup = f"{speedup:.2f}×"

                latency = perf.get("latency_ms", "N/A")
                if isinstance(latency, (int, float)):
                    latency = f"{latency:.3f}"

                max_error = corr.get("max_absolute_error", "N/A")
                if isinstance(max_error, (int, float)):
                    max_error = f"{max_error:.2e}"

                logger.info(
                    f"{def_name:<15} {trace.solution:<25} {status:<10} {speedup:<16} {latency:<12} {max_error:<15}"
                )

        # Print best solutions
        logger.info("\nBest Solutions:")
        logger.info("-" * 80)
        for def_name in trace_set.definitions.keys():
            best_trace = trace_set.get_best_op(def_name)
            if best_trace:
                perf = best_trace.evaluation.get("performance", {})
                corr = best_trace.evaluation.get("correctness", {})
                speedup = perf.get("speedup_factor", "N/A")
                if isinstance(speedup, (int, float)):
                    speedup = f"{speedup:.2f}×"
                logger.info(f"{def_name}: {best_trace.solution} (Speedup vs. ref: {speedup})")
            else:
                logger.warning(f"{def_name}: No valid solution found")


def serve(args: argparse.Namespace):
    """Start the benchmark HTTP server."""
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "uvicorn is required for the serve command. "
            "Install with: pip install flashinfer-bench[serve]"
        )

    from flashinfer_bench.bench import BenchmarkConfig
    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.serve.app import init_app
    from flashinfer_bench.serve.scheduler import Scheduler

    trace_set = TraceSet.from_path(str(args.local))

    devices = args.devices.split(",") if args.devices else None
    if devices is None:
        import flashinfer_bench.utils as fib_utils

        devices = fib_utils.list_cuda_devices()
    if not devices:
        raise RuntimeError("No CUDA devices available")

    config = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=args.rtol,
        atol=args.atol,
        timeout_seconds=args.timeout,
    )

    scheduler = Scheduler(trace_set=trace_set, config=config, devices=devices)
    app = init_app(scheduler)

    logger.info(f"Starting server on {args.host}:{args.port} with devices {devices}")
    uvicorn.run(app, host=args.host, port=args.port)


def run(args: argparse.Namespace):
    """Benchmark run: executes benchmarks and writes results."""
    if not args.local:
        raise ValueError("A data source is required. Please use --local <PATH>.")
    # Only support --local for now
    for path in args.local:
        trace_set = TraceSet.from_path(str(path))

        config = BenchmarkConfig(
            warmup_runs=args.warmup_runs,
            iterations=args.iterations,
            num_trials=args.num_trials,
            rtol=args.rtol,
            atol=args.atol,
            use_isolated_runner=args.use_isolated_runner,
            definitions=args.definitions,
            solutions=args.solutions,
            timeout_seconds=args.timeout,
            required_matched_ratio=args.required_matched_ratio,
            profile_baseline=getattr(args, "profile_baseline", True),
        )
        benchmark = Benchmark(trace_set, config)
        logger.info(f"Running benchmark on FlashInfer Trace Dataset: {Path(path).resolve()}")
        resume = getattr(args, "resume", False)
        if resume:
            logger.info("Resume mode enabled: will skip already evaluated solutions")

        try:
            benchmark.run_all(args.save_results, resume=resume)
        except Exception:
            logger.exception("Benchmark run failed")
            raise
        finally:
            benchmark.close()

        message = "Benchmark run complete."
        if args.save_results:
            message += " Results saved."
        else:
            message += " Results not saved (use --save-results to enable saving)."
        logger.info(message)


def _load_traces(args: argparse.Namespace) -> List[TraceSet]:
    trace_sets = []
    if not args.local:
        raise ValueError("A data source is required. Please use --local <PATH>.")

    if args.local:
        loaded_paths: List[Path] = args.local
        for path in loaded_paths:
            trace_sets.append(TraceSet.from_path(str(path)))
    return trace_sets


def cli():
    parser = argparse.ArgumentParser(
        description="FlashInfer Bench CLI", formatter_class=argparse.RawTextHelpFormatter
    )

    command_subparsers = parser.add_subparsers(
        dest="command", required=True, help="Primary commands"
    )

    serve_parser = command_subparsers.add_parser("serve", help="Start the benchmark HTTP server.")
    serve_parser.add_argument(
        "--local", type=Path, required=True, help="Path to the trace set dataset."
    )
    serve_parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA devices (e.g. cuda:0,cuda:1). Default: all available.",
    )
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--warmup-runs", type=int, default=10)
    serve_parser.add_argument("--iterations", type=int, default=50)
    serve_parser.add_argument("--num-trials", type=int, default=3)
    serve_parser.add_argument("--rtol", type=float, default=1e-2)
    serve_parser.add_argument("--atol", type=float, default=1e-2)
    serve_parser.add_argument("--timeout", type=int, default=300)
    serve_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    serve_parser.set_defaults(func=serve)

    run_parser = command_subparsers.add_parser("run", help="Execute a new benchmark run.")
    run_parser.add_argument(
        "--warmup-runs", type=int, default=10, help="Number of warmup runs before measurement"
    )
    run_parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations for benchmarking"
    )
    run_parser.add_argument(
        "--num-trials", type=int, default=3, help="Number of trials for each benchmark"
    )
    run_parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance for correctness checks"
    )
    run_parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for correctness checks"
    )
    run_parser.add_argument(
        "--required-matched-ratio",
        type=float,
        default=None,
        help="Required ratio of elements within tolerance. Overrides evaluator default (0.95 for low-bit).",
    )
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    run_parser.add_argument(
        "--use-isolated-runner",
        action="store_true",
        help="Use IsolatedRunner instead of the default PersistentRunner",
    )
    run_parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume run, skip already evaluated solution-workload jobs",
    )
    run_parser.add_argument(
        "--definitions",
        type=str,
        nargs="+",
        help="List of definition names to run. If not specified, runs all definitions.",
    )
    run_parser.add_argument(
        "--solutions",
        type=str,
        nargs="+",
        help="List of solution names to run. If not specified, runs all solutions.",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each solution evaluation (default: 300)",
    )
    run_parser.add_argument(
        "--no-profile-baseline",
        dest="profile_baseline",
        action="store_false",
        default=True,
        help="Skip profiling the reference implementation (correctness check still runs). "
        "Useful when the reference is slow (e.g. large prefill workloads).",
    )
    run_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    run_parser.set_defaults(func=run)

    report_parser = command_subparsers.add_parser(
        "report", help="Analyze and manage existing traces."
    )
    report_subparsers = report_parser.add_subparsers(
        dest="report_subcommand", required=True, help="Report actions"
    )

    summary_parser = report_subparsers.add_parser(
        "summary", help="Prints a human-readable summary of loaded traces."
    )
    summary_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    summary_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    summary_parser.set_defaults(func=summary)

    best_parser = report_subparsers.add_parser("best", help="Find best solution for a definition.")
    best_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    best_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    best_parser.set_defaults(func=best)

    merge_parser = report_subparsers.add_parser("merge", help="Merges multiple traces.")
    merge_parser.add_argument("--output", type=Path)
    merge_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    merge_parser.set_defaults(func=merge)

    visualize_parser = report_subparsers.add_parser(
        "visualize", help="Generates a visual representation of benchmark results."
    )
    visualize_parser.add_argument(
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    visualize_parser.set_defaults(func=visualize)

    args = parser.parse_args()

    cli_config_logging(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
