"""Main CLI entry point and command implementations."""

import argparse
import logging
from pathlib import Path
from typing import List

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import TraceSet, save_json_file, save_jsonl_file
from flashinfer_bench.logging import configure_logging, get_logger

logger = get_logger("CLI")
pkg_name = __name__.split(".")[0]


def cli_config_logging(args: argparse.Namespace):
    """Configure logging for the CLI. Now we have two set of loggers:
    - package logger: obtained by logging.getLogger(__name__)
    - custom logger: obtained by get_logger("ModuleName")

    In the future we will deprecate the custom logger and use the package logger only.
    Now we need to configure both loggers to the same level.
    """
    log_level = getattr(args, "log_level", "WARNING")
    pkg_logger = logging.getLogger(pkg_name)
    pkg_logger.setLevel(log_level)
    pkg_logger.addHandler(logging.StreamHandler())
    configure_logging(level=log_level)


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
            logger.info(f"- Speedup:  {trace.evaluation.performance.speedup_factor:.2f}×")
            logger.info(
                f"- Errors:   abs={trace.evaluation.correctness.max_absolute_error:.2e}, "
                f"rel={trace.evaluation.correctness.max_relative_error:.2e}"
            )
            if trace.evaluation.log:
                logger.info("- Log snippet:")
                for line in trace.evaluation.log.splitlines()[:5]:
                    logger.info("  %s", line)


def summary(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for trace_set in trace_sets:
        logger.info("%s", trace_set.summary())


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


def export_trace_set(trace_set, output_dir):
    """Export a TraceSet to a directory in the expected structure."""
    output_dir = Path(output_dir)
    (output_dir / "definitions").mkdir(parents=True, exist_ok=True)
    (output_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(parents=True, exist_ok=True)
    # Save definitions
    for definition in trace_set.definitions.values():
        out_path = output_dir / "definitions" / f"{definition.name}.json"
        save_json_file(definition, out_path)
    # Save solutions
    for def_name, solutions in trace_set.solutions.items():
        for solution in solutions:
            out_path = output_dir / "solutions" / f"{solution.name}.json"
            save_json_file(solution, out_path)
    # Save workload traces
    for def_name, workloads in trace_set.workload.items():
        if workloads:
            out_path = output_dir / "traces" / f"{def_name}_workloads.jsonl"
            save_jsonl_file(workloads, out_path)
    # Save regular traces
    for def_name, traces in trace_set.traces.items():
        if traces:
            out_path = output_dir / "traces" / f"{def_name}.jsonl"
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
        logger.info(f"Summary: {summary['passed']}/{summary['total']} traces passed")
        if summary["avg_latency_ms"]:
            logger.info(f"Average latency: {summary['avg_latency_ms']:.3f}ms")

        # Print detailed results table
        logger.info("\nDetailed Results:")
        logger.info("-" * 80)
        logger.info(
            f"{'Definition':<15} {'Solution':<25} {'Status':<10} {'Speedup':<10} {'Latency(ms)':<12} {'Max Error':<15}"
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
                    f"{def_name:<15} {trace.solution:<25} {status:<10} {speedup:<10} {latency:<12} {max_error:<15}"
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
                logger.info(f"{def_name}: {best_trace.solution} (Speedup: {speedup})")
            else:
                logger.warning(f"{def_name}: No valid solution found")


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
    if not args.local and not args.hub:
        raise ValueError("A data source is required. Please use --local <PATH> or --hub.")

    if args.hub:
        raise NotImplementedError("Loading from --hub is not implemented yet.")

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
        "--local",
        type=Path,
        action="append",
        help="Specifies one or more local paths to load traces from.",
    )
    run_parser.add_argument(
        "--hub", action="store_true", help="Load the latest traces from the FlashInfer Hub."
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
        "--hub", action="store_true", help="Load the latest traces from the FlashInfer Hub."
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
        "--hub", action="store_true", help="Load the latest traces from the FlashInfer Hub."
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
    merge_parser.add_argument(
        "--hub", action="store_true", help="Load the latest traces from the FlashInfer Hub."
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
    visualize_parser.add_argument(
        "--hub", action="store_true", help="Load the latest traces from the FlashInfer Hub."
    )
    visualize_parser.set_defaults(func=visualize)

    args = parser.parse_args()

    cli_config_logging(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
