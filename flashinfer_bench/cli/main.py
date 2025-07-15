import argparse
from pathlib import Path
from typing import List

from flashinfer_bench import TraceSet, Benchmark, BenchmarkConfig
from flashinfer_bench.utils.json_utils import save_jsonl


def best(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for trace_set in trace_sets:
        definitions = trace_set.definitions.keys()
        for definition in definitions:
            trace = trace_set.get_best_op(definition)
            if not trace:
                print(f"No valid solution found for {definition}.")
                continue
            print(f"Best solution for {definition}:")
            print(f"- Solution: {trace.solution}")
            print(f"- Speedup:  {trace.evaluation['performance']['speedup_factor']:.2f}×")
            print(
                f"- Errors:   abs={trace.evaluation['correctness']['max_absolute_error']:.2e}, "
                f"rel={trace.evaluation['correctness']['max_relative_error']:.2e}"
            )
            print(f"- Log:      {trace.evaluation['log_file']}")


def summary(args: argparse.Namespace):
    trace_sets = _load_traces(args)
    for trace_set in trace_sets:
        print(trace_set.summary())


def merge(args: argparse.Namespace):
    """Merge multiple TraceSets into a single one. WIP"""
    print(f"Received arguments: {args}")
    raise NotImplementedError("Merge is not implemented yet.")


def visualize(args: argparse.Namespace):
    """Visualize benchmark results. WIP"""
    print(f"Received arguments: {args}")
    raise NotImplementedError("Visualization is not implemented yet.")


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
            device=args.device,
            log_level=args.log_level,
        )
        benchmark = Benchmark(trace_set)
        print(f"Running benchmark for: {path}")
        benchmark.run(config)
        if args.save_results:
            # Save updated traces back to the traces directory
            traces_dir = Path(path) / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            for def_name, traces in trace_set.traces.items():
                if not traces:
                    continue
                out_path = traces_dir / f"{def_name}.jsonl"
                save_jsonl(traces, out_path)
            print(f"Results saved to {traces_dir}")
        else:
            print("Benchmark run complete. Results not saved (use --save-results to enable saving).")


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
    # TODO: Implement flashinfer-bench run
    run_parser.add_argument("--warmup-runs", type=int, default=10)
    run_parser.add_argument("--iterations", type=int, default=50)
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--log-level", default="INFO")
    run_parser.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
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
    best_parser.set_defaults(func=best)

    merge_parser = report_subparsers.add_parser("merge", help="Merges multiple traces.")
    merge_parser.add_argument("--output", type=Path)
    merge_parser.set_defaults(func=merge)

    visualize_parser = report_subparsers.add_parser(
        "visualize", help="Generates a visual representation of benchmark results."
    )
    visualize_parser.set_defaults(func=visualize)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
