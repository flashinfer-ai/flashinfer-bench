#!/usr/bin/env python3
"""CLI registration and dispatch for model onboarding stages."""

from __future__ import annotations

import argparse
from typing import Any

from flashinfer_bench.onboarding.definition_stage import run_analyze_definition, run_dump_definition
from flashinfer_bench.onboarding.workload_stage import run_dump_workload


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Run path below runs/.")
    parser.add_argument("--model-name")
    parser.add_argument("--tp-size", type=int)
    parser.add_argument(
        "--agent",
        choices=["codex"],
        help="Analyze runtime/source evidence and write reviewed definitions in place.",
    )


def add_cli_subcommands(subparsers: Any) -> None:
    """Register onboarding commands on an argparse subparser collection."""
    definitions = subparsers.add_parser(
        "dump-definition",
        help="Run the bounded request matrix on the current GPU and write definitions.",
    )
    _add_runtime_args(definitions)
    definitions.add_argument(
        "--overwrite", action="store_true", help="Replace an existing definitions/ directory."
    )
    definitions.add_argument(
        "--isl",
        type=int,
        help="Medium synthetic input length; the bounded matrix also includes 128 and long context.",
    )
    definitions.add_argument("--osl", type=int, help="Synthetic output length (default: 8).")
    definitions.add_argument(
        "--random-range-ratio",
        type=float,
        help="Sample request lengths between this ratio and 100%% of each matrix target.",
    )
    definitions.add_argument("--seed", type=int, help="Synthetic request matrix seed.")
    definitions.set_defaults(onboarding_command="dump-definition", func=_run_from_official_cli)

    analysis = subparsers.add_parser(
        "analyze-definition",
        help="Re-run local Agent definition analysis from existing runtime evidence.",
    )
    analysis.add_argument("--run", required=True, help="Run path below runs/.")
    analysis.add_argument("--agent", choices=["codex"], default="codex")
    analysis.set_defaults(onboarding_command="analyze-definition", func=_run_from_official_cli)

    workloads = subparsers.add_parser(
        "dump-workload",
        help="Replay the definition-stage request matrix and collect reviewed workloads.",
    )
    _add_runtime_args(workloads)
    workloads.add_argument("--max-new-workloads", type=int)
    workloads.set_defaults(onboarding_command="dump-workload", func=_run_from_official_cli)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage model onboarding")
    subparsers = parser.add_subparsers(dest="onboarding_command", required=True)
    add_cli_subcommands(subparsers)
    return parser


def run_command(args: argparse.Namespace) -> int:
    """Dispatch one parsed onboarding command to its stage implementation."""
    if args.onboarding_command == "dump-definition":
        return run_dump_definition(args)
    if args.onboarding_command == "analyze-definition":
        return run_analyze_definition(args)
    if args.onboarding_command == "dump-workload":
        return run_dump_workload(args)
    raise AssertionError(f"unsupported command: {args.onboarding_command}")


def _run_from_official_cli(args: argparse.Namespace) -> None:
    returncode = run_command(args)
    if returncode:
        raise SystemExit(returncode)


def main(argv: list[str] | None = None) -> int:
    return run_command(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
