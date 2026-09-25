"""FlashInfer-Bench evaluation adapter for GPU-Mode.

Invoked by discord-cluster-manager as: python3 eval.py <mode> <cases_file>
Communicates results via the POPCORN_FD pipe protocol.
"""

import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Optional


class PopcornOutput:
    def __init__(self, fd: int):
        self.file = os.fdopen(fd, "w")
        os.set_inheritable(fd, False)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.file.close()

    def log(self, key, value):
        print(f"{key}: {value}", file=self.file, flush=True)


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def get_test_cases(file_name: str) -> list[TestCase]:
    content = Path(file_name).read_text()
    tests = []
    pattern = r"\s*([a-zA-Z_]+):\s*([a-zA-Z_]+[a-zA-Z0-9_]*|[+-]?[0-9]+)\s*"
    for line in content.splitlines():
        if not line.strip():
            continue
        case = {}
        for part in line.split(";"):
            m = re.match(pattern, part)
            if not m:
                print(f"invalid test case: '{line}': '{part}'", file=sys.stderr)
                sys.exit(113)
            key, val = m[1], m[2]
            try:
                val = int(val)
            except ValueError:
                pass
            case[key] = val
        tests.append(TestCase(spec=line, args=case))
    return tests


def _load_trace_set(data_path: Optional[str] = None):
    from flashinfer_bench.data import TraceSet

    path = data_path or os.environ.get("FIB_DATASET_PATH") or "/data/flashinfer-trace"
    return TraceSet.from_path(path)


def _make_solution(definition_name: str, submission_code: str):
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile

    return Solution(
        name="gpumode-submission",
        definition=definition_name,
        author="gpumode-user",
        spec=BuildSpec(
            language="python",
            target_hardware=["cuda"],
            entry_point="submission.py::custom_kernel",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="submission.py", content=submission_code)],
    )


def _run_benchmark(definition_name: str, trace_set, solution, fast: bool = False):
    """Run flashinfer-bench evaluation and return result traces."""
    from flashinfer_bench.bench import Benchmark, BenchmarkConfig

    trace_set.add_solution(solution)

    cfg_kwargs = {
        "definitions": [definition_name],
        "solutions": [solution.name],
        "profile_baseline": True,
    }
    if fast:
        cfg_kwargs.update(warmup_runs=2, iterations=5, num_trials=1)

    config = BenchmarkConfig(**cfg_kwargs)
    benchmark = Benchmark(trace_set, config)
    try:
        result = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()
    return result.traces.get(definition_name, [])


def run_testing(logger: PopcornOutput, definition_name: str, trace_set, solution):
    """Run correctness check and report via POPCORN_FD."""
    traces = _run_benchmark(definition_name, trace_set, solution, fast=True)

    logger.log("test-count", len(traces))
    passed = True
    for i, t in enumerate(traces):
        wl_uuid = t.workload.uuid if t.workload else f"workload_{i}"
        logger.log(f"test.{i}.spec", f"workload: {wl_uuid}")
        if t.evaluation and t.evaluation.status.value == "PASSED":
            logger.log(f"test.{i}.status", "pass")
        else:
            logger.log(f"test.{i}.status", "fail")
            error = ""
            if t.evaluation:
                error = t.evaluation.log or t.evaluation.status.value
            logger.log(f"test.{i}.error", error)
            passed = False

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def run_benchmarking(logger: PopcornOutput, definition_name: str, trace_set, solution):
    """Run timing benchmark and report via POPCORN_FD."""
    traces = _run_benchmark(definition_name, trace_set, solution, fast=False)

    passed = True
    logger.log("benchmark-count", len(traces))
    for i, t in enumerate(traces):
        wl_uuid = t.workload.uuid if t.workload else f"workload_{i}"
        logger.log(f"benchmark.{i}.spec", f"workload: {wl_uuid}")

        if not t.evaluation or t.evaluation.status.value != "PASSED":
            passed = False
            error = ""
            if t.evaluation:
                error = t.evaluation.log or t.evaluation.status.value
            logger.log(f"benchmark.{i}.status", "fail")
            logger.log(f"benchmark.{i}.error", error)
            continue

        latency_ns = int(t.evaluation.performance.latency_ms * 1e6)
        logger.log(f"benchmark.{i}.runs", 1)
        logger.log(f"benchmark.{i}.mean", latency_ns)
        logger.log(f"benchmark.{i}.std", 0)
        logger.log(f"benchmark.{i}.err", 0)
        logger.log(f"benchmark.{i}.best", latency_ns)
        logger.log(f"benchmark.{i}.worst", latency_ns)

    logger.log("check", "pass" if passed else "fail")
    return 0 if passed else 112


def main():
    fd = os.getenv("POPCORN_FD")
    if not fd:
        print("POPCORN_FD not set", file=sys.stderr)
        return 111

    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <mode> <cases_file>", file=sys.stderr)
        return 2

    mode = sys.argv[1]
    tests = get_test_cases(sys.argv[2])
    if not tests:
        print("No test cases found", file=sys.stderr)
        return 113

    definition_name = tests[0].args.get("definition")
    if not definition_name:
        print("Test case missing 'definition' key", file=sys.stderr)
        return 113

    data_path = os.environ.get("FIB_DATASET_PATH")
    trace_set = _load_trace_set(data_path)
    submission_code = Path("submission.py").read_text()
    solution = _make_solution(definition_name, submission_code)

    with PopcornOutput(int(fd)) as logger:
        if mode == "test":
            return run_testing(logger, definition_name, trace_set, solution)
        elif mode in ("benchmark", "leaderboard"):
            return run_benchmarking(logger, definition_name, trace_set, solution)
        else:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    sys.exit(main())
