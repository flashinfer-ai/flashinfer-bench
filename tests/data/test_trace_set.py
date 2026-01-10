import sys
from pathlib import Path

import pytest

from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)


def test_trace_set_from_path_and_queries(tmp_path: Path):
    # Create directory structure
    (tmp_path / "definitions").mkdir()
    (tmp_path / "solutions").mkdir()
    (tmp_path / "traces").mkdir()

    # Definition
    ref = "def run(a):\n    return a\n"
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref,
    )
    save_json_file(definition, tmp_path / "definitions" / "d1.json")

    # Solutions
    solution1 = Solution(
        name="s1",
        definition="d1",
        author="a",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    solution2 = Solution(
        name="s2",
        definition="d1",
        author="b",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    save_json_file(solution1, tmp_path / "solutions" / "s1.json")
    save_json_file(solution2, tmp_path / "solutions" / "s2.json")

    # Traces JSONL
    trace_pass = Trace(
        definition="d1",
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="tw1"),
        solution="s1",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="log",
            environment=Environment(hardware="cpu"),
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )
    trace_fail = Trace(
        definition="d1",
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="tw2"),
        solution="s2",
        evaluation=Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            log="log",
            environment=Environment(hardware="cpu"),
            timestamp="t",
        ),
    )
    trace_workload = Trace(
        definition="d1", workload=Workload(axes={"M": 3}, inputs={"A": RandomInput()}, uuid="tw3")
    )
    save_jsonl_file([trace_workload], tmp_path / "workloads" / "d1.jsonl")
    save_jsonl_file([trace_pass, trace_fail], tmp_path / "traces" / "d1.jsonl")

    # Load
    trace_set = TraceSet.from_path(str(tmp_path))

    # Queries
    assert trace_set.definitions.get("d1").name == "d1"
    assert trace_set.get_solution("s1").name == "s1"
    assert len(trace_set.workloads.get("d1", [])) == 1
    assert len(trace_set.traces.get("d1", [])) == 2  # pass + fail

    # Best trace should pick the passed one with higher speedup
    best = trace_set.get_best_trace("d1", axes={"M": 2})
    assert best is not None and best.solution == "s1"

    # Summary
    summary = trace_set.summary()
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["min_latency_ms"] == 1.0
    assert summary["avg_latency_ms"] == 1.0
    assert summary["max_latency_ms"] == 1.0


if __name__ == "__main__":
    pytest.main(sys.argv)
