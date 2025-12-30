from __future__ import annotations

import json
import sys

import pytest

from flashinfer_bench.apply import ApplyConfig
from flashinfer_bench.apply.key import ApplyKeyFactory
from flashinfer_bench.apply.table import ApplyTable, _apply_table_dir
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
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def make_minimal_def() -> Definition:
    return Definition(
        name="add",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X\n",
    )


def make_python_solution(name: str, body: str = "def run(X, Y):\n    return X\n") -> Solution:
    return Solution(
        name=name,
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content=body)],
    )


def make_eval(speedup: float) -> Evaluation:
    return Evaluation(
        status=EvaluationStatus.PASSED,
        log="log",
        environment=Environment(hardware="cpu"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
        performance=Performance(
            latency_ms=1.0 / max(speedup, 1e-6), reference_latency_ms=1.0, speedup_factor=speedup
        ),
    )


def make_traces() -> tuple[Definition, list[Solution], list[Trace]]:
    definition = make_minimal_def()
    solution1 = make_python_solution("add_fast")
    solution2 = make_python_solution("add_slow")

    # Two keys: M=2 best solution1, M=3 best solution2
    workload2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    workload3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")

    trace21 = Trace(
        definition="add", workload=workload2, solution="add_fast", evaluation=make_eval(3.0)
    )
    trace22 = Trace(
        definition="add", workload=workload2, solution="add_slow", evaluation=make_eval(1.2)
    )
    trace31 = Trace(
        definition="add", workload=workload3, solution="add_fast", evaluation=make_eval(0.9)
    )
    trace32 = Trace(
        definition="add", workload=workload3, solution="add_slow", evaluation=make_eval(2.5)
    )

    return definition, [solution1, solution2], [trace21, trace22, trace31, trace32]


def test_apply_table_build_and_match(tmp_path, monkeypatch):
    # Route caches (apply table + python builder) to test tmp dir
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    definition, solutions, traces = make_traces()
    trace_set = TraceSet(
        root=tmp_path,
        definitions={"add": definition},
        solutions={"add": solutions},
        traces={"add": traces},
    )

    config = ApplyConfig(aot_ratio=1.0)
    table = ApplyTable.load_or_build(trace_set, config)

    # Build keys for lookup
    builder = ApplyKeyFactory.specialize(definition)
    key2 = builder.build_from_args((FakeTensor((2, 2)), FakeTensor((2, 2))))
    key3 = builder.build_from_args((FakeTensor((3, 2)), FakeTensor((3, 2))))

    assert table.match_solution("add", key2) == "add_fast"
    assert table.match_solution("add", key3) == "add_slow"

    # Ensure digest is a stable hex string
    assert isinstance(table.digest, str) and len(table.digest) >= 16


def test_apply_table_persistent_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create a single-def dataset on disk
    (dataset_dir / "definitions").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "traces").mkdir(parents=True, exist_ok=True)

    definition = Definition(
        name="add",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X\n",
    )
    from flashinfer_bench.data import save_json_file, save_jsonl_file

    save_json_file(definition, dataset_dir / "definitions" / "add.json")

    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    solution_fast = Solution(
        name="add_fast",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    solution_slow = Solution(
        name="add_slow",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )
    save_json_file(solution_fast, dataset_dir / "solutions" / "add_fast.json")
    save_json_file(solution_slow, dataset_dir / "solutions" / "add_slow.json")

    env = Environment(hardware="cpu")

    def make_evaluation(speedup: float) -> Evaluation:
        return Evaluation(
            status=EvaluationStatus.PASSED,
            log="log",
            environment=env,
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(
                latency_ms=1.0 / max(speedup, 1e-6),
                reference_latency_ms=1.0,
                speedup_factor=speedup,
            ),
        )

    workload2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    workload3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")
    traces = [
        Trace(
            definition="add",
            workload=workload2,
            solution="add_fast",
            evaluation=make_evaluation(3.0),
        ),
        Trace(
            definition="add",
            workload=workload2,
            solution="add_slow",
            evaluation=make_evaluation(1.0),
        ),
        Trace(
            definition="add",
            workload=workload3,
            solution="add_fast",
            evaluation=make_evaluation(0.9),
        ),
        Trace(
            definition="add",
            workload=workload3,
            solution="add_slow",
            evaluation=make_evaluation(2.0),
        ),
    ]
    save_jsonl_file(traces, dataset_dir / "traces" / "add.jsonl")

    config = ApplyConfig(aot_ratio=0.0, on_miss_policy="use_def_best")
    trace_set = TraceSet.from_path(str(dataset_dir))
    table1 = ApplyTable.load_or_build(trace_set, config)

    # Check persisted index file
    apply_dir = _apply_table_dir()
    digest = table1.digest
    index_path = apply_dir / f"{digest}.json"
    assert index_path.exists()

    raw = json.loads(index_path.read_text())
    assert raw.get("digest") == digest
    # Ensure we have mapping for two keys and both solutions appear as winners
    idx = raw.get("index", {}).get("add", {})
    assert isinstance(idx, dict) and len(idx) == 2
    assert set(idx.values()) == {"add_fast", "add_slow"}
    # def_best recorded
    assert raw.get("def_best", {}).get("add") in {"add_fast", "add_slow"}

    # Ensure second call uses persisted file rather than _build
    def fail_build(*args, **kwargs):
        raise AssertionError("_build should not be called on cache hit")

    orig_build = ApplyTable._build
    try:
        ApplyTable._build = fail_build  # type: ignore[assignment]
        table2 = ApplyTable.load_or_build(trace_set, config)
    finally:
        ApplyTable._build = orig_build  # type: ignore[assignment]

    assert table2.digest == table1.digest


if __name__ == "__main__":
    pytest.main(sys.argv)
