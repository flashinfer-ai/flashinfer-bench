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
    d = make_minimal_def()
    s1 = make_python_solution("add_fast")
    s2 = make_python_solution("add_slow")

    # Two keys: M=2 best s1, M=3 best s2
    wl2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    wl3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")

    t21 = Trace(definition="add", workload=wl2, solution="add_fast", evaluation=make_eval(3.0))
    t22 = Trace(definition="add", workload=wl2, solution="add_slow", evaluation=make_eval(1.2))
    t31 = Trace(definition="add", workload=wl3, solution="add_fast", evaluation=make_eval(0.9))
    t32 = Trace(definition="add", workload=wl3, solution="add_slow", evaluation=make_eval(2.5))

    return d, [s1, s2], [t21, t22, t31, t32]


def test_apply_table_build_and_match(tmp_path, monkeypatch):
    # Route caches (apply table + python builder) to test tmp dir
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    d, sols, traces = make_traces()
    ts = TraceSet(
        root=tmp_path, definitions={"add": d}, solutions={"add": sols}, traces={"add": traces}
    )

    cfg = ApplyConfig(aot_ratio=1.0)
    table = ApplyTable.load_or_build(ts, cfg)

    # Build keys for lookup
    builder = ApplyKeyFactory.specialize(d)
    k2 = builder.build_from_args((FakeTensor((2, 2)), FakeTensor((2, 2))))
    k3 = builder.build_from_args((FakeTensor((3, 2)), FakeTensor((3, 2))))

    assert table.match_solution("add", k2) == "add_fast"
    assert table.match_solution("add", k3) == "add_slow"

    # Ensure digest is a stable hex string
    assert isinstance(table.digest, str) and len(table.digest) >= 16


def test_apply_table_persistent_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    ds = tmp_path / "ds"
    ds.mkdir(parents=True, exist_ok=True)

    # Create a single-def dataset on disk
    (ds / "definitions").mkdir(parents=True, exist_ok=True)
    (ds / "solutions").mkdir(parents=True, exist_ok=True)
    (ds / "traces").mkdir(parents=True, exist_ok=True)

    d = Definition(
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

    save_json_file(d, ds / "definitions" / "add.json")

    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages

    s_fast = Solution(
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
    s_slow = Solution(
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
    save_json_file(s_fast, ds / "solutions" / "add_fast.json")
    save_json_file(s_slow, ds / "solutions" / "add_slow.json")

    env = Environment(hardware="cpu")

    def ev(sp: float) -> Evaluation:
        return Evaluation(
            status=EvaluationStatus.PASSED,
            log="log",
            environment=env,
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(
                latency_ms=1.0 / max(sp, 1e-6), reference_latency_ms=1.0, speedup_factor=sp
            ),
        )

    wl2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    wl3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")
    traces = [
        Trace(definition="add", workload=wl2, solution="add_fast", evaluation=ev(3.0)),
        Trace(definition="add", workload=wl2, solution="add_slow", evaluation=ev(1.0)),
        Trace(definition="add", workload=wl3, solution="add_fast", evaluation=ev(0.9)),
        Trace(definition="add", workload=wl3, solution="add_slow", evaluation=ev(2.0)),
    ]
    save_jsonl_file(traces, ds / "traces" / "add.jsonl")

    cfg = ApplyConfig(aot_ratio=0.0, on_miss_policy="use_def_best")
    ts = TraceSet.from_path(str(ds))
    table1 = ApplyTable.load_or_build(ts, cfg)

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
        table2 = ApplyTable.load_or_build(ts, cfg)
    finally:
        ApplyTable._build = orig_build  # type: ignore[assignment]

    assert table2.digest == table1.digest


if __name__ == "__main__":
    pytest.main(sys.argv)
