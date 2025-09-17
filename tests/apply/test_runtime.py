from __future__ import annotations

import sys

import pytest

from flashinfer_bench.apply.config import ApplyConfig
from flashinfer_bench.apply.hook import set_apply_hook
from flashinfer_bench.apply.runtime import ApplyRuntime, set_runtime
from flashinfer_bench.data.definition import AxisConst, AxisVar, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Trace,
    Workload,
)
from flashinfer_bench.data.traceset import TraceSet


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def make_def_and_solutions():
    d = Definition(
        name="add",
        type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X\n",
    )
    sol1 = Solution(
        name="add_fast",
        definition="add",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    sol2 = Solution(
        name="add_slow",
        definition="add",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )
    return d, sol1, sol2


def make_eval(speedup: float) -> Evaluation:
    return Evaluation(
        status=EvaluationStatus.PASSED,
        log_file="log",
        environment=Environment(hardware="cpu"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
        performance=Performance(
            latency_ms=1.0 / max(speedup, 1e-6), reference_latency_ms=1.0, speedup_factor=speedup
        ),
    )


def make_traces():
    d, s1, s2 = make_def_and_solutions()
    wl2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    wl3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")
    t21 = Trace(definition="add", workload=wl2, solution="add_fast", evaluation=make_eval(3.0))
    t22 = Trace(definition="add", workload=wl2, solution="add_slow", evaluation=make_eval(1.2))
    t31 = Trace(definition="add", workload=wl3, solution="add_fast", evaluation=make_eval(0.9))
    t32 = Trace(definition="add", workload=wl3, solution="add_slow", evaluation=make_eval(2.5))
    ts = TraceSet(
        root=None,
        definitions={"add": d},
        solutions={"add": [s1, s2]},
        traces={"add": [t21, t22, t31, t32]},
    )
    return d, ts


def test_runtime_dispatch_hit_and_miss(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    d, ts = make_traces()
    rt = ApplyRuntime(ts, ApplyConfig(aot_ratio=1.0))
    set_runtime(rt)

    calls = []
    set_apply_hook(lambda name, kw: calls.append((name, dict(kw))))

    out = rt.dispatch(
        "add",
        {"X": FakeTensor((2, 2)), "Y": FakeTensor((2, 2))},
        fallback=lambda **_: "fallback",
    )
    # Routed to the winning solution implementation; our sources return string tags
    assert out == "fast"
    assert calls and calls[0][0] == "add"
    # Reset hook
    set_apply_hook(None)

    # Miss with fallback policy: returns fallback
    miss_out = rt.dispatch(
        "add",
        {"X": FakeTensor((99, 2)), "Y": FakeTensor((99, 2))},
        fallback=lambda **_: "fallback",
    )
    assert miss_out == "fallback"

    # Miss without fallback: error
    with pytest.raises(RuntimeError):
        rt.dispatch("add", {"X": FakeTensor((999, 2)), "Y": FakeTensor((999, 2))}, fallback=None)


def test_runtime_dispatch_def_best_policy_without_fallback(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    d, ts = make_traces()
    # Choose def_best when miss
    rt = ApplyRuntime(ts, ApplyConfig(on_miss_policy="use_def_best", aot_ratio=0.0))

    # Miss should use def_best; our setup has both sol names ranked; accept either
    out = rt.dispatch("add", {"X": FakeTensor((100, 2)), "Y": FakeTensor((100, 2))}, fallback=None)
    assert out in ("fast", "slow")


def test_runtime_dispatch_unknown_definition_uses_fallback_or_raises(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    d, ts = make_traces()
    rt = ApplyRuntime(ts, ApplyConfig())

    with pytest.raises(RuntimeError):
        rt.dispatch(
            "unknown_def", {"X": FakeTensor((2, 2)), "Y": FakeTensor((2, 2))}, fallback=None
        )

    fb_val = rt.dispatch(
        "unknown_def",
        {"X": FakeTensor((2, 2)), "Y": FakeTensor((2, 2))},
        fallback=lambda **_: "fb",
    )
    assert fb_val == "fb"


@pytest.mark.skip(reason="TODO: fix this test")
def test_runnable_cache_used_by_registry(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    ds = tmp_path / "ds"
    ds.mkdir(parents=True, exist_ok=True)

    # Create dataset on disk
    d = Definition(
        name="add",
        type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X\n",
    )
    from flashinfer_bench.data.json_utils import save_json_file, save_jsonl_file

    (ds / "definitions").mkdir(parents=True, exist_ok=True)
    (ds / "solutions").mkdir(parents=True, exist_ok=True)
    (ds / "traces").mkdir(parents=True, exist_ok=True)
    save_json_file(d, ds / "definitions" / "add.json")

    s_fast = Solution(
        name="add_fast",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    s_slow = Solution(
        name="add_slow",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )
    save_json_file(s_fast, ds / "solutions" / "add_fast.json")
    save_json_file(s_slow, ds / "solutions" / "add_slow.json")

    env = Environment(hardware="cpu")

    def ev(sp: float) -> Evaluation:
        return Evaluation(
            status=EvaluationStatus.PASSED,
            log_file="log",
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

    ts = TraceSet.from_path(str(ds))

    # Avoid AOT to simplify counting builder invocations
    rt = ApplyRuntime(ts, ApplyConfig(aot_ratio=0.0))
    set_runtime(rt)

    # Patch PythonBuilder._build to count actual builds
    from flashinfer_bench.compile.builders.python_builder import PythonBuilder

    counts = {"build": 0}
    orig_build = PythonBuilder._build

    def counting_build(self, definition, solution):
        counts["build"] += 1
        return orig_build(self, definition, solution)

    try:
        # Patch at class so the registry instance picks it up
        PythonBuilder._build = counting_build  # type: ignore[assignment]

        # Two dispatches for same key should reuse cached runnable
        class T:
            def __init__(self, shape):
                self.shape = shape

        assert (
            rt.dispatch("add", {"X": T((2, 2)), "Y": T((2, 2))}, fallback=lambda **_: "fb")
            == "fast"
        )
        assert (
            rt.dispatch("add", {"X": T((2, 2)), "Y": T((2, 2))}, fallback=lambda **_: "fb")
            == "fast"
        )
        # Only one real build should have occurred
        assert counts["build"] == 1
    finally:
        PythonBuilder._build = orig_build  # type: ignore[assignment]
        set_runtime(None)


if __name__ == "__main__":
    pytest.main(sys.argv)
