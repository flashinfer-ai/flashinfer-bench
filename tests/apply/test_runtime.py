from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest

from flashinfer_bench.apply import ApplyConfig, ApplyRuntime, set_apply_runtime
from flashinfer_bench.compile import Runnable
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
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = tuple(shape)


def make_def_and_solutions() -> Tuple[Definition, Solution, Solution]:
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
    solution1 = Solution(
        name="add_fast",
        definition="add",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,  # VR style
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    solution2 = Solution(
        name="add_slow",
        definition="add",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,  # VR style
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )
    return definition, solution1, solution2


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


def make_traces() -> Tuple[Definition, TraceSet]:
    definition, solution1, solution2 = make_def_and_solutions()
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
    trace_set = TraceSet(
        root=None,
        definitions={"add": definition},
        solutions={"add": [solution1, solution2]},
        traces={"add": [trace21, trace22, trace31, trace32]},
    )
    return definition, trace_set


def test_runtime_dispatch_hit_and_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    definition, trace_set = make_traces()
    runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))
    set_apply_runtime(runtime)

    # Test with positional args (VR style)
    out = runtime.dispatch(
        "add",
        args=(FakeTensor((2, 2)), FakeTensor((2, 2))),
        kwargs={},
        fallback=lambda *_: "fallback",
    )
    # Routed to the winning solution implementation; our sources return string tags
    assert out == "fast"

    # Miss with fallback policy: returns fallback
    miss_out = runtime.dispatch(
        "add",
        args=(FakeTensor((99, 2)), FakeTensor((99, 2))),
        kwargs={},
        fallback=lambda *_: "fallback",
    )
    assert miss_out == "fallback"

    # Miss without fallback: error
    with pytest.raises(RuntimeError):
        runtime.dispatch(
            "add", args=(FakeTensor((999, 2)), FakeTensor((999, 2))), kwargs={}, fallback=None
        )


def test_runtime_dispatch_def_best_policy_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    definition, trace_set = make_traces()
    # Choose def_best when miss
    runtime = ApplyRuntime(trace_set, ApplyConfig(on_miss_policy="use_def_best", aot_ratio=0.0))

    # Miss should use def_best; our setup has both solution names ranked; accept either
    out = runtime.dispatch(
        "add", args=(FakeTensor((100, 2)), FakeTensor((100, 2))), kwargs={}, fallback=None
    )
    assert out in ("fast", "slow")


def test_runtime_dispatch_unknown_definition_uses_fallback_or_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    definition, trace_set = make_traces()
    runtime = ApplyRuntime(trace_set, ApplyConfig())

    with pytest.raises(RuntimeError):
        runtime.dispatch(
            "unknown_def", args=(FakeTensor((2, 2)), FakeTensor((2, 2))), kwargs={}, fallback=None
        )

    fallback_val = runtime.dispatch(
        "unknown_def",
        args=(FakeTensor((2, 2)), FakeTensor((2, 2))),
        kwargs={},
        fallback=lambda *args, **kwargs: "fallback",
    )
    assert fallback_val == "fallback"


@pytest.mark.skip(reason="TODO: fix this test")
def test_runnable_cache_used_by_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset on disk
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

    (dataset_dir / "definitions").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "solutions").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "traces").mkdir(parents=True, exist_ok=True)
    save_json_file(definition, dataset_dir / "definitions" / "add.json")

    solution_fast = Solution(
        name="add_fast",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    solution_slow = Solution(
        name="add_slow",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
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

    trace_set = TraceSet.from_path(str(dataset_dir))

    # Avoid AOT to simplify counting builder invocations
    runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=0.0))
    set_apply_runtime(runtime)

    # Patch PythonBuilder._build to count actual builds
    from flashinfer_bench.compile.builders.python_builder import PythonBuilder

    counts = {"build": 0}
    orig_build = PythonBuilder.build

    def counting_build(self, definition: Definition, solution: Solution) -> Runnable:
        counts["build"] += 1
        return orig_build(self, definition, solution)

    try:
        # Patch at class so the registry instance picks it up
        PythonBuilder.build = counting_build  # type: ignore[assignment]

        # Two dispatches for same key should reuse cached runnable
        class T:
            def __init__(self, shape: Tuple[int, ...]):
                self.shape = shape

        assert (
            runtime.dispatch(
                "add", {"X": T((2, 2)), "Y": T((2, 2))}, fallback=lambda **_: "fallback"
            )
            == "fast"
        )
        assert (
            runtime.dispatch(
                "add", {"X": T((2, 2)), "Y": T((2, 2))}, fallback=lambda **_: "fallback"
            )
            == "fast"
        )
        # Only one real build should have occurred
        assert counts["build"] == 1
    finally:
        PythonBuilder.build = orig_build  # type: ignore[assignment]
        set_apply_runtime(None)


class TestDispatchArgsKwargs:
    """Tests for dispatch args/kwargs handling."""

    def test_dispatch_with_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch merges kwargs into args correctly."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        # Partial positional args with kwargs
        out = runtime.dispatch(
            "add",
            args=(FakeTensor((2, 2)),),
            kwargs={"Y": FakeTensor((2, 2))},
            fallback=lambda *_: "fallback",
        )
        assert out == "fast"

    def test_dispatch_with_all_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch with all kwargs and no positional args."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        out = runtime.dispatch(
            "add",
            args=(),
            kwargs={"X": FakeTensor((2, 2)), "Y": FakeTensor((2, 2))},
            fallback=lambda *_: "fallback",
        )
        assert out == "fast"

    def test_dispatch_invalid_arg_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch raises ValueError for invalid argument count."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        # Only 1 arg when definition expects 2 inputs (VR) or 3 (DPS)
        with pytest.raises(ValueError):
            runtime.dispatch("add", args=(FakeTensor((2, 2)),), kwargs={}, fallback=None)


class TestDispatchCallingConvention:
    """Tests for dispatch calling convention detection."""

    def test_dispatch_value_returning_style(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch with value-returning style (inputs only)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        # 2 args = 2 inputs -> VR style
        out = runtime.dispatch(
            "add",
            args=(FakeTensor((2, 2)), FakeTensor((2, 2))),
            kwargs={},
            fallback=lambda *_: "fallback",
        )
        # VR style returns the result
        assert out == "fast"


def make_dps_def_and_solutions() -> Tuple[Definition, Solution, Solution]:
    """Create definition and DPS-style solutions for testing."""
    definition = Definition(
        name="add_dps",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X + Y\n",
    )
    # DPS style solution - modifies output in place
    solution1 = Solution(
        name="add_dps_fast",
        definition="add_dps",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,  # DPS style
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y, Z):\n    Z.fill_('fast')\n")],
    )
    solution2 = Solution(
        name="add_dps_slow",
        definition="add_dps",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,  # DPS style
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y, Z):\n    Z.fill_('slow')\n")],
    )
    return definition, solution1, solution2


def make_dps_traces() -> Tuple[Definition, TraceSet]:
    """Create TraceSet with DPS-style solutions."""
    definition, solution1, solution2 = make_dps_def_and_solutions()
    workload2 = Workload(
        axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="dps_w2"
    )
    workload3 = Workload(
        axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="dps_w3"
    )
    trace21 = Trace(
        definition="add_dps", workload=workload2, solution="add_dps_fast", evaluation=make_eval(3.0)
    )
    trace22 = Trace(
        definition="add_dps", workload=workload2, solution="add_dps_slow", evaluation=make_eval(1.2)
    )
    trace31 = Trace(
        definition="add_dps", workload=workload3, solution="add_dps_fast", evaluation=make_eval(0.9)
    )
    trace32 = Trace(
        definition="add_dps", workload=workload3, solution="add_dps_slow", evaluation=make_eval(2.5)
    )
    trace_set = TraceSet(
        root=None,
        definitions={"add_dps": definition},
        solutions={"add_dps": [solution1, solution2]},
        traces={"add_dps": [trace21, trace22, trace31, trace32]},
    )
    return definition, trace_set


class FakeTensorWithFill:
    """Fake tensor that supports fill_ for DPS testing."""

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = tuple(shape)
        self.value = None

    def fill_(self, value):
        self.value = value


class TestDispatchDPSStyle:
    """Tests for dispatch with destination-passing style solutions."""

    def test_dispatch_dps_style_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch with DPS style returns None and modifies output tensor."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_dps_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        X = FakeTensorWithFill((2, 2))
        Y = FakeTensorWithFill((2, 2))
        Z = FakeTensorWithFill((2, 2))

        # 3 args = 2 inputs + 1 output -> DPS style
        out = runtime.dispatch("add_dps", args=(X, Y, Z), kwargs={}, fallback=lambda *_: "fallback")

        # DPS style returns None
        assert out is None
        # Output tensor should be modified
        assert Z.value == "fast"

    def test_dispatch_dps_with_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch DPS style with kwargs for output."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_dps_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        X = FakeTensorWithFill((2, 2))
        Y = FakeTensorWithFill((2, 2))
        Z = FakeTensorWithFill((2, 2))

        # Mix args and kwargs for DPS
        out = runtime.dispatch(
            "add_dps", args=(X, Y), kwargs={"Z": Z}, fallback=lambda *_: "fallback"
        )

        assert out is None
        assert Z.value == "fast"

    def test_dispatch_dps_miss_uses_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch DPS miss uses fallback correctly."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_dps_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(aot_ratio=1.0))

        X = FakeTensorWithFill((99, 2))  # M=99 not in traces
        Y = FakeTensorWithFill((99, 2))
        Z = FakeTensorWithFill((99, 2))

        fallback_called = {"called": False}

        def fallback(*args):
            fallback_called["called"] = True
            return "fallback_result"

        out = runtime.dispatch("add_dps", args=(X, Y, Z), kwargs={}, fallback=fallback)

        assert fallback_called["called"]
        assert out == "fallback_result"

    def test_dispatch_dps_best_policy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test dispatch DPS with def_best miss policy."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))

        definition, trace_set = make_dps_traces()
        runtime = ApplyRuntime(trace_set, ApplyConfig(on_miss_policy="use_def_best", aot_ratio=0.0))

        X = FakeTensorWithFill((100, 2))  # M=100 not in traces
        Y = FakeTensorWithFill((100, 2))
        Z = FakeTensorWithFill((100, 2))

        out = runtime.dispatch("add_dps", args=(X, Y, Z), kwargs={}, fallback=None)

        assert out is None
        assert Z.value in ("fast", "slow")


if __name__ == "__main__":
    pytest.main(sys.argv)
