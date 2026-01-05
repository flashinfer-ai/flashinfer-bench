from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch

from flashinfer_bench.apply import ApplyConfig, ApplyConfigRegistry, ApplyRuntime
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.compile.builders.python_builder import PythonBuilder
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


@pytest.fixture(autouse=True)
def reset_runtime_state():
    """Reset ApplyRuntime class state before and after each test."""
    ApplyRuntime._stack = []
    ApplyRuntime._env_initialized = False
    yield
    ApplyRuntime._stack = []
    ApplyRuntime._env_initialized = False


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
    """Create VR-style traces: M=2 -> fast wins, M=3 -> slow wins."""
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
    solution_fast = Solution(
        name="add_fast",
        definition="add",
        author="t",
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
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )

    workload2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w2")
    workload3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="w3")

    traces = [
        Trace(definition="add", workload=workload2, solution="add_fast", evaluation=make_eval(3.0)),
        Trace(definition="add", workload=workload2, solution="add_slow", evaluation=make_eval(1.2)),
        Trace(definition="add", workload=workload3, solution="add_fast", evaluation=make_eval(0.9)),
        Trace(definition="add", workload=workload3, solution="add_slow", evaluation=make_eval(2.5)),
    ]

    trace_set = TraceSet(
        root=None,
        definitions={"add": definition},
        solutions={"add": [solution_fast, solution_slow]},
        traces={"add": traces},
    )
    return definition, trace_set


def make_dps_traces() -> Tuple[Definition, TraceSet]:
    """Create DPS-style traces: M=2 -> fast wins, M=3 -> slow wins."""
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
    solution_fast = Solution(
        name="add_dps_fast",
        definition="add_dps",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y, Z):\n    Z.fill_(1.0)\n")],
    )
    solution_slow = Solution(
        name="add_dps_slow",
        definition="add_dps",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y, Z):\n    Z.fill_(2.0)\n")],
    )

    workload2 = Workload(
        axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="dps_w2"
    )
    workload3 = Workload(
        axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="dps_w3"
    )

    traces = [
        Trace(
            definition="add_dps",
            workload=workload2,
            solution="add_dps_fast",
            evaluation=make_eval(3.0),
        ),
        Trace(
            definition="add_dps",
            workload=workload2,
            solution="add_dps_slow",
            evaluation=make_eval(1.2),
        ),
        Trace(
            definition="add_dps",
            workload=workload3,
            solution="add_dps_fast",
            evaluation=make_eval(0.9),
        ),
        Trace(
            definition="add_dps",
            workload=workload3,
            solution="add_dps_slow",
            evaluation=make_eval(2.5),
        ),
    ]

    trace_set = TraceSet(
        root=None,
        definitions={"add_dps": definition},
        solutions={"add_dps": [solution_fast, solution_slow]},
        traces={"add_dps": traces},
    )
    return definition, trace_set


def make_multi_def_trace_set() -> TraceSet:
    """Create trace set with two definitions for per-definition config tests."""
    def_add = Definition(
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
    def_mul = Definition(
        name="mul",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X * Y\n",
    )

    sol_add = Solution(
        name="add_sol",
        definition="add",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'add_result'\n")],
    )
    sol_mul = Solution(
        name="mul_sol",
        definition="mul",
        author="t",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'mul_result'\n")],
    )

    workload_add = Workload(
        axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="add_w"
    )
    workload_mul = Workload(
        axes={"M": 4}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="mul_w"
    )

    return TraceSet(
        root=None,
        definitions={"add": def_add, "mul": def_mul},
        solutions={"add": [sol_add], "mul": [sol_mul]},
        traces={
            "add": [
                Trace(
                    definition="add",
                    workload=workload_add,
                    solution="add_sol",
                    evaluation=make_eval(2.0),
                )
            ],
            "mul": [
                Trace(
                    definition="mul",
                    workload=workload_mul,
                    solution="mul_sol",
                    evaluation=make_eval(3.0),
                )
            ],
        },
    )


def test_runtime_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test runtime lifecycle: start/stop, stack, nested runtimes, context manager."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    _, trace_set = make_traces()
    config = ApplyConfigRegistry(default=ApplyConfig())

    # Lazy init returns None without env vars
    assert ApplyRuntime.get_instance() is None
    assert ApplyRuntime._env_initialized is True

    # Start/stop and stack operations
    rt1 = ApplyRuntime(trace_set, config)
    rt1.start()
    assert ApplyRuntime._stack == [rt1]
    assert ApplyRuntime.get_instance() is rt1

    # Nested runtime
    rt2 = ApplyRuntime(trace_set, config)
    rt2.start()
    assert ApplyRuntime._stack == [rt1, rt2]
    assert ApplyRuntime.get_instance() is rt2

    # Stop in wrong order raises
    with pytest.raises(RuntimeError, match="LIFO"):
        rt1.stop()

    # Correct LIFO order
    rt2.stop()
    assert ApplyRuntime.get_instance() is rt1
    rt1.stop()
    assert ApplyRuntime.get_instance() is None

    # Context manager
    rt3 = ApplyRuntime(trace_set, config)
    with rt3:
        assert ApplyRuntime.get_instance() is rt3
    assert ApplyRuntime.get_instance() is None


def test_runtime_multiple_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test start() idempotency and reactivation (stack re-push)."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    _, trace_set = make_traces()
    rt1 = ApplyRuntime(trace_set, ApplyConfigRegistry(default=ApplyConfig()))
    rt2 = ApplyRuntime(trace_set, ApplyConfigRegistry(default=ApplyConfig()))

    # Idempotent: multiple start() on active runtime has no effect
    rt1.start()
    rt1.start()
    assert ApplyRuntime._stack == [rt1]

    # Reactivate: start() on inactive runtime pushes it again
    rt2.start()
    rt1.start()
    assert ApplyRuntime._stack == [rt1, rt2, rt1]


def test_dispatch_logic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test dispatch hit/miss/unknown definition logic."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    _, trace_set = make_traces()
    runtime = ApplyRuntime(trace_set, ApplyConfigRegistry(default=ApplyConfig(aot_ratio=1.0)))

    # Hit: M=2 -> fast wins
    out = runtime.dispatch(
        "add",
        args=(torch.zeros(2, 2), torch.zeros(2, 2)),
        kwargs={},
        fallback=lambda *_: "fallback",
    )
    assert out == "fast"

    # Miss with fallback
    miss_out = runtime.dispatch(
        "add",
        args=(torch.zeros(99, 2), torch.zeros(99, 2)),
        kwargs={},
        fallback=lambda *_: "fallback",
    )
    assert miss_out == "fallback"

    # Miss without fallback -> error
    with pytest.raises(RuntimeError):
        runtime.dispatch(
            "add", args=(torch.zeros(999, 2), torch.zeros(999, 2)), kwargs={}, fallback=None
        )

    # Unknown definition with fallback
    fallback_val = runtime.dispatch(
        "unknown_def",
        args=(torch.zeros(2, 2), torch.zeros(2, 2)),
        kwargs={},
        fallback=lambda *args, **kwargs: "unknown_fallback",
    )
    assert fallback_val == "unknown_fallback"

    # Unknown definition without fallback -> error
    with pytest.raises(RuntimeError):
        runtime.dispatch(
            "unknown_def", args=(torch.zeros(2, 2), torch.zeros(2, 2)), kwargs={}, fallback=None
        )


def test_dispatch_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test dispatch with VR/DPS modes and args/kwargs combinations."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    # VR style
    _, vr_trace_set = make_traces()
    vr_runtime = ApplyRuntime(vr_trace_set, ApplyConfigRegistry(default=ApplyConfig(aot_ratio=1.0)))

    # VR: positional args
    out = vr_runtime.dispatch(
        "add",
        args=(torch.zeros(2, 2), torch.zeros(2, 2)),
        kwargs={},
        fallback=lambda *_: "fallback",
    )
    assert out == "fast"

    # VR: partial kwargs
    out = vr_runtime.dispatch(
        "add",
        args=(torch.zeros(2, 2),),
        kwargs={"Y": torch.zeros(2, 2)},
        fallback=lambda *_: "fallback",
    )
    assert out == "fast"

    # VR: all kwargs
    out = vr_runtime.dispatch(
        "add",
        args=(),
        kwargs={"X": torch.zeros(2, 2), "Y": torch.zeros(2, 2)},
        fallback=lambda *_: "fallback",
    )
    assert out == "fast"

    # VR: invalid arg count
    with pytest.raises(ValueError):
        vr_runtime.dispatch("add", args=(torch.zeros(2, 2),), kwargs={}, fallback=None)

    # DPS style
    _, dps_trace_set = make_dps_traces()
    dps_runtime = ApplyRuntime(
        dps_trace_set, ApplyConfigRegistry(default=ApplyConfig(aot_ratio=1.0))
    )

    X, Y, Z = torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2)

    # DPS: returns None, modifies output
    out = dps_runtime.dispatch("add_dps", args=(X, Y, Z), kwargs={}, fallback=lambda *_: "fallback")
    assert out is None
    assert Z[0, 0].item() == 1.0  # fast solution fills with 1.0

    # DPS: with kwargs for output
    Z2 = torch.zeros(2, 2)
    out = dps_runtime.dispatch(
        "add_dps", args=(X, Y), kwargs={"Z": Z2}, fallback=lambda *_: "fallback"
    )
    assert out is None
    assert Z2[0, 0].item() == 1.0  # fast solution fills with 1.0


def test_on_miss_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test on_miss_policy='use_def_best' behavior."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    # VR style with def_best policy
    _, trace_set = make_traces()
    registry = ApplyConfigRegistry(
        default=ApplyConfig(on_miss_policy="use_def_best", aot_ratio=0.0)
    )
    runtime = ApplyRuntime(trace_set, registry)

    # Miss should use def_best instead of fallback
    out = runtime.dispatch(
        "add", args=(torch.zeros(100, 2), torch.zeros(100, 2)), kwargs={}, fallback=None
    )
    assert out in ("fast", "slow")

    # DPS style with def_best policy
    _, dps_trace_set = make_dps_traces()
    dps_runtime = ApplyRuntime(dps_trace_set, registry)

    Z = torch.zeros(100, 2)
    out = dps_runtime.dispatch(
        "add_dps", args=(torch.zeros(100, 2), torch.zeros(100, 2), Z), kwargs={}, fallback=None
    )
    assert out is None
    assert Z[0, 0].item() in (1.0, 2.0)  # fast=1.0, slow=2.0


def test_per_definition_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test different configs for different definitions."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    trace_set = make_multi_def_trace_set()

    # "add" uses fallback_only, "mul" uses use_def_best
    registry = ApplyConfigRegistry(default=ApplyConfig(on_miss_policy="fallback_only"))
    registry.register("mul", ApplyConfig(on_miss_policy="use_def_best"))

    runtime = ApplyRuntime(trace_set, registry)

    # Miss on "add" (M=99 not traced) - should use fallback
    out_add = runtime.dispatch(
        "add",
        args=(torch.zeros(99, 2), torch.zeros(99, 2)),
        kwargs={},
        fallback=lambda *_: "add_fallback",
    )
    assert out_add == "add_fallback"

    # Miss on "mul" (M=99 not traced) - should use def_best
    out_mul = runtime.dispatch(
        "mul",
        args=(torch.zeros(99, 2), torch.zeros(99, 2)),
        kwargs={},
        fallback=lambda *_: "mul_fallback",
    )
    assert out_mul == "mul_result"


def test_build_caching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that repeated dispatches reuse cached Runnable instead of rebuilding."""
    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path / "cache"))
    (tmp_path / "cache").mkdir()

    _, trace_set = make_traces()
    runtime = ApplyRuntime(trace_set, ApplyConfigRegistry(default=ApplyConfig(aot_ratio=0.0)))

    # Clear registry cache to ensure f
    # resh build
    BuilderRegistry.get_instance()._cache.clear()

    build_count = {"n": 0}
    orig_build = PythonBuilder.build

    def counting_build(self, definition, solution):
        build_count["n"] += 1
        return orig_build(self, definition, solution)

    PythonBuilder.build = counting_build
    try:
        # Two dispatches with same key should only build once
        runtime.dispatch(
            "add", args=(torch.zeros(2, 2), torch.zeros(2, 2)), kwargs={}, fallback=None
        )
        runtime.dispatch(
            "add", args=(torch.zeros(2, 2), torch.zeros(2, 2)), kwargs={}, fallback=None
        )
        assert build_count["n"] == 1
    finally:
        PythonBuilder.build = orig_build


if __name__ == "__main__":
    pytest.main(sys.argv)
