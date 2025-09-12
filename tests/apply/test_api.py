import sys
import types
from pathlib import Path

import pytest

from flashinfer_bench.apply import apply as apply_api
from flashinfer_bench.apply import disable_apply, enable_apply
from flashinfer_bench.apply.runtime import set_runtime
from flashinfer_bench.data.definition import AxisConst, AxisVar, Definition, TensorSpec
from flashinfer_bench.data.json_utils import save_json_file, save_jsonl_file
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


class DummyRuntime:
    def __init__(self):
        self.calls = []

    def dispatch(self, def_name, runtime_kwargs, fallback):
        self.calls.append((def_name, dict(runtime_kwargs), fallback))
        return {
            "def_name": def_name,
            "kwargs": dict(runtime_kwargs),
            "fallback_name": getattr(fallback, "__name__", None),
        }


def teardown_module(module):
    # Ensure global runtime is cleared after this module
    disable_apply()
    set_runtime(None)


def test_apply_imperative_when_disabled_calls_fallback():
    set_runtime(None)

    def fb(**kw):
        return {"fb": True, "kw": kw}

    out = apply_api("some_def", runtime_kwargs={"x": 1}, fallback=fb)
    assert out == {"fb": True, "kw": {"x": 1}}


def test_apply_imperative_raises_without_fallback_when_disabled():
    set_runtime(None)
    with pytest.raises(RuntimeError):
        apply_api("d", runtime_kwargs={"x": 1}, fallback=None)


def test_apply_decorator_without_runtime_is_transparent(monkeypatch):
    # Ensure no env auto-init and runtime is absent
    set_runtime(None)
    monkeypatch.delenv("FIB_ENABLE_APPLY", raising=False)
    monkeypatch.delenv("FIB_DATASET_PATH", raising=False)

    @apply_api(lambda a, b: f"sum_{a}_{b}")
    def f(a, b):
        return a + b

    # No runtime installed: decorator must be a no-op
    assert f(2, 3) == 5


def test_apply_decorator_with_runtime_dispatches_and_preserves_metadata():
    rt = DummyRuntime()
    set_runtime(rt)

    @apply_api(lambda a, b: f"sum_{a}_{b}")
    def f(a, b):
        """docstring here"""
        return a + b

    out = f(7, b=9)
    # Routed to runtime
    assert out["def_name"] == "sum_7_9"
    assert out["kwargs"] == {"a": 7, "b": 9}
    # Metadata preserved
    assert f.__name__ == "f"
    assert f.__doc__ == "docstring here"
    assert isinstance(getattr(f, "__wrapped__", None), types.FunctionType)
    # cleanup
    set_runtime(None)


def test_apply_decorator_merge_conflicts_and_positional_overflow():
    rt = DummyRuntime()
    set_runtime(rt)

    @apply_api("foo")
    def g(a, b):
        return a + b

    # Too many positional args
    with pytest.raises(TypeError):
        g(1, 2, 3)  # type: ignore[misc]

    # Duplicate parameter via positional + keyword
    with pytest.raises(TypeError):
        g(1, a=2)  # type: ignore[call-arg]
    # cleanup
    set_runtime(None)


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _make_dataset(root: Path) -> None:
    (root / "definitions").mkdir(parents=True, exist_ok=True)
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    (root / "traces").mkdir(parents=True, exist_ok=True)

    # Definition: add
    add_def = Definition(
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

    # Definition: mul
    mul_def = Definition(
        name="mul",
        type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(A, B):\n    return A\n",
    )

    save_json_file(add_def, root / "definitions" / "add.json")
    save_json_file(mul_def, root / "definitions" / "mul.json")

    # Solutions for add
    add_fast = Solution(
        name="add_fast",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'fast'\n")],
    )
    add_slow = Solution(
        name="add_slow",
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return 'slow'\n")],
    )
    save_json_file(add_fast, root / "solutions" / "add_fast.json")
    save_json_file(add_slow, root / "solutions" / "add_slow.json")

    # Solutions for mul
    mul_fast = Solution(
        name="mul_fast",
        definition="mul",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(A, B):\n    return 'mulfast'\n")],
    )
    mul_slow = Solution(
        name="mul_slow",
        definition="mul",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(A, B):\n    return 'mulslow'\n")],
    )
    save_json_file(mul_fast, root / "solutions" / "mul_fast.json")
    save_json_file(mul_slow, root / "solutions" / "mul_slow.json")

    # Workloads and traces for add
    env = Environment(hardware="cpu")

    def make_eval(speedup: float) -> Evaluation:
        return Evaluation(
            status=EvaluationStatus.PASSED,
            log_file="log",
            environment=env,
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(
                latency_ms=1.0 / max(speedup, 1e-6),
                reference_latency_ms=1.0,
                speedup_factor=speedup,
            ),
        )

    # add: M=2 -> fast wins; M=3 -> slow wins
    wl_add_2 = Workload(axes={"M": 2}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="wa2")
    wl_add_3 = Workload(axes={"M": 3}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="wa3")
    add_traces = [
        Trace(definition="add", workload=wl_add_2, solution="add_fast", evaluation=make_eval(3.0)),
        Trace(definition="add", workload=wl_add_2, solution="add_slow", evaluation=make_eval(1.1)),
        Trace(definition="add", workload=wl_add_3, solution="add_fast", evaluation=make_eval(0.8)),
        Trace(definition="add", workload=wl_add_3, solution="add_slow", evaluation=make_eval(2.5)),
    ]
    save_jsonl_file(add_traces, root / "traces" / "add.jsonl")

    # mul: M=8 -> mul_fast wins; M=16 -> mul_slow wins
    wl_mul_8 = Workload(axes={"M": 8}, inputs={"A": RandomInput(), "B": RandomInput()}, uuid="wm8")
    wl_mul_16 = Workload(
        axes={"M": 16}, inputs={"A": RandomInput(), "B": RandomInput()}, uuid="wm16"
    )
    mul_traces = [
        Trace(definition="mul", workload=wl_mul_8, solution="mul_fast", evaluation=make_eval(4.0)),
        Trace(definition="mul", workload=wl_mul_8, solution="mul_slow", evaluation=make_eval(1.2)),
        Trace(definition="mul", workload=wl_mul_16, solution="mul_fast", evaluation=make_eval(0.9)),
        Trace(definition="mul", workload=wl_mul_16, solution="mul_slow", evaluation=make_eval(3.5)),
    ]
    save_jsonl_file(mul_traces, root / "traces" / "mul.jsonl")


def test_end_to_end_apply_substitution(tmp_path, monkeypatch):
    # Route caches (apply table + python builder) to test tmp dir
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_DIR", str(cache_dir))

    # Build dataset tree
    ds_root = tmp_path / "dataset"
    _make_dataset(ds_root)

    # Enable apply using dataset path
    with enable_apply(str(ds_root)):
        # Decorated functions for two definitions
        @apply_api("add")
        def add_fn(X, Y):
            return "fallback_add"

        @apply_api("mul")
        def mul_fn(A, B):
            return "fallback_mul"

        # add: M=2 -> add_fast, M=3 -> add_slow
        assert add_fn(FakeTensor((2, 2)), FakeTensor((2, 2))) == "fast"
        assert add_fn(FakeTensor((3, 2)), FakeTensor((3, 2))) == "slow"

        # mul: M=8 -> mul_fast, M=16 -> mul_slow
        assert mul_fn(FakeTensor((8, 2)), FakeTensor((8, 2))) == "mulfast"
        assert mul_fn(FakeTensor((16, 2)), FakeTensor((16, 2))) == "mulslow"


if __name__ == "__main__":
    pytest.main(sys.argv)
