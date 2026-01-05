import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import pytest
import torch

from flashinfer_bench.data import AxisVar, Definition, TensorSpec, TraceSet
from flashinfer_bench.tracing import (
    TracingConfig,
    TracingConfigRegistry,
    TracingRuntime,
    WorkloadEntry,
)


@pytest.fixture
def minimal_trace_set(tmp_path: Path) -> TraceSet:
    """Create a minimal TraceSet for testing."""
    definitions = {
        "def1": Definition(
            name="def1",
            op_type="op",
            axes={"n": AxisVar()},
            inputs={"x": TensorSpec(shape=["n"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["n"], dtype="float32")},
            reference="def run(x, y):\n    return x\n",
        ),
        "def2": Definition(
            name="def2",
            op_type="op",
            axes={"m": AxisVar()},
            inputs={"a": TensorSpec(shape=["m"], dtype="float32")},
            outputs={"b": TensorSpec(shape=["m"], dtype="float32")},
            reference="def run(a, b):\n    return a\n",
        ),
    }
    return TraceSet(root=str(tmp_path), definitions=definitions, traces=[])


@pytest.fixture(autouse=True)
def reset_runtime_state():
    """Reset TracingRuntime class state before and after each test."""
    TracingRuntime._stack = []
    TracingRuntime._cleanup_registered = False
    TracingRuntime._env_initialized = False
    yield
    TracingRuntime._stack = []
    TracingRuntime._cleanup_registered = False
    TracingRuntime._env_initialized = False


def test_runtime_lifecycle(minimal_trace_set: TraceSet):
    """Test runtime lifecycle: start/stop, stack, nested runtimes, context manager."""
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 2}
    )

    # Lazy init returns None without env vars
    assert TracingRuntime.get_instance() is None
    assert TracingRuntime._env_initialized is True

    # Start/stop and stack operations
    rt1 = TracingRuntime(minimal_trace_set, config)
    rt1.start()
    assert TracingRuntime._stack == [rt1]
    assert TracingRuntime.get_instance() is rt1
    assert TracingRuntime._cleanup_registered is True

    # Nested runtime
    rt2 = TracingRuntime(minimal_trace_set, config)
    rt2.start()
    assert TracingRuntime._stack == [rt1, rt2]
    assert TracingRuntime.get_instance() is rt2

    # Stop in wrong order raises
    with pytest.raises(RuntimeError, match="LIFO"):
        rt1.stop()

    # Correct LIFO order
    rt2.stop()
    assert TracingRuntime.get_instance() is rt1
    rt1.stop()
    assert TracingRuntime.get_instance() is None

    # Context manager
    rt3 = TracingRuntime(minimal_trace_set, config)
    with rt3:
        assert TracingRuntime.get_instance() is rt3
    assert TracingRuntime.get_instance() is None


def test_runtime_multiple_start(minimal_trace_set: TraceSet):
    """Test start() idempotency and reactivation (stack re-push)."""
    config = TracingConfig(input_dump_policy=[], filter_policy="keep_all")
    rt1 = TracingRuntime(minimal_trace_set, config)
    rt2 = TracingRuntime(minimal_trace_set, config)

    # Idempotent: multiple start() on active runtime has no effect
    rt1.start()
    rt1.start()
    assert TracingRuntime._stack == [rt1]

    # Reactivate: start() on inactive runtime pushes it again
    rt2.start()
    rt1.start()
    assert TracingRuntime._stack == [rt1, rt2, rt1]


def test_policy_isolation(minimal_trace_set: TraceSet):
    """Test filter policies are isolated between definitions and runtimes."""
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 2}
    )

    # Different definitions get independent policies
    registry = TracingConfigRegistry(per_definition={"def1": config, "def2": config})
    rt1 = TracingRuntime(minimal_trace_set, registry)
    assert rt1._filter_policies["def1"] is not rt1._filter_policies["def2"]

    # Different runtimes sharing same config get independent policies
    rt2 = TracingRuntime(minimal_trace_set, registry)
    assert rt1._filter_policies["def1"] is not rt2._filter_policies["def1"]

    # State isolation: submit to rt1 doesn't affect rt2
    entry = WorkloadEntry(def_name="def1", axes={"n": 10}, inputs_to_dump={}, order=0)
    rt1._filter_policies["def1"].submit(entry)
    assert len(rt1._filter_policies["def1"].entries) == 1
    assert len(rt2._filter_policies["def1"].entries) == 0

    # Default config pre-allocates policies for all definitions
    default_registry = TracingConfigRegistry(default=config)
    rt3 = TracingRuntime(minimal_trace_set, default_registry)
    assert "def1" in rt3._filter_policies and "def2" in rt3._filter_policies


def test_collect_arguments(minimal_trace_set: TraceSet):
    """Test collect with various argument styles."""
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 10}
    )
    registry = TracingConfigRegistry(per_definition={"def1": config})
    runtime = TracingRuntime(minimal_trace_set, registry)

    # Positional args
    runtime.collect("def1", args=(torch.zeros(5),), kwargs={})
    assert runtime._filter_policies["def1"].entries[-1].axes == {"n": 5}

    # Kwargs only
    runtime.collect("def1", args=(), kwargs={"x": torch.zeros(7)})
    assert runtime._filter_policies["def1"].entries[-1].axes == {"n": 7}

    # DPS style: 2 args = input + output, only input traced
    runtime.collect("def1", args=(torch.zeros(12), torch.zeros(12)), kwargs={})
    assert runtime._filter_policies["def1"].entries[-1].axes == {"n": 12}

    assert len(runtime._filter_policies["def1"].entries) == 3


def test_online_dedup(minimal_trace_set: TraceSet):
    """Test online deduplication via filter policy."""
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 2}
    )
    registry = TracingConfigRegistry(per_definition={"def1": config})
    runtime = TracingRuntime(minimal_trace_set, registry)

    # Collect 3, but policy keeps only first 2
    for _ in range(3):
        runtime.collect("def1", args=(torch.zeros(10),), kwargs={})

    policy = runtime._filter_policies["def1"]
    assert len(policy.entries) == 2

    # Drain clears entries
    selected = policy.drain()
    assert len(selected) == 2
    assert len(policy.entries) == 0


def test_error_handling(minimal_trace_set: TraceSet):
    """Test error handling for invalid args and unconfigured definitions."""
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 10}
    )
    registry = TracingConfigRegistry(per_definition={"def1": config})
    runtime = TracingRuntime(minimal_trace_set, registry)

    # Invalid arg count: expects 1 (VR) or 2 (DPS), got 3
    runtime.collect("def1", args=(torch.zeros(10),) * 3, kwargs={})
    assert len(runtime._filter_policies["def1"].entries) == 0

    # Unconfigured definition is skipped
    runtime.collect("def2", args=(torch.zeros(10),), kwargs={})
    assert "def2" not in runtime._filter_policies


def test_end_to_end(tmp_path: Path):
    """Test end-to-end tracing and flushing."""
    definitions = {
        "def1": Definition(
            name="def1",
            op_type="op",
            axes={"n": AxisVar()},
            inputs={"x": TensorSpec(shape=["n"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["n"], dtype="float32")},
            reference="def run(x, y):\n    return x\n",
        )
    }
    trace_set = TraceSet(root=tmp_path, definitions=definitions)
    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 10}
    )

    with TracingRuntime(trace_set, config) as runtime:
        runtime.collect("def1", args=(torch.zeros(10),), kwargs={})
        runtime.collect("def1", args=(torch.zeros(20),), kwargs={})

    # Verify data was flushed to disk
    trace_set = TraceSet.from_path(tmp_path)
    assert "def1" in trace_set.workloads
    assert len(trace_set.workloads["def1"]) == 2


def _signal_test_worker(tmp_path: str, ready_event: multiprocessing.Event):
    """Child process worker for signal cleanup test."""
    definitions = {
        "def1": Definition(
            name="def1",
            op_type="op",
            axes={"n": AxisVar()},
            inputs={"x": TensorSpec(shape=["n"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["n"], dtype="float32")},
            reference="def run(x, y):\n    return x\n",
        )
    }
    trace_set = TraceSet(root=Path(tmp_path), definitions=definitions)

    config = TracingConfig(
        input_dump_policy=[], filter_policy="keep_first_k", filter_policy_kwargs={"k": 10}
    )
    runtime = TracingRuntime(trace_set, config)
    runtime.start()
    runtime.collect("def1", args=(torch.zeros(10),), kwargs={})

    ready_event.set()
    time.sleep(60)


def test_signal_cleanup(tmp_path: Path):
    """Test SIGTERM triggers cleanup and flushes data to disk."""
    ready_event = multiprocessing.Event()
    p = multiprocessing.Process(target=_signal_test_worker, args=(str(tmp_path), ready_event))
    p.start()

    assert ready_event.wait(timeout=30), "Child process did not become ready"
    os.kill(p.pid, signal.SIGTERM)
    p.join(timeout=5)

    # Verify data was flushed to disk (workloads, not traces)
    trace_set = TraceSet.from_path(tmp_path)
    assert "def1" in trace_set.workloads
    assert len(trace_set.workloads["def1"]) == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
