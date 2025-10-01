import sys
from pathlib import Path

import pytest

from flashinfer_bench.data import AxisVar, Definition, TensorSpec, TraceSet
from flashinfer_bench.tracing import KeepFirstKPolicy, TracingConfig, TracingRuntime, WorkloadEntry


@pytest.fixture
def minimal_traceset(tmp_path: Path) -> TraceSet:
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


def test_runtime_creates_independent_policies_per_definition(minimal_traceset: TraceSet):
    """Test that each definition gets an independent policy instance."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config, "def2": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    policy1 = runtime._dedup_policies["def1"]
    policy2 = runtime._dedup_policies["def2"]

    assert policy1 is not policy2
    assert policy1.k == 2
    assert policy2.k == 2


def test_runtime_policy_state_isolation(minimal_traceset: TraceSet):
    """Test that policies for different definitions have isolated state."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config, "def2": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    # Manually submit entries to policies
    entry1 = WorkloadEntry(def_name="def1", axes={"n": 10}, tensors_to_dump={}, order=0)
    entry2 = WorkloadEntry(def_name="def2", axes={"m": 20}, tensors_to_dump={}, order=1)

    runtime._dedup_policies["def1"].submit(entry1)
    runtime._dedup_policies["def1"].submit(entry1)
    runtime._dedup_policies["def2"].submit(entry2)

    assert len(runtime._dedup_policies["def1"].entries) == 2
    assert len(runtime._dedup_policies["def2"].entries) == 1


def test_multiple_runtimes_share_config_safely(minimal_traceset: TraceSet):
    """Test that multiple runtimes can share the same TracingConfig without conflicts."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config}

    runtime1 = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)
    runtime2 = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    policy1 = runtime1._dedup_policies["def1"]
    policy2 = runtime2._dedup_policies["def1"]

    assert policy1 is not policy2

    # Verify state isolation
    entry = WorkloadEntry(def_name="def1", axes={"n": 10}, tensors_to_dump={}, order=0)
    policy1.submit(entry)

    assert len(policy1.entries) == 1
    assert len(policy2.entries) == 0


def test_online_deduplication_on_collect(minimal_traceset: TraceSet):
    """Test that entries are submitted to dedup policy immediately during collect."""
    import torch

    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    # Collect 3 entries, but policy only keeps first 2
    for i in range(3):
        runtime.collect("def1", {"x": torch.zeros(10)})

    # Verify policy has exactly 2 entries
    policy = runtime._dedup_policies["def1"]
    assert len(policy.entries) == 2

    # Flush and verify drain works
    selected = policy.drain()
    assert len(selected) == 2
    assert len(policy.entries) == 0  # After drain, entries are cleared


if __name__ == "__main__":
    pytest.main(sys.argv)
