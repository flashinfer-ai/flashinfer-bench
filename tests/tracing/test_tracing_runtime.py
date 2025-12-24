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
    config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config, "def2": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    policy1 = runtime._filter_policies["def1"]
    policy2 = runtime._filter_policies["def2"]

    assert policy1 is not policy2
    assert policy1.k == 2
    assert policy2.k == 2


def test_runtime_policy_state_isolation(minimal_traceset: TraceSet):
    """Test that policies for different definitions have isolated state."""
    config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config, "def2": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    # Manually submit entries to policies
    entry1 = WorkloadEntry(def_name="def1", axes={"n": 10}, inputs_to_dump={}, order=0)
    entry2 = WorkloadEntry(def_name="def2", axes={"m": 20}, inputs_to_dump={}, order=1)

    runtime._filter_policies["def1"].submit(entry1)
    runtime._filter_policies["def1"].submit(entry1)
    runtime._filter_policies["def2"].submit(entry2)

    assert len(runtime._filter_policies["def1"].entries) == 2
    assert len(runtime._filter_policies["def2"].entries) == 1


def test_multiple_runtimes_share_config_safely(minimal_traceset: TraceSet):
    """Test that multiple runtimes can share the same TracingConfig without conflicts."""
    config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config}

    runtime1 = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)
    runtime2 = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    policy1 = runtime1._filter_policies["def1"]
    policy2 = runtime2._filter_policies["def1"]

    assert policy1 is not policy2

    # Verify state isolation
    entry = WorkloadEntry(def_name="def1", axes={"n": 10}, inputs_to_dump={}, order=0)
    policy1.submit(entry)

    assert len(policy1.entries) == 1
    assert len(policy2.entries) == 0


def test_online_deduplication_on_collect(minimal_traceset: TraceSet):
    """Test that entries are submitted to filter policy immediately during collect."""
    import torch

    config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=2))
    tracing_configs = {"def1": config}

    runtime = TracingRuntime(minimal_traceset, tracing_configs, prev_tracing_runtime=None)

    # Collect 3 entries, but policy only keeps first 2
    # Using new args/kwargs signature
    for i in range(3):
        runtime.collect("def1", args=(torch.zeros(10),), kwargs={})

    # Verify policy has exactly 2 entries
    policy = runtime._filter_policies["def1"]
    assert len(policy.entries) == 2

    # Flush and verify drain works
    selected = policy.drain()
    assert len(selected) == 2
    assert len(policy.entries) == 0  # After drain, entries are cleared


class TestCollectArgsKwargs:
    """Tests for collect with args/kwargs handling."""

    def test_collect_with_positional_args(self, minimal_traceset: TraceSet):
        """Test collect with positional args only."""
        import torch

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(minimal_traceset, {"def1": config}, prev_tracing_runtime=None)

        runtime.collect("def1", args=(torch.zeros(5),), kwargs={})

        policy = runtime._filter_policies["def1"]
        assert len(policy.entries) == 1
        assert policy.entries[0].axes == {"n": 5}

    def test_collect_with_kwargs(self, minimal_traceset: TraceSet):
        """Test collect with kwargs."""
        import torch

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(minimal_traceset, {"def1": config}, prev_tracing_runtime=None)

        runtime.collect("def1", args=(), kwargs={"x": torch.zeros(7)})

        policy = runtime._filter_policies["def1"]
        assert len(policy.entries) == 1
        assert policy.entries[0].axes == {"n": 7}

    def test_collect_with_mixed_args_kwargs(self, minimal_traceset: TraceSet):
        """Test collect with mixed positional and keyword arguments."""
        import torch

        # Create a definition with multiple inputs
        definitions = {
            "multi_input": Definition(
                name="multi_input",
                op_type="op",
                axes={"n": AxisVar()},
                inputs={
                    "x": TensorSpec(shape=["n"], dtype="float32"),
                    "y": TensorSpec(shape=["n"], dtype="float32"),
                },
                outputs={"z": TensorSpec(shape=["n"], dtype="float32")},
                reference="def run(x, y):\n    return x + y\n",
            )
        }
        traceset = TraceSet(root=None, definitions=definitions, traces=[])

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(traceset, {"multi_input": config}, prev_tracing_runtime=None)

        # First arg positional, second as kwarg
        runtime.collect("multi_input", args=(torch.zeros(8),), kwargs={"y": torch.zeros(8)})

        policy = runtime._filter_policies["multi_input"]
        assert len(policy.entries) == 1
        assert policy.entries[0].axes == {"n": 8}


class TestCollectCallingConvention:
    """Tests for collect with different calling conventions."""

    def test_collect_value_returning_style(self, minimal_traceset: TraceSet):
        """Test collect with value-returning style (inputs only)."""
        import torch

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(minimal_traceset, {"def1": config}, prev_tracing_runtime=None)

        # 1 arg = 1 input -> VR style
        runtime.collect("def1", args=(torch.zeros(10),), kwargs={})

        policy = runtime._filter_policies["def1"]
        assert len(policy.entries) == 1
        assert policy.entries[0].axes == {"n": 10}

    def test_collect_destination_passing_style(self, minimal_traceset: TraceSet):
        """Test collect with destination-passing style (inputs + outputs)."""
        import torch

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(minimal_traceset, {"def1": config}, prev_tracing_runtime=None)

        # 2 args = 1 input + 1 output -> DPS style
        # Only the input should be traced
        runtime.collect(
            "def1", args=(torch.zeros(12), torch.zeros(12)), kwargs={}  # input x, output y
        )

        policy = runtime._filter_policies["def1"]
        assert len(policy.entries) == 1
        # Axes should be inferred from input only
        assert policy.entries[0].axes == {"n": 12}

    def test_collect_invalid_arg_count(self, minimal_traceset: TraceSet):
        """Test collect logs error for invalid argument count."""
        import torch

        config = TracingConfig(input_dump_policy=[], filter_policy=lambda: KeepFirstKPolicy(k=10))
        runtime = TracingRuntime(minimal_traceset, {"def1": config}, prev_tracing_runtime=None)

        # 3 args when definition expects 1 input (VR) or 2 (DPS)
        # Should log error and return without adding entry
        runtime.collect("def1", args=(torch.zeros(10), torch.zeros(10), torch.zeros(10)), kwargs={})

        policy = runtime._filter_policies["def1"]
        assert len(policy.entries) == 0  # No entry added due to error


if __name__ == "__main__":
    pytest.main(sys.argv)
