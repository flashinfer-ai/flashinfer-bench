import sys

import pytest
import torch

from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata
from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec


def _make_definition() -> Definition:
    """Create a simple definition for testing."""
    return Definition(
        name="test_op",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(A, B):\n    return A + B\n",
    )


def test_runnable_single_tuple_unpack_and_close_idempotent():
    calls = {"closed": 0}

    def fn(*args):
        return (42,)

    def closer():
        calls["closed"] += 1

    metadata = RunnableMetadata(
        build_type="python", definition_name="test", solution_name="test", misc={"k": 1}
    )

    r = Runnable(callable=fn, cleaner=closer, metadata=metadata)
    assert r() == 42
    # Close twice should not error and closer should be called once
    r.cleanup()
    r.cleanup()
    r.cleanup()
    assert calls["closed"] == 1


def test_runnable_call_with_positional_args():
    """Test that __call__ works with positional arguments."""

    def fn(*args):
        return args[0] + args[1]

    metadata = RunnableMetadata(build_type="python", definition_name="test", solution_name="test")
    r = Runnable(callable=fn, metadata=metadata)

    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = r(a, b)
    assert torch.allclose(result, torch.tensor([4.0, 6.0]))


def test_runnable_revise_return_value():
    """Test _revise_return_value unpacking behavior."""
    metadata = RunnableMetadata(build_type="python", definition_name="test", solution_name="test")

    # Empty tuple -> None
    r = Runnable(callable=lambda: (), metadata=metadata)
    assert r() is None

    # Single element tuple -> unpacked
    r = Runnable(callable=lambda: (42,), metadata=metadata)
    assert r() == 42

    # Multi element tuple -> unchanged
    r = Runnable(callable=lambda: (1, 2, 3), metadata=metadata)
    assert r() == (1, 2, 3)

    # Non-tuple -> unchanged
    r = Runnable(callable=lambda: 42, metadata=metadata)
    assert r() == 42


class TestCallDestinationPassing:
    """Tests for call_destination_passing method."""

    def test_native_dps_callable(self):
        """Test calling a native DPS callable directly."""
        definition = _make_definition()

        def dps_fn(A, B, C):
            C.copy_(A + B)

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=dps_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        C = torch.zeros((1, 4))

        r.call_destination_passing(A, B, C)
        assert torch.allclose(C, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))

    def test_convert_vr_to_dps(self):
        """Test converting a value-returning callable to DPS style."""
        definition = _make_definition()

        def vr_fn(A, B):
            return A + B

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=False,
            definition=definition,
        )
        r = Runnable(callable=vr_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        C = torch.zeros((1, 4))

        r.call_destination_passing(A, B, C)
        assert torch.allclose(C, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))

    def test_convert_vr_to_dps_no_outputs(self):
        """Test VR to DPS conversion when there are no outputs."""
        definition = Definition(
            name="no_output_op",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={},
            reference="def run(A):\n    return ()\n",
        )

        def vr_fn(A):
            return ()

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="no_output_op",
            solution_name="test",
            destination_passing_style=False,
            definition=definition,
        )
        r = Runnable(callable=vr_fn, metadata=metadata)

        A = torch.tensor([1.0, 2.0])
        # Should not raise
        r.call_destination_passing(A)


class TestCallValueReturning:
    """Tests for call_value_returning method."""

    def test_native_vr_callable(self):
        """Test calling a native VR callable directly."""
        definition = _make_definition()

        def vr_fn(A, B):
            return A + B

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=False,
            definition=definition,
        )
        r = Runnable(callable=vr_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        result = r.call_value_returning(A, B)
        assert torch.allclose(result, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))

    def test_convert_dps_to_vr(self):
        """Test converting a DPS callable to value-returning style."""
        definition = _make_definition()

        def dps_fn(A, B, C):
            C.copy_(A + B)

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=dps_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        result = r.call_value_returning(A, B)
        assert torch.allclose(result, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))

    def test_convert_dps_to_vr_multiple_outputs(self):
        """Test DPS to VR conversion with multiple outputs."""
        definition = Definition(
            name="multi_output_op",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={
                "B": TensorSpec(shape=["M"], dtype="float32"),
                "C": TensorSpec(shape=["M"], dtype="float32"),
            },
            reference="def run(A):\n    return A, A * 2\n",
        )

        def dps_fn(A, B, C):
            B.copy_(A)
            C.copy_(A * 2)

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="multi_output_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=dps_fn, metadata=metadata)

        A = torch.tensor([1.0, 2.0, 3.0])
        B, C = r.call_value_returning(A)

        assert torch.allclose(B, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(C, torch.tensor([2.0, 4.0, 6.0]))


class TestCallKwargs:
    """Tests for call_kwargs method."""

    def test_call_kwargs_dps(self):
        """Test calling with kwargs in DPS style."""
        definition = _make_definition()

        def dps_fn(A, B, C):
            C.copy_(A + B)

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=dps_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        C = torch.zeros((1, 4))

        r.call_kwargs(A=A, B=B, C=C)
        assert torch.allclose(C, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))

    def test_call_kwargs_vr(self):
        """Test calling with kwargs in VR style."""
        definition = _make_definition()

        def vr_fn(A, B):
            return A + B

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=False,
            definition=definition,
        )
        r = Runnable(callable=vr_fn, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        result = r.call_kwargs(A=A, B=B)
        assert torch.allclose(result, torch.tensor([[6.0, 8.0, 10.0, 12.0]]))


class TestAllocateOutputTensors:
    """Tests for _allocate_output_tensors method."""

    def test_allocate_output_tensors(self):
        """Test output tensor allocation with variable axes."""
        definition = _make_definition()

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=lambda *args: None, metadata=metadata)

        A = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # shape (2, 4)
        B = torch.zeros((2, 4))

        outputs = r._allocate_output_tensors(A, B)
        assert len(outputs) == 1
        assert outputs[0].shape == (2, 4)
        assert outputs[0].dtype == torch.float32

    def test_allocate_output_tensors_multiple(self):
        """Test allocation of multiple output tensors."""
        definition = Definition(
            name="multi_output_op",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={
                "B": TensorSpec(shape=["M"], dtype="float32"),
                "C": TensorSpec(shape=["M"], dtype="int32"),
            },
            reference="def run(A):\n    return A, A.int()\n",
        )

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="multi_output_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=lambda *args: None, metadata=metadata)

        A = torch.tensor([1.0, 2.0, 3.0])
        outputs = r._allocate_output_tensors(A)

        assert len(outputs) == 2
        assert outputs[0].shape == (3,)
        assert outputs[0].dtype == torch.float32
        assert outputs[1].shape == (3,)
        assert outputs[1].dtype == torch.int32


class TestSetupHook:
    """Tests for the optional per-workload setup() hook (ab70330)."""

    def test_no_setup_callable_keeps_state_none(self):
        """If no setup_callable is provided, _workload_state stays None and __call__ goes the original path."""

        def fn(a, b):
            return a + b

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=fn, metadata=metadata)
        assert r._setup_callable is None
        assert r._workload_state is None

        # setup_for_workload on a no-setup runnable is a no-op
        r.setup_for_workload(1, 2)
        assert r._workload_state is None

        # __call__ ignores the (absent) state
        assert r(1, 2) == 3

    def test_setup_returns_dict_state_injected_as_kwargs(self):
        """setup() returning a dict caches the state and run() receives it as kwargs."""
        seen_kwargs = {}

        def setup(a, b):
            return {"derived": a * 10}

        def run(a, b, *, derived):
            seen_kwargs["derived"] = derived
            return a + b + derived

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=run, metadata=metadata, setup_callable=setup)
        r.setup_for_workload(2, 3)
        assert r._workload_state == {"derived": 20}

        result = r(2, 3)
        assert result == 25
        assert seen_kwargs == {"derived": 20}

    def test_setup_returns_none_skips_kwargs_splat(self):
        """setup() returning None caches an empty dict and run() is called without state kwargs."""

        def setup(a):
            return None

        def run(a):
            return a + 1

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=run, metadata=metadata, setup_callable=setup)
        r.setup_for_workload(5)
        assert r._workload_state == {}
        assert r(5) == 6  # no kwargs splat

    def test_setup_hook_call_before_setup_raises(self):
        """A setup-hook runnable invoked before setup_for_workload must fail
        loudly. Silently running would use default kwarg values — misleading
        results, and an evaluator that skips setup could time run() against
        state it never built (the item-1 gaming vector through a side door)."""

        def setup(a):
            return {"bias": 1}

        def run(a, *, bias=0):  # default makes the silent path tempting
            return a + bias

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=run, metadata=metadata, setup_callable=setup)
        with pytest.raises(RuntimeError, match="setup_for_workload"):
            r(1)

        # After setup, the call works and sees the real state.
        r.setup_for_workload(1)
        assert r(1) == 2

    def test_setup_returns_non_dict_raises_type_error(self):
        """setup() returning a non-dict (e.g. tuple) should raise TypeError on setup_for_workload."""

        def setup(a):
            return (a, a * 2)  # not a dict

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=lambda *a: None, metadata=metadata, setup_callable=setup)
        with pytest.raises(TypeError, match="setup\\(\\) must return a dict"):
            r.setup_for_workload(1)

    def test_setup_state_persists_across_calls(self):
        """setup_for_workload is called once; subsequent __call__ all see the same state."""
        setup_call_count = {"n": 0}

        def setup(a):
            setup_call_count["n"] += 1
            return {"k": setup_call_count["n"]}

        def run(a, *, k):
            return a + k

        metadata = RunnableMetadata(
            build_type="python", definition_name="test", solution_name="test"
        )
        r = Runnable(callable=run, metadata=metadata, setup_callable=setup)

        r.setup_for_workload(10)
        assert r(10) == 11  # k=1 from setup
        assert r(10) == 11  # state reused, no extra setup call
        assert setup_call_count["n"] == 1

        # Re-run setup_for_workload re-invokes setup
        r.setup_for_workload(20)
        assert setup_call_count["n"] == 2
        assert r(20) == 22  # k=2 now

    def test_setup_state_injected_in_dps_call(self):
        """State kwargs are splatted into call_destination_passing for native DPS callables."""
        definition = _make_definition()

        def setup(A, B, C):
            return {"scale": 10.0}

        def dps_fn(A, B, C, *, scale):
            C.copy_((A + B) * scale)

        metadata = RunnableMetadata(
            build_type="python",
            definition_name="test_op",
            solution_name="test",
            destination_passing_style=True,
            definition=definition,
        )
        r = Runnable(callable=dps_fn, metadata=metadata, setup_callable=setup)

        A = torch.ones((3, 4), dtype=torch.float32)
        B = torch.ones((3, 4), dtype=torch.float32) * 2
        C = torch.zeros((3, 4), dtype=torch.float32)
        r.setup_for_workload(A, B, C)
        r.call_destination_passing(A, B, C)
        assert torch.allclose(C, torch.ones((3, 4)) * 30.0)


if __name__ == "__main__":
    pytest.main(sys.argv)
