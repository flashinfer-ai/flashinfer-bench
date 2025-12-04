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


if __name__ == "__main__":
    pytest.main(sys.argv)
