"""Tests for bench/evaluators/utils.py"""

import sys

import pytest
import torch

from flashinfer_bench.bench.evaluators.utils import allocate_outputs, normalize_result
from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec


def _make_single_output_def() -> Definition:
    """Create a definition with a single output."""
    return Definition(
        name="single_output",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )


def _make_multi_output_def() -> Definition:
    """Create a definition with multiple outputs."""
    return Definition(
        name="multi_output",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={
            "B": TensorSpec(shape=["M"], dtype="float32"),
            "C": TensorSpec(shape=["M"], dtype="int32"),
        },
        reference="def run(A):\n    return A, A.int()\n",
    )


class TestAllocateOutputs:
    """Tests for allocate_outputs function."""

    def test_single_output(self):
        """Test allocating a single output tensor."""
        definition = _make_single_output_def()
        inputs = [torch.zeros((3, 4), dtype=torch.float32)]
        device = "cpu"

        outputs = allocate_outputs(definition, inputs, device)

        assert len(outputs) == 1
        assert outputs[0].shape == (3, 4)
        assert outputs[0].dtype == torch.float32
        assert str(outputs[0].device) == device

    def test_multiple_outputs(self):
        """Test allocating multiple output tensors."""
        definition = _make_multi_output_def()
        inputs = [torch.zeros((5,), dtype=torch.float32)]
        device = "cpu"

        outputs = allocate_outputs(definition, inputs, device)

        assert len(outputs) == 2
        assert outputs[0].shape == (5,)
        assert outputs[0].dtype == torch.float32
        assert outputs[1].shape == (5,)
        assert outputs[1].dtype == torch.int32

    def test_variable_axes_inferred(self):
        """Test that variable axes are correctly inferred from inputs."""
        definition = _make_single_output_def()
        inputs = [torch.zeros((7, 4), dtype=torch.float32)]  # M=7
        device = "cpu"

        outputs = allocate_outputs(definition, inputs, device)

        assert outputs[0].shape == (7, 4)  # M should be 7

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA not available")
    def test_cuda_device(self):
        """Test allocating on CUDA device."""
        definition = _make_single_output_def()
        inputs = [torch.zeros((3, 4), dtype=torch.float32)]
        device = "cuda:0"

        outputs = allocate_outputs(definition, inputs, device)

        assert str(outputs[0].device) == device


class TestNormalizeResult:
    """Tests for normalize_result function."""

    def test_single_tensor_result(self):
        """Test normalizing a single tensor result."""
        definition = _make_single_output_def()
        result = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        device = "cpu"

        normalized = normalize_result(definition, result, device)

        assert len(normalized) == 1
        assert torch.equal(normalized[0], result)

    def test_tuple_result(self):
        """Test normalizing a tuple of tensors."""
        definition = _make_multi_output_def()
        result = (torch.tensor([1.0, 2.0]), torch.tensor([3, 4], dtype=torch.int32))
        device = "cpu"

        normalized = normalize_result(definition, result, device)

        assert len(normalized) == 2
        assert torch.equal(normalized[0], result[0])
        assert torch.equal(normalized[1], result[1])

    def test_list_result(self):
        """Test normalizing a list of tensors."""
        definition = _make_multi_output_def()
        result = [torch.tensor([1.0, 2.0]), torch.tensor([3, 4], dtype=torch.int32)]
        device = "cpu"

        normalized = normalize_result(definition, result, device)

        assert len(normalized) == 2
        assert torch.equal(normalized[0], result[0])
        assert torch.equal(normalized[1], result[1])

    def test_int_result(self):
        """Test normalizing an int result."""
        definition = Definition(
            name="int_output",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=None, dtype="int32")},
            reference="def run(A):\n    return len(A)\n",
        )
        result = 10
        device = "cpu"

        normalized = normalize_result(definition, result, device)

        assert len(normalized) == 1
        assert normalized[0].item() == 10
        assert normalized[0].dtype == torch.int32

    def test_wrong_tuple_length_raises(self):
        """Test that wrong tuple length raises ValueError."""
        definition = _make_multi_output_def()
        result = (torch.tensor([1.0]),)  # Only 1 element, but 2 expected
        device = "cpu"

        with pytest.raises(ValueError, match="2 outputs expected"):
            normalize_result(definition, result, device)

    def test_single_value_for_multi_output_raises(self):
        """Test that single value for multi-output definition raises ValueError."""
        definition = _make_multi_output_def()
        result = torch.tensor([1.0, 2.0])  # Single tensor, but 2 expected
        device = "cpu"

        with pytest.raises(ValueError, match="2 outputs expected"):
            normalize_result(definition, result, device)

    @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA not available")
    def test_moves_tensor_to_device(self):
        """Test that result tensors are moved to the specified device."""
        definition = _make_single_output_def()
        result = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # CPU tensor
        device = "cuda:0"

        normalized = normalize_result(definition, result, device)

        assert str(normalized[0].device) == device


if __name__ == "__main__":
    pytest.main(sys.argv)
