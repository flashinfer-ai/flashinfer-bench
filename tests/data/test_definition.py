import sys

import pytest

from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec


@pytest.fixture
def sample_reference_code():
    # Minimal valid reference function
    return "def run(A, B):\n    return A\n"


def make_minimal_definition(ref_code: str) -> Definition:
    return Definition(
        name="def1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=16)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref_code,
    )


def test_axisconst_valid_and_invalid():
    AxisConst(value=1)
    # We allow zero axis for now
    AxisConst(value=0)
    with pytest.raises(ValueError):
        AxisConst(value=-3)


def test_axisvar_basic():
    ax = AxisVar()
    assert ax.type == "var"


def test_tensorspec_validation():
    TensorSpec(shape=["M"], dtype="int32")
    with pytest.raises(ValueError):
        TensorSpec(shape="M", dtype="int32")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TensorSpec(shape=["M"], dtype="not_a_dtype")  # type: ignore[arg-type]


def test_definition_basic_validation(sample_reference_code):
    definition = make_minimal_definition(sample_reference_code)
    assert definition.name == "def1"
    assert set(definition.const_axes.keys()) == {"N"}
    assert set(definition.var_axes) == {"M"}


def test_definition_axis_reference_checks(sample_reference_code):
    # Input referencing undefined axis should error
    with pytest.raises(ValueError):
        Definition(
            name="bad",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["X"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
        )


def test_definition_reference_must_define_run():
    with pytest.raises(ValueError):
        make_minimal_definition("def not_run():\n    pass\n")
    with pytest.raises(ValueError):
        make_minimal_definition("def run(:\n    pass\n")  # invalid syntax


def test_definition_tags_and_constraints(sample_reference_code):
    # Valid
    Definition(
        name="definition",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference=sample_reference_code,
        tags=["a", "b"],
        constraints=["M > 0", "M <= 4096"],
    )

    # Invalid tags
    with pytest.raises(ValueError):
        Definition(
            name="definition",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            tags=["", 1],  # type: ignore[list-item]
        )

    # Invalid constraints content and syntax
    with pytest.raises(ValueError):
        Definition(
            name="definition",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=[""],
        )
    with pytest.raises(ValueError):
        Definition(
            name="definition",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=["M >"],  # invalid python expression
        )


def test_definition_input_output_names_no_overlap(sample_reference_code):
    """Test that input and output names must not overlap."""
    with pytest.raises(ValueError):
        Definition(
            name="overlapping",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"A": TensorSpec(shape=["M"], dtype="float32")},  # Same name as input
            reference=sample_reference_code,
        )


class TestMergeKwargsToArgs:
    """Tests for merge_kwargs_to_args method."""

    @pytest.fixture
    def definition(self, sample_reference_code):
        return Definition(
            name="test_def",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={
                "A": TensorSpec(shape=["M"], dtype="float32"),
                "B": TensorSpec(shape=["M"], dtype="float32"),
            },
            outputs={"C": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
        )

    def test_no_kwargs(self, definition):
        """Test with no kwargs - args should be unchanged."""
        args = (1, 2)
        result = definition.merge_kwargs_to_args(args, {})
        assert result == (1, 2)

    def test_kwargs_only(self, definition):
        """Test with kwargs only."""
        result = definition.merge_kwargs_to_args((), {"A": 1, "B": 2})
        assert result == (1, 2)

    def test_mixed_args_and_kwargs(self, definition):
        """Test with both args and kwargs."""
        result = definition.merge_kwargs_to_args((1,), {"B": 2})
        assert result == (1, 2)

    def test_includes_output_kwargs(self, definition):
        """Test that output kwargs are also merged."""
        result = definition.merge_kwargs_to_args((1, 2), {"C": 3})
        assert result == (1, 2, 3)

    def test_partial_kwargs(self, definition):
        """Test with partial kwargs - stops at first missing."""
        result = definition.merge_kwargs_to_args((1,), {"C": 3})  # B is missing
        assert result == (1,)  # Stops before B


class TestGetAxesValuesFromInputs:
    """Tests for get_axes_values_from_inputs method."""

    @pytest.fixture
    def definition(self, sample_reference_code):
        return Definition(
            name="test_def",
            op_type="op",
            axes={"M": AxisVar(), "N": AxisConst(value=4)},
            inputs={
                "A": TensorSpec(shape=["M", "N"], dtype="float32"),
                "B": TensorSpec(shape=["M"], dtype="float32"),
            },
            outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
            reference=sample_reference_code,
        )

    def test_with_tensors(self, definition):
        """Test extracting axes from tensor inputs."""
        import torch

        A = torch.zeros((8, 4))
        B = torch.zeros((8,))
        result = definition.get_axes_values_from_inputs([A, B])
        assert result == {"M": 8}

    def test_with_non_tensor(self, definition):
        """Test with non-tensor inputs (scalars)."""
        # Create a definition with a scalar input
        scalar_def = Definition(
            name="scalar_def",
            op_type="op",
            axes={"M": AxisVar()},
            inputs={
                "A": TensorSpec(shape=["M"], dtype="float32"),
                "scale": TensorSpec(shape=None, dtype="float32"),  # scalar
            },
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference="def run(A, scale):\n    return A * scale\n",
        )
        import torch

        A = torch.zeros((5,))
        scale = 2.0  # scalar, no shape
        result = scalar_def.get_axes_values_from_inputs([A, scale])
        assert result == {"M": 5}


def test_get_input_shapes_with_scalar():
    """Test get_input_shapes returns None for scalar inputs and maintains correct length/order."""
    definition = Definition(
        name="scalar_test",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "scale": TensorSpec(shape=None, dtype="float32"),  # scalar in the middle
            "B": TensorSpec(shape=["M"], dtype="float32"),
        },
        outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(A, scale, B):\n    return A * scale + B\n",
    )

    shapes = definition.get_input_shapes({"M": 8})

    # Length should match number of inputs
    assert len(shapes) == len(definition.inputs)

    # Verify each position
    assert shapes[0] == [8, 4]  # A: [M, N]
    assert shapes[1] is None  # scale: scalar
    assert shapes[2] == [8]  # B: [M]


class TestGetAxesValues:
    """Tests for get_axes_values method."""

    @pytest.fixture
    def definition(self, sample_reference_code):
        return Definition(
            name="test_def",
            op_type="op",
            axes={"M": AxisVar(), "N": AxisConst(value=4)},
            inputs={
                "A": TensorSpec(shape=["M", "N"], dtype="float32"),
                "B": TensorSpec(shape=["M"], dtype="float32"),
            },
            outputs={"C": TensorSpec(shape=["M", "N"], dtype="float32")},
            reference=sample_reference_code,
        )

    def test_from_shapes(self, definition):
        """Test extracting axes from shape tuples."""
        shapes = [(8, 4), (8,)]
        result = definition.get_axes_values(shapes)
        assert result == {"M": 8}

    def test_inconsistent_axis_values(self, definition):
        """Test that inconsistent axis values raise an error."""
        shapes = [(8, 4), (10,)]  # M is 8 in A but 10 in B
        with pytest.raises(ValueError):
            definition.get_axes_values(shapes)


if __name__ == "__main__":
    pytest.main(sys.argv)
