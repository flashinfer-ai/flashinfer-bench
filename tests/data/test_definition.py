import sys

import pytest

from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    Definition,
    TensorSpec,
)


def make_minimal_definition(ref_code: str) -> Definition:
    return Definition(
        name="def1",
        type="op",
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
    d = make_minimal_definition(sample_reference_code)
    assert d.name == "def1"
    assert set(d.get_const_axes().keys()) == {"N"}
    assert set(d.get_var_axes()) == {"M"}


def test_definition_axis_reference_checks(sample_reference_code):
    # Input referencing undefined axis should error
    with pytest.raises(ValueError):
        Definition(
            name="bad",
            type="op",
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
        name="d",
        type="op",
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
            name="d",
            type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            tags=["", 1],  # type: ignore[list-item]
        )

    # Invalid constraints content and syntax
    with pytest.raises(ValueError):
        Definition(
            name="d",
            type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=[""],
        )
    with pytest.raises(ValueError):
        Definition(
            name="d",
            type="op",
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference=sample_reference_code,
            constraints=["M >"],  # invalid python expression
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
