from __future__ import annotations

import json
import sys

import pytest

from flashinfer_bench.apply.key import ApplyKey, ApplyKeyFactory
from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


def make_minimal_def() -> Definition:
    return Definition(
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


def test_applykey_encode_decode_equality():
    k = ApplyKey(axes=(("M", 2), ("N", 4)), feats=(("avg", 1.5), ("flag", True)))
    s = k.encode
    # ensure stable json
    json.loads(s)
    k2 = ApplyKey.from_encoded(s)
    assert k2 == k
    assert hash(k2) == hash(k)


def test_axes_only_key_builder_materializes_axes_and_errors():
    d = make_minimal_def()
    builder = ApplyKeyFactory.specialize(d)

    # Valid runtime kwargs
    key = builder.build_from_runtime({"X": FakeTensor((4, 2)), "Y": FakeTensor((4, 2))})
    # Only var axes are materialized; const axes are not included in key.
    assert dict(key.axes) == {"M": 4}
    assert dict(key.axes).get("M") == 4
    assert "N" not in dict(key.axes)

    # Missing input for axis projection: 'M' is first seen on 'X', so omitting 'X' should error
    with pytest.raises(KeyError):
        builder.build_from_runtime({"Y": FakeTensor((4, 2))})

    # Rank too small: X is expected to have at least 1 dim at index 0; providing 0-dim tensor causes error
    with pytest.raises(ValueError):
        builder.build_from_runtime({"X": FakeTensor(()), "Y": FakeTensor((4, 2))})


if __name__ == "__main__":
    pytest.main(sys.argv)
