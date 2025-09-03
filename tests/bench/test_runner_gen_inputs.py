import pytest
import torch

from flashinfer_bench.bench.runner import Runner
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.trace import RandomInput, SafetensorsInput, ScalarInput, Workload


def _def2d():
    return Definition(
        name="d",
        type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=3)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="int32"),
            "S": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y, S):\n    return X\n",
    )


def test_gen_inputs_random_and_scalar_cpu():
    r = Runner(device="cpu")
    d = _def2d()
    wl = Workload(
        axes={"M": 2, "N": 3},
        inputs={"X": RandomInput(), "Y": RandomInput(), "S": ScalarInput(value=7)},
        uuid="w1",
    )
    out = r._gen_inputs(d, wl)
    assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
    assert out["Y"].shape == (2, 3) and out["Y"].dtype == torch.int32
    assert out["S"] == 7


def test_gen_inputs_requires_host_tensor_for_safetensors():
    r = Runner(device="cpu")
    d = _def2d()
    wl = Workload(
        axes={"M": 2, "N": 3},
        inputs={"X": SafetensorsInput(path="/p", tensor_key="k"), "S": ScalarInput(value=1)},
        uuid="w2",
    )
    with pytest.raises(RuntimeError, match="Missing host tensor"):
        r._gen_inputs(d, wl)

    # Provide host tensor
    host = {"X": torch.zeros((2, 3), dtype=torch.float32)}
    out = r._gen_inputs(d, wl, host_tensors=host)
    assert isinstance(out["X"], torch.Tensor)
    assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
