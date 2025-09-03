import pytest
import torch

from flashinfer_bench.bench.runner import _normalize_outputs, _rand_tensor


def test_rand_tensor_basic_float_bool_int_cpu():
    dev = torch.device("cpu")

    t = _rand_tensor([2, 3], torch.float32, dev)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 3)
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"

    b = _rand_tensor([5], torch.bool, dev)
    assert b.dtype == torch.bool
    assert b.device.type == "cpu"
    # Values should be 0/1
    assert set(b.unique().tolist()).issubset({False, True})

    i = _rand_tensor([4], torch.int32, dev)
    assert i.dtype == torch.int32
    assert i.device.type == "cpu"
    # Should be within the configured range [-1024, 1024)
    assert int(i.min()) >= -1280  # loose lower bound
    assert int(i.max()) <= 2048  # loose upper bound


def test_normalize_outputs_from_dict_tensor_scalar():
    dev = torch.device("cpu")
    output_names = ["Y"]
    output_dtypes = {"Y": torch.float32}

    # From dict with scalar value and extra key filtered out
    out = _normalize_outputs(
        {"Y": 3.14, "EXTRA": 1}, device=dev, output_names=output_names, output_dtypes=output_dtypes
    )
    assert set(out.keys()) == {"Y"}
    assert out["Y"].dtype == torch.float32 and out["Y"].device.type == "cpu"
    assert out["Y"].shape == ()  # 0-D scalar tensor

    # From single tensor
    y = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _normalize_outputs(y, device=dev, output_names=["Y"], output_dtypes=output_dtypes)
    assert torch.allclose(out["Y"], y)

    # From scalar
    out = _normalize_outputs(7, device=dev, output_names=["Y"], output_dtypes=output_dtypes)
    assert out["Y"].shape == () and out["Y"].item() == 7


def test_normalize_outputs_shape_mismatch_errors():
    dev = torch.device("cpu")
    dtypes = {"A": torch.float32, "B": torch.float32}
    # Single tensor/scalar with multiple outputs must error
    with pytest.raises(RuntimeError):
        _normalize_outputs(
            torch.tensor(1.0), device=dev, output_names=["A", "B"], output_dtypes=dtypes
        )
    with pytest.raises(RuntimeError):
        _normalize_outputs(1, device=dev, output_names=["A", "B"], output_dtypes=dtypes)
