import os

import pytest

from flashinfer_bench.bench.benchmark import Benchmark
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.trace import SafetensorsInput, Workload
from flashinfer_bench.data.traceset import TraceSet


def test_prefetch_safetensors_happy_path(monkeypatch, tmp_path):
    try:
        import safetensors.torch as st  # type: ignore
        import torch
    except Exception:
        pytest.skip("safetensors or torch not available")

    # Ensure non-empty device list
    monkeypatch.setattr("flashinfer_bench.bench.benchmark.list_cuda_devices", lambda: ["dev0"])
    ts = TraceSet(root=tmp_path)
    bench = Benchmark(ts)

    d = Definition(
        name="d",
        type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"O": TensorSpec(shape=["N"], dtype="float32")},
        reference="def run(A):\n    return A\n",
    )

    data = {"x": torch.arange(4, dtype=torch.float32)}
    sf = tmp_path / "a.safetensors"
    st.save_file(data, str(sf))

    wl = Workload(
        axes={"N": 4}, inputs={"A": SafetensorsInput(path=str(sf), tensor_key="x")}, uuid="w"
    )

    host = bench._prefetch_safetensors(d, wl)
    assert "A" in host
    assert host["A"].shape == (4,) and host["A"].dtype == torch.float32
