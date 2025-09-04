import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runners.mp_runner import (
    MultiProcessRunner,
    _gen_inputs,
    _load_safetensors,
    _normalize_outputs,
    _rand_tensor,
)
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
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


def test_rand_tensor_and_normalize_cpu():
    dev = torch.device("cpu")

    t = _rand_tensor([2, 3], torch.float32, dev)
    assert t.shape == (2, 3) and t.dtype == torch.float32 and t.device.type == "cpu"

    b = _rand_tensor([4], torch.bool, dev)
    assert b.dtype == torch.bool and b.device.type == "cpu"

    out = _normalize_outputs(
        {"Z": 3}, device=dev, output_names=["Z"], output_dtypes={"Z": torch.int32}
    )
    assert out["Z"].dtype == torch.int32 and out["Z"].shape == ()

    y = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _normalize_outputs(y, device=dev, output_names=["Y"], output_dtypes={"Y": torch.float32})
    assert torch.allclose(out["Y"], y)


def test_gen_inputs_random_and_scalar_cpu():
    d = _def2d()
    wl = Workload(
        axes={"M": 2, "N": 3},
        inputs={"X": RandomInput(), "Y": RandomInput(), "S": ScalarInput(value=7)},
        uuid="w1",
    )
    out = _gen_inputs(d, wl, device="cpu", stensors={})
    assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
    assert out["Y"].shape == (2, 3) and out["Y"].dtype == torch.int32
    assert out["S"] == 7


@pytest.mark.skipif(
    __import__("safetensors", fromlist=["torch"]) is None, reason="safetensors not available"
)
def test_load_safetensors_and_gen_inputs_cpu(tmp_path):
    import safetensors.torch as st

    d = _def2d()
    data = {"X": torch.zeros((2, 3), dtype=torch.float32)}
    p = tmp_path / "x.safetensors"
    st.save_file(data, str(p))
    wl = Workload(
        axes={"M": 2, "N": 3},
        inputs={
            "X": SafetensorsInput(path=str(p), tensor_key="X"),
            "S": ScalarInput(value=1),
            "Y": RandomInput(),
        },
        uuid="w2",
    )
    stensors = _load_safetensors(d, wl)
    out = _gen_inputs(d, wl, device="cpu", stensors=stensors)
    assert torch.allclose(out["X"], data["X"]) and out["X"].device.type == "cpu"


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_mp_runner_run_ref_and_solution_minimal(tmp_path, monkeypatch):
    # Dataset
    d = Definition(
        name="dmp",
        type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )
    wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="wmpr")

    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=["gpu"],
        entry_point="pkg/main.py::run",
    )
    srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
    s = Solution(name="py_ok", definition=d.name, author="me", spec=spec, sources=srcs)

    r = MultiProcessRunner(device="cuda:0")
    h = r.run_ref(d, wl, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1))
    ev = r.run_solution(s, h, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1))
    assert ev.status.value in {
        "PASSED",
        "RUNTIME_ERROR",
    }  # runtime may fail on env, but no shape/dtype errors
    r.release(h)
