import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner import (
    BaselineHandle,
    Runner,
    _DeviceBaseline,
    _normalize_outputs,
    _rand_tensor,
)
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.trace import RandomInput, SafetensorsInput, ScalarInput, Workload


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
    assert set(b.unique().tolist()).issubset({False, True})

    i = _rand_tensor([4], torch.int32, dev)
    assert i.dtype == torch.int32
    assert i.device.type == "cpu"


def test_normalize_outputs_from_dict_tensor_scalar():
    dev = torch.device("cpu")
    output_names = ["Y"]
    output_dtypes = {"Y": torch.float32}

    out = _normalize_outputs(
        {"Y": 3.14, "EXTRA": 1}, device=dev, output_names=output_names, output_dtypes=output_dtypes
    )
    assert set(out.keys()) == {"Y"}
    assert out["Y"].dtype == torch.float32 and out["Y"].device.type == "cpu"
    assert out["Y"].shape == ()

    y = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _normalize_outputs(y, device=dev, output_names=["Y"], output_dtypes=output_dtypes)
    assert torch.allclose(out["Y"], y)

    out = _normalize_outputs(7, device=dev, output_names=["Y"], output_dtypes=output_dtypes)
    assert out["Y"].shape == () and out["Y"].item() == 7


def test_normalize_outputs_shape_mismatch_errors():
    dev = torch.device("cpu")
    dtypes = {"A": torch.float32, "B": torch.float32}
    with pytest.raises(RuntimeError):
        _normalize_outputs(
            torch.tensor(1.0), device=dev, output_names=["A", "B"], output_dtypes=dtypes
        )
    with pytest.raises(RuntimeError):
        _normalize_outputs(1, device=dev, output_names=["A", "B"], output_dtypes=dtypes)


# ---------- Runner._gen_inputs ----------


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

    host = {"X": torch.zeros((2, 3), dtype=torch.float32)}
    out = r._gen_inputs(d, wl, host_tensors=host)
    assert isinstance(out["X"], torch.Tensor)
    assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32


# ---------- Runner.run_reference / run_solution ----------


def _mk_runnable(fn, name: str):
    return Runnable(fn=fn, closer=lambda: None, meta={"name": name})


def _mk_baseline_cpu(ref_tensor: torch.Tensor):
    handle = BaselineHandle("h1")
    return handle, _DeviceBaseline(
        handle=handle,
        device="cpu",
        num_trials=1,
        output_names=["O"],
        output_dtypes={"O": ref_tensor.dtype},
        inputs_dev=[{"A": torch.tensor([1.0])}],
        ref_outputs_dev=[{"O": ref_tensor}],
        ref_latencies_ms=[10.0],
        ref_mean_latency_ms=10.0,
    )


def test_run_reference_raises_on_non_cuda():
    r = Runner(device="cpu")
    cfg = BenchmarkConfig()
    with pytest.raises(RuntimeError, match="supports CUDA devices only"):
        r.run_reference(defn=object(), workload=object(), cfg=cfg, runnable_ref=object())  # type: ignore[arg-type]


def test_run_solution_pass_numerical_and_perf(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([2.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl

    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 5.0)

    run_ok = _mk_runnable(lambda **kw: torch.tensor([2.0], dtype=torch.float32), name="ok")
    ev = r.run_solution(run_ok, h, BenchmarkConfig())
    assert ev.status.name == "PASSED"
    assert ev.correctness is not None and ev.performance is not None
    assert ev.performance.latency_ms == pytest.approx(5.0)
    assert ev.log_file.endswith("ok.log")


def test_run_solution_incorrect_numerical(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([1.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl
    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 1.0)

    run_bad = _mk_runnable(lambda **kw: torch.tensor([1.1], dtype=torch.float32), name="bad")
    ev = r.run_solution(run_bad, h, BenchmarkConfig())
    assert ev.status.name == "INCORRECT_NUMERICAL"
    assert ev.correctness is not None and ev.performance is None


def test_run_solution_incorrect_shape(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([0.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl
    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 1.0)

    run_shape = _mk_runnable(lambda **kw: torch.zeros((2,), dtype=torch.float32), name="shape")
    ev = r.run_solution(run_shape, h, BenchmarkConfig())
    assert ev.status.name == "INCORRECT_SHAPE"


def test_run_solution_incorrect_dtype(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([0.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl
    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 1.0)

    run_dtype = _mk_runnable(lambda **kw: torch.tensor([0.0], dtype=torch.float16), name="dtype")
    ev = r.run_solution(run_dtype, h, BenchmarkConfig())
    assert ev.status.name == "INCORRECT_DTYPE"


def test_run_solution_runtime_error(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([0.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl
    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 1.0)

    def boom(**kw):
        raise RuntimeError("boom")

    run_err = _mk_runnable(boom, name="err")
    ev = r.run_solution(run_err, h, BenchmarkConfig())
    assert ev.status.name == "RUNTIME_ERROR"
