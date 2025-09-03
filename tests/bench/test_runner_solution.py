import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner import BaselineHandle, Runner, _DeviceBaseline
from flashinfer_bench.compile.runnable import Runnable


def _mk_runnable(fn, name: str):
    return Runnable(fn=fn, closer=lambda: None, meta={"solution": name})


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
        # Early guard triggers; other args are not used
        r.run_reference(defn=object(), workload=object(), cfg=cfg, runnable_ref=object())  # type: ignore[arg-type]


def test_run_solution_pass_numerical_and_perf(monkeypatch):
    r = Runner(device="cpu")
    ref = torch.tensor([2.0], dtype=torch.float32)
    h, bl = _mk_baseline_cpu(ref)
    r._baselines[h] = bl

    # Stub timer to avoid CUDA synchronization and return constant latency
    monkeypatch.setattr(Runner, "_time_runnable", lambda self, fn, inputs, warmup, iters: 5.0)

    # Exact match returns PASSED
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

    # Different shape
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
