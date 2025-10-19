import logging
import sys
from pathlib import Path

import pytest
import torch

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.bench.runner import IsolatedRunner
from flashinfer_bench.bench.runner.isolated_runner import SubprocessWorker
from flashinfer_bench.bench.utils import (
    _rand_tensor,
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Workload,
)


def test_isolated_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "flashinfer_bench.utils.list_cuda_devices", lambda: ["dev0", "dev1", "dev2"]
    )

    # Replace SubprocessWorker with a lightweight dummy to avoid actual process spawning
    class _Dummy:
        def __init__(self, device: str, log_dir: str) -> None:
            self._device = device

        def is_healthy(self) -> bool:
            return True

        def close(self) -> None:
            pass

        def run_ref(self, *a, **k):
            raise NotImplementedError

        def run_solution(self, *a, **k):
            raise NotImplementedError

        def release(self, *a, **k):
            pass

    monkeypatch.setattr("flashinfer_bench.bench.runner.isolated_runner.SubprocessWorker", _Dummy)
    b = IsolatedRunner(logging.getLogger(__name__))

    b._workers = [object(), object(), object()]
    b._curr_worker_idx = 0

    sel1 = b._pick_workers(2)
    assert sel1 == [b._workers[0], b._workers[1]]
    sel2 = b._pick_workers(2)
    assert sel2 == [b._workers[2], b._workers[0]]
    sel3 = b._pick_workers(1)
    assert sel3 == [b._workers[1]]
    assert b._pick_workers(0) == []


def _def2d():
    return Definition(
        name="d",
        op_type="op",
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

    out = normalize_outputs(
        {"Z": 3}, device=dev, output_names=["Z"], output_dtypes={"Z": torch.int32}
    )
    assert out["Z"].dtype == torch.int32 and out["Z"].shape == ()

    y = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = normalize_outputs(y, device=dev, output_names=["Y"], output_dtypes={"Y": torch.float32})
    assert torch.allclose(out["Y"], y)


def test_gen_inputs_random_and_scalar_cpu():
    d = _def2d()
    wl = Workload(
        axes={"M": 2, "N": 3},
        inputs={"X": RandomInput(), "Y": RandomInput(), "S": ScalarInput(value=7)},
        uuid="w1",
    )
    out = gen_inputs(d, wl, device="cpu", stensors={})
    assert out["X"].shape == (2, 3) and out["X"].dtype == torch.float32
    assert out["Y"].shape == (2, 3) and out["Y"].dtype == torch.int32
    assert out["S"] == 7


@pytest.mark.skipif(
    __import__("safetensors", fromlist=["torch"]) is None, reason="safetensors not available"
)
def test_load_safetensors_and_gen_inputs_cpu(tmp_path: Path):
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
    stensors = load_safetensors(d, wl)
    out = gen_inputs(d, wl, device="cpu", stensors=stensors)
    assert torch.allclose(out["X"], data["X"]) and out["X"].device.type == "cpu"


def test_compute_error_stats():
    cfg = BenchmarkConfig(
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        rtol=float(1e-2),
        atol=float(1e-2),
        log_dir="ignored",
    )

    ref = torch.tensor([0.0, 10.0, -2.0], dtype=torch.float32)
    tol = cfg.atol + cfg.rtol * ref.abs()

    sol_pass = ref + tol * torch.tensor([0.5, -0.9, 0.3], dtype=torch.float32)
    sol_fail = ref + tol * torch.tensor([0.5, -1.2, 0.3], dtype=torch.float32)

    abs_pass, rel_pass, exceeds_pass, matched_ratio_pass = compute_error_stats(sol_pass, ref, cfg)
    diff_pass = (sol_pass - ref).abs()
    eps = 1e-8
    expected_rel_pass = (diff_pass / (ref.abs() + eps)).max().item()
    assert abs_pass == pytest.approx(diff_pass.max().item())
    assert rel_pass == pytest.approx(expected_rel_pass)
    assert not exceeds_pass
    assert matched_ratio_pass == pytest.approx(1.0)
    assert torch.allclose(sol_pass, ref, atol=cfg.atol, rtol=cfg.rtol)

    abs_fail, rel_fail, exceeds_fail, matched_ratio_fail = compute_error_stats(sol_fail, ref, cfg)
    diff_fail = (sol_fail - ref).abs()
    expected_rel_fail = (diff_fail / (ref.abs() + eps)).max().item()
    assert abs_fail == pytest.approx(diff_fail.max().item())
    assert rel_fail == pytest.approx(expected_rel_fail)
    assert exceeds_fail
    assert matched_ratio_fail == pytest.approx(2.0 / 3.0)
    assert not torch.allclose(sol_fail, ref, atol=cfg.atol, rtol=cfg.rtol)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_isolated_runner_run_ref_and_solution_minimal():
    # Dataset
    d = Definition(
        name="dmp",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )
    wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="wmpr")

    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["gpu"], entry_point="pkg/main.py::run"
    )
    srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
    s = Solution(name="py_ok", definition=d.name, author="me", spec=spec, sources=srcs)

    r = SubprocessWorker(device="cuda:0")
    h = r.run_ref(d, wl, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1), None)
    ev = r.run_solution(s, h, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1))
    assert ev.status.value in {
        "PASSED",
        "RUNTIME_ERROR",
        "COMPILE_ERROR",
    }  # runtime may fail on env, but no shape/dtype errors
    r.release(h)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
def test_isolated_worker_embeds_stdout(tmp_path: Path):
    d = Definition(
        name="dmp_log",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )
    wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="wmpr_log")

    message = "isolated worker log line"
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["gpu"], entry_point="pkg/main.py::run"
    )
    srcs = [
        SourceFile(
            path="pkg/main.py",
            content=(
                "import torch\n" f"def run(A):\n" f"    print({message!r})\n" "    return A\n"
            ),
        )
    ]
    sol = Solution(name="py_log", definition=d.name, author="me", spec=spec, sources=srcs)

    worker = SubprocessWorker(device="cuda:0", log_dir=str(tmp_path / "logs"))
    cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
    handle = None
    try:
        handle = worker.run_ref(d, wl, cfg, None)
        evaluation = worker.run_solution(sol, handle, cfg)
        assert isinstance(evaluation.log, str)
        assert message in evaluation.log
    finally:
        if handle is not None:
            worker.release(handle)
        worker.close()


if __name__ == "__main__":
    pytest.main(sys.argv)
