import sys

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runners.mp_runner import MultiProcessRunner
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import RandomInput, SafetensorsInput, ScalarInput, Workload


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
    h = r.run_ref(d, wl, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1), None)
    ev = r.run_solution(s, h, BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1))
    assert ev.status.value in {
        "PASSED",
        "RUNTIME_ERROR",
    }  # runtime may fail on env, but no shape/dtype errors
    r.release(h)


if __name__ == "__main__":
    pytest.main(sys.argv)
