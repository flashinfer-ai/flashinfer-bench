import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_end_to_end_multi_gpu_with_safetensors(monkeypatch, tmp_path: Path):
    import torch

    try:
        import safetensors.torch as st
    except Exception:
        pytest.skip("safetensors not available")

    # Use all available GPUs (ensures multi-device path when >1 GPU)
    num_gpus = torch.cuda.device_count()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(num_gpus)))

    from flashinfer_bench.bench.benchmark import Benchmark
    from flashinfer_bench.bench.config import BenchmarkConfig
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
        Trace,
        Workload,
        save_json_file,
        save_jsonl_file,
    )
    from flashinfer_bench.data.traceset import TraceSet

    # Build dataset structure
    (tmp_path / "definitions").mkdir(parents=True)
    (tmp_path / "solutions").mkdir(parents=True)
    (tmp_path / "traces").mkdir(parents=True)

    # Definition: A, B tensors and S scalar -> O tensor
    defn = Definition(
        name="add_with_scalar",
        type="op",
        axes={"M": AxisConst(value=8), "N": AxisConst(value=16)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
            "S": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=(
            "import torch\n\n"
            "def run(A: torch.Tensor, B: torch.Tensor, S: int):\n"
            "    return A + B + S\n"
        ),
    )
    save_json_file(defn, tmp_path / "definitions" / "add_with_scalar.json")

    # Solutions: one correct, one incorrect (omits S)
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=["gpu"],
        entry_point="pkg/main.py::run",
    )
    src_ok = SourceFile(
        path="pkg/main.py",
        content=(
            "import torch\n\n"
            "def run(A: torch.Tensor, B: torch.Tensor, S: int):\n"
            "    return A + B + S\n"
        ),
    )
    src_bad = SourceFile(
        path="pkg/main.py",
        content=(
            "import torch\n\n"
            "def run(A: torch.Tensor, B: torch.Tensor, S: int):\n"
            "    return A + B\n"
        ),
    )

    sol_ok = Solution(
        name="py_add_with_scalar_ok",
        definition="add_with_scalar",
        author="tester",
        spec=spec,
        sources=[src_ok],
    )
    sol_bad = Solution(
        name="py_add_with_scalar_bad",
        definition="add_with_scalar",
        author="tester",
        spec=spec,
        sources=[src_bad],
    )
    save_json_file(sol_ok, tmp_path / "solutions" / f"{sol_ok.name}.json")
    save_json_file(sol_bad, tmp_path / "solutions" / f"{sol_bad.name}.json")

    # Create safetensors input for A
    a_tensor = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    safep = tmp_path / "A.safetensors"
    st.save_file({"A": a_tensor}, str(safep))

    # Workload-only trace JSONL: A from safetensors, B random, S scalar
    wl = Workload(
        axes={"M": 8, "N": 16},
        inputs={
            "A": SafetensorsInput(path=str(safep), tensor_key="A"),
            "B": RandomInput(),
            "S": ScalarInput(value=1),
        },
        uuid="wl1",
    )
    t_workload = Trace(definition=defn.name, workload=wl)
    save_jsonl_file([t_workload], tmp_path / "traces" / f"{defn.name}.jsonl")

    # Load, run, flush
    ts = TraceSet.from_path(str(tmp_path))
    bench = Benchmark(ts)
    cfg = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    bench.run(cfg)
    bench.flush()

    # Verify output traces written under <root>/<type>/<name>.jsonl
    out_file = tmp_path / defn.type / f"{defn.name}.jsonl"
    assert out_file.exists()
    lines = [json.loads(x) for x in out_file.read_text().strip().splitlines() if x.strip()]
    # We should have one line per solution
    sols = {x["solution"] for x in lines}
    assert {sol_ok.name, sol_bad.name} <= sols

    # Find statuses
    statuses = {x["solution"]: x["evaluation"]["status"].upper() for x in lines}
    # ok passes, bad is incorrect_numerical
    assert statuses[sol_ok.name] == "PASSED"
    assert statuses[sol_bad.name] == "INCORRECT_NUMERICAL"
