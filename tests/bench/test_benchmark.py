from pathlib import Path

import pytest
import safetensors.torch as st
import torch

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
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)


def test_benchmark_pick_runners_round_robin(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "flashinfer_bench.bench.benchmark.list_cuda_devices",
        lambda: ["dev0", "dev1", "dev2"],
    )
    ts = TraceSet(root=tmp_path)
    b = Benchmark(ts)

    b._runners = [object(), object(), object()]
    b._curr_device_idx = 0

    sel1 = b._pick_runners(2)
    assert sel1 == [b._runners[0], b._runners[1]]
    sel2 = b._pick_runners(2)
    assert sel2 == [b._runners[2], b._runners[0]]
    sel3 = b._pick_runners(1)
    assert sel3 == [b._runners[1]]
    assert b._pick_runners(0) == []


def test_prefetch_safetensors_happy_path(monkeypatch, tmp_path):
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


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_end_to_end_multi_gpu_with_safetensors(monkeypatch, tmp_path: Path):
    # Build dataset structure
    (tmp_path / "definitions").mkdir(parents=True)
    (tmp_path / "solutions").mkdir(parents=True)
    (tmp_path / "traces").mkdir(parents=True)

    def make_def(name: str, axes: dict, inputs: dict, outputs: dict, body: str) -> Definition:
        return Definition(
            name=name,
            type="op",
            axes=axes,
            inputs=inputs,
            outputs=outputs,
            reference=("import torch\n\n" + f"def run({', '.join(inputs.keys())}):\n    {body}\n"),
        )

    d_add = make_def(
        "add_var",
        axes={"M": AxisConst(value=8), "N": AxisConst(value=16)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        body="return A + B",
    )
    d_mul = make_def(
        "mul_var",
        axes={"M": AxisConst(value=8), "N": AxisConst(value=16)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        body="return X * Y",
    )
    d_adds = make_def(
        "add_with_scalar",
        axes={"M": AxisConst(value=8), "N": AxisConst(value=16)},
        inputs={
            "A": TensorSpec(shape=["M", "N"], dtype="float32"),
            "B": TensorSpec(shape=["M", "N"], dtype="float32"),
            "S": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        body="return A + B + S",
    )
    for d in (d_add, d_mul, d_adds):
        save_json_file(d, tmp_path / "definitions" / f"{d.name}.json")

    def mk_solution(def_name: str, sol_name: str, params: list[str], body: str) -> Solution:
        path = f"pkg/{sol_name}.py"
        content = "import torch\n\n" + f"def run({', '.join(params)}):\n    {body}\n"
        return Solution(
            name=sol_name,
            definition=def_name,
            author="tester",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point=f"{path}::run",
            ),
            sources=[SourceFile(path=path, content=content)],
        )

    sol_matrix = {
        d_add.name: [
            mk_solution(d_add.name, "add_ok", ["A", "B"], "return A + B"),
            mk_solution(d_add.name, "add_off1", ["A", "B"], "return A + B + 1"),
            mk_solution(d_add.name, "add_half", ["A", "B"], "return (A + B).half()"),
            mk_solution(d_add.name, "add_boom", ["A", "B"], "raise RuntimeError('boom')"),
        ],
        d_mul.name: [
            mk_solution(d_mul.name, "mul_ok", ["X", "Y"], "return X * Y"),
            mk_solution(d_mul.name, "mul_off1", ["X", "Y"], "return X * Y + 1"),
            mk_solution(d_mul.name, "mul_half", ["X", "Y"], "return (X * Y).half()"),
            mk_solution(d_mul.name, "mul_boom", ["X", "Y"], "raise RuntimeError('boom')"),
        ],
        d_adds.name: [
            mk_solution(d_adds.name, "adds_ok", ["A", "B", "S"], "return A + B + S"),
            mk_solution(d_adds.name, "adds_off1", ["A", "B", "S"], "return A + B + S + 1"),
            mk_solution(d_adds.name, "adds_half", ["A", "B", "S"], "return (A + B + S).half()"),
            mk_solution(d_adds.name, "adds_boom", ["A", "B", "S"], "raise RuntimeError('boom')"),
        ],
    }
    for sols in sol_matrix.values():
        for s in sols:
            save_json_file(s, tmp_path / "solutions" / f"{s.name}.json")

    safes = {}
    for tensor_name in ("A", "B", "X", "Y"):
        t = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
        p = tmp_path / f"{tensor_name}.safetensors"
        st.save_file({tensor_name: t}, str(p))
        safes[tensor_name] = p

    wl_matrix = {
        d_add.name: [
            Workload(
                axes={"M": 8, "N": 16}, inputs={"A": RandomInput(), "B": RandomInput()}, uuid="a1"
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": SafetensorsInput(path=str(safes["A"]), tensor_key="A"),
                    "B": RandomInput(),
                },
                uuid="a2",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": RandomInput(),
                    "B": SafetensorsInput(path=str(safes["B"]), tensor_key="B"),
                },
                uuid="a3",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": SafetensorsInput(path=str(safes["A"]), tensor_key="A"),
                    "B": SafetensorsInput(path=str(safes["B"]), tensor_key="B"),
                },
                uuid="a4",
            ),
        ],
        d_mul.name: [
            Workload(
                axes={"M": 8, "N": 16}, inputs={"X": RandomInput(), "Y": RandomInput()}, uuid="m1"
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "X": SafetensorsInput(path=str(safes["X"]), tensor_key="X"),
                    "Y": RandomInput(),
                },
                uuid="m2",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "X": RandomInput(),
                    "Y": SafetensorsInput(path=str(safes["Y"]), tensor_key="Y"),
                },
                uuid="m3",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "X": SafetensorsInput(path=str(safes["X"]), tensor_key="X"),
                    "Y": SafetensorsInput(path=str(safes["Y"]), tensor_key="Y"),
                },
                uuid="m4",
            ),
        ],
        d_adds.name: [
            Workload(
                axes={"M": 8, "N": 16},
                inputs={"A": RandomInput(), "B": RandomInput(), "S": ScalarInput(value=1)},
                uuid="s1",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": SafetensorsInput(path=str(safes["A"]), tensor_key="A"),
                    "B": RandomInput(),
                    "S": ScalarInput(value=2),
                },
                uuid="s2",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": RandomInput(),
                    "B": SafetensorsInput(path=str(safes["B"]), tensor_key="B"),
                    "S": ScalarInput(value=3),
                },
                uuid="s3",
            ),
            Workload(
                axes={"M": 8, "N": 16},
                inputs={
                    "A": SafetensorsInput(path=str(safes["A"]), tensor_key="A"),
                    "B": SafetensorsInput(path=str(safes["B"]), tensor_key="B"),
                    "S": ScalarInput(value=4),
                },
                uuid="s4",
            ),
        ],
    }
    for defn_name, wls in wl_matrix.items():
        save_jsonl_file(
            [Trace(definition=defn_name, workload=w) for w in wls],
            tmp_path / "traces" / "workloads" / f"{defn_name}.jsonl",
        )

    ts = TraceSet.from_path(str(tmp_path))
    bench = Benchmark(ts)

    cfg = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    bench.run(cfg)

    # Validate in-memory results (avoid on-disk overwrite issue)
    total_expected = sum(len(wls) * len(sol_matrix[d]) for d, wls in wl_matrix.items())
    assert len(bench._staging_traces) == total_expected
    statuses = [t.evaluation.status.value for t in bench._staging_traces]
    assert any(s == "PASSED" for s in statuses)
    assert any(s == "INCORRECT_NUMERICAL" for s in statuses)
    assert any(s == "INCORRECT_DTYPE" for s in statuses)
    assert any(s == "RUNTIME_ERROR" for s in statuses)

    # Flush for I/O path and check files created
    bench.flush()
    for d in (d_add, d_mul, d_adds):
        out_file = tmp_path / d.type / f"{d.name}.jsonl"
        assert out_file.exists()
