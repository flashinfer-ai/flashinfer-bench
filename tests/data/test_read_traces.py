import json
import sys
from pathlib import Path

import pytest

from flashinfer_bench.data import (
    Definition,
    Solution,
    Trace,
    TraceSet,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)


def test_end_to_end_minimal_roundtrip(tmp_path: Path):
    # Minimal definition JSON
    def_json = {
        "name": "min_gemm",
        "op_type": "gemm",
        "axes": {"M": {"type": "var"}, "N": {"type": "const", "value": 4}},
        "inputs": {"A": {"shape": ["M", "N"], "dtype": "float32"}},
        "outputs": {"C": {"shape": ["M", "N"], "dtype": "float32"}},
        "reference": "def run(a):\n    return a\n",
    }

    # Minimal solution JSON
    sol_json = {
        "name": "torch_min_gemm",
        "definition": "min_gemm",
        "author": "tester",
        "spec": {"language": "python", "target_hardware": ["cpu"], "entry_point": "main.py::run"},
        "sources": [{"path": "main.py", "content": "def run():\n    pass\n"}],
    }

    # Two traces: one workload-only, one passed
    tr_workload = {
        "definition": "min_gemm",
        "workload": {"axes": {"M": 2}, "inputs": {"A": {"type": "random"}}, "uuid": "wrt1"},
        "solution": None,
        "evaluation": None,
    }
    tr_passed = {
        "definition": "min_gemm",
        "workload": {"axes": {"M": 2}, "inputs": {"A": {"type": "random"}}, "uuid": "wrt2"},
        "solution": "torch_min_gemm",
        "evaluation": {
            "status": "PASSED",
            "log": "log",
            "environment": {"hardware": "cpu"},
            "timestamp": "t",
            "correctness": {"max_relative_error": 0.0, "max_absolute_error": 0.0},
            "performance": {"latency_ms": 1.0, "reference_latency_ms": 2.0, "speedup_factor": 2.0},
        },
    }

    # Write into temp structured dataset
    ddir = tmp_path / "definitions"
    sdir = tmp_path / "solutions"
    wdir = tmp_path / "workloads"
    tdir = tmp_path / "traces"
    ddir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    wdir.mkdir(parents=True)
    tdir.mkdir(parents=True)

    (ddir / "min_gemm.json").write_text(json.dumps(def_json), encoding="utf-8")
    (sdir / "torch_min_gemm.json").write_text(json.dumps(sol_json), encoding="utf-8")
    (wdir / "min_gemm.jsonl").write_text(
        json.dumps(tr_workload, indent=None) + "\n", encoding="utf-8"
    )
    (tdir / "min_gemm.jsonl").write_text(
        json.dumps(tr_passed, indent=None) + "\n", encoding="utf-8"
    )

    # Load via our codecs/TraceSet
    loaded_def = load_json_file(Definition, ddir / "min_gemm.json")
    loaded_sol = load_json_file(Solution, sdir / "torch_min_gemm.json")
    loaded_workload = load_jsonl_file(Trace, wdir / "min_gemm.jsonl")
    loaded_traces = load_jsonl_file(Trace, tdir / "min_gemm.jsonl")

    assert loaded_def.name == "min_gemm"
    assert loaded_sol.definition == loaded_def.name
    assert all(t.is_workload_trace() for t in loaded_workload)
    assert all((not t.is_workload_trace()) for t in loaded_traces)

    # Roundtrip save new copies
    out_dir = tmp_path / "roundtrip"
    save_json_file(loaded_def, out_dir / "def.json")
    save_json_file(loaded_sol, out_dir / "solution.json")
    save_jsonl_file(loaded_workload, out_dir / "workloads.jsonl")
    save_jsonl_file(loaded_traces, out_dir / "traces.jsonl")

    # Reload and validate basic invariants
    loaded_def2 = load_json_file(Definition, out_dir / "def.json")
    loaded_sol2 = load_json_file(Solution, out_dir / "solution.json")
    loaded_workload2 = load_jsonl_file(Trace, out_dir / "workloads.jsonl")
    loaded_traces2 = load_jsonl_file(Trace, out_dir / "traces.jsonl")

    assert loaded_def2.name == loaded_def.name
    assert loaded_sol2.name == loaded_sol.name
    assert len(loaded_workload2) == 1
    assert loaded_workload2[0].is_workload_trace()
    assert len(loaded_traces2) == 1
    assert not loaded_traces2[0].is_workload_trace()

    # End-to-end via TraceSet
    ts = TraceSet.from_path(str(tmp_path))
    assert ts.definitions.get("min_gemm").name == "min_gemm"
    assert ts.get_solution("torch_min_gemm").name == "torch_min_gemm"
    assert len(ts.traces.get("min_gemm", [])) == 1  # only the passed one
    assert len(ts.workloads.get("min_gemm", [])) == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
