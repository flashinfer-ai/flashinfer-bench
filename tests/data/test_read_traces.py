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
        "type": "gemm",
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
        "spec": {
            "language": "python",
            "target_hardware": ["cpu"],
            "entry_point": "main.py::run",
        },
        "sources": [
            {"path": "main.py", "content": "def run():\n    pass\n"},
        ],
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
            "status": "passed",
            "log_file": "log",
            "environment": {"hardware": "cpu"},
            "timestamp": "t",
            "correctness": {},
            "performance": {},
        },
    }

    # Write into temp structured dataset
    ddir = tmp_path / "definitions"
    sdir = tmp_path / "solutions"
    tdir = tmp_path / "traces"
    ddir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    tdir.mkdir(parents=True)

    (ddir / "min_gemm.json").write_text(json.dumps(def_json), encoding="utf-8")
    (sdir / "torch_min_gemm.json").write_text(json.dumps(sol_json), encoding="utf-8")
    # JSONL
    lines = [json.dumps(tr_workload), json.dumps(tr_passed)]
    (tdir / "min_gemm.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Load via our codecs/TraceSet
    d = load_json_file(ddir / "min_gemm.json", Definition)
    s = load_json_file(sdir / "torch_min_gemm.json", Solution)
    traces = load_jsonl_file(tdir / "min_gemm.jsonl", Trace)

    assert d.name == "min_gemm"
    assert s.definition == d.name
    assert any(t.is_workload() for t in traces)
    assert any((not t.is_workload()) for t in traces)

    # Roundtrip save new copies
    out_dir = tmp_path / "roundtrip"
    save_json_file(d, out_dir / "def.json")
    save_json_file(s, out_dir / "sol.json")
    save_jsonl_file(traces, out_dir / "tr.jsonl")

    # Reload and validate basic invariants
    d2 = load_json_file(out_dir / "def.json", Definition)
    s2 = load_json_file(out_dir / "sol.json", Solution)
    t2 = load_jsonl_file(out_dir / "tr.jsonl", Trace)

    assert d2.name == d.name
    assert s2.name == s.name
    assert len(t2) == 2

    # End-to-end via TraceSet
    ts = TraceSet.from_path(str(tmp_path))
    assert ts.definitions.get("min_gemm").name == "min_gemm"
    assert ts.get_solution("torch_min_gemm").name == "torch_min_gemm"
    assert len(ts.traces.get("min_gemm", [])) == 1  # only the passed one
    assert len(ts.workload.get("min_gemm", [])) == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
