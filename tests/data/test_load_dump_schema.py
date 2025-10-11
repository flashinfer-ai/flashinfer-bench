import sys
from pathlib import Path
from typing import Tuple

import pytest

from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    Workload,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)


def make_minimal_objects() -> Tuple[Definition, Solution, Trace]:
    ref = "def run(a):\n    return a\n"
    d = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref,
    )
    s = Solution(
        name="s1",
        definition="d1",
        author="me",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    wl = Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="w1")
    ev = Evaluation(
        status=EvaluationStatus.PASSED,
        log="log",
        environment=Environment(hardware="cpu"),
        timestamp="t",
        correctness=Correctness(),
        performance=Performance(),
    )
    t = Trace(definition="d1", workload=wl, solution="s1", evaluation=ev)
    return d, s, t


def test_roundtrip_to_from_json():
    d, s, t = make_minimal_objects()
    d2 = Definition.model_validate_json(d.model_dump_json())
    s2 = Solution.model_validate_json(s.model_dump_json())
    t2 = Trace.model_validate_json(t.model_dump_json())
    assert d2.name == d.name
    assert s2.name == s.name
    assert t2.solution == t.solution


def test_preserve_null_fields_in_trace_json():
    wl = Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="w2")
    t = Trace(definition="d1", workload=wl)  # workload-only
    obj = t.model_dump(mode="json")
    # solution and evaluation must be present and null
    assert "solution" in obj and obj["solution"] is None
    assert "evaluation" in obj and obj["evaluation"] is None


def test_language_and_status_string_decoding():
    data = {"language": "triton", "target_hardware": ["cuda"], "entry_point": "main.py::run"}
    bs = BuildSpec.model_validate(data)
    assert bs.language == SupportedLanguages.TRITON

    ev_data = {
        "status": "PASSED",
        "log": "log",
        "environment": {"hardware": "cpu"},
        "timestamp": "t",
        "correctness": {},
        "performance": {},
    }
    ev = Evaluation.model_validate(ev_data)
    assert ev.status == EvaluationStatus.PASSED


def test_save_and_load_json_and_jsonl(tmp_path: Path):
    d, s, t = make_minimal_objects()
    # JSON file roundtrip
    path = tmp_path / "obj.json"
    save_json_file(d, path)
    loaded = load_json_file(Definition, path)
    assert loaded.name == d.name

    # JSONL file roundtrip
    pathl = tmp_path / "objs.jsonl"
    traces = [
        Trace(
            definition="d1",
            workload=Workload(axes={"M": 1}, inputs={"A": RandomInput()}, uuid="w3"),
        ),
        Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="w4"),
        ),
    ]
    save_jsonl_file(traces, pathl)
    loaded_list = load_jsonl_file(Trace, pathl)
    assert len(loaded_list) == 2
    assert loaded_list[0].is_workload_trace()


def test_dict_to_dataclass_with_invalid_fields():
    # Unsupported axis type
    bad_def = {
        "name": "d",
        "op_type": "op",
        "axes": {"M": {"type": "unknown"}},
        "inputs": {"A": {"shape": ["M"], "dtype": "float32"}},
        "outputs": {"B": {"shape": ["M"], "dtype": "float32"}},
        "reference": "def run():\n    pass\n",
    }
    with pytest.raises(ValueError):
        Definition.model_validate(bad_def)


if __name__ == "__main__":
    pytest.main(sys.argv)
