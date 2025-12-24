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
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref,
    )
    solution = Solution(
        name="s1",
        definition="d1",
        author="me",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    workload = Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="w1")
    evaluation = Evaluation(
        status=EvaluationStatus.PASSED,
        log="log",
        environment=Environment(hardware="cpu"),
        timestamp="t",
        correctness=Correctness(),
        performance=Performance(),
    )
    trace = Trace(definition="d1", workload=workload, solution="s1", evaluation=evaluation)
    return definition, solution, trace


def test_roundtrip_to_from_json():
    definition, solution, trace = make_minimal_objects()
    definition2 = Definition.model_validate_json(definition.model_dump_json())
    solution2 = Solution.model_validate_json(solution.model_dump_json())
    trace2 = Trace.model_validate_json(trace.model_dump_json())
    assert definition2.name == definition.name
    assert solution2.name == solution.name
    assert trace2.solution == trace.solution


def test_preserve_null_fields_in_trace_json():
    workload = Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="w2")
    trace = Trace(definition="d1", workload=workload)  # workload-only
    obj = trace.model_dump(mode="json")
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
    definition, solution, trace = make_minimal_objects()
    # JSON file roundtrip
    path = tmp_path / "obj.json"
    save_json_file(definition, path)
    loaded = load_json_file(Definition, path)
    assert loaded.name == definition.name

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
