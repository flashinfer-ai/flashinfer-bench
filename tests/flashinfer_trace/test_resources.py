from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pytest

from flashinfer_bench.data.definition import AxisConst, Definition
from flashinfer_bench.data.json_utils import load_json_file, load_jsonl_file
from flashinfer_bench.data.trace import Trace

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS_DIR = REPO_ROOT / "flashinfer_trace" / "definitions"
WORKLOADS_DIR = REPO_ROOT / "flashinfer_trace" / "workloads"

DEFINITION_FILES: List[Path] = sorted(DEFINITIONS_DIR.rglob("*.json"))
WORKLOAD_FILES: List[Path] = sorted(WORKLOADS_DIR.rglob("*.jsonl"))


@lru_cache(maxsize=None)
def _load_definition(path: Path) -> Definition:
    return load_json_file(Definition, path)


def _definitions_by_name() -> Dict[str, Definition]:
    return {definition.name: definition for definition in map(_load_definition, DEFINITION_FILES)}


def _relative_id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_flashinfer_trace_definition_files_present():
    assert DEFINITION_FILES, "No definition files found under flashinfer_trace/definitions"


def test_flashinfer_trace_workload_files_present():
    assert WORKLOAD_FILES, "No workload files found under flashinfer_trace/workloads"


@pytest.mark.parametrize("definition_path", DEFINITION_FILES, ids=_relative_id)
def test_definition_files_conform_to_schema(definition_path: Path) -> None:
    definition = _load_definition(definition_path)
    assert definition.name == definition_path.stem
    assert set(definition.inputs.keys()).isdisjoint(definition.outputs.keys())


@pytest.mark.parametrize("workload_path", WORKLOAD_FILES, ids=_relative_id)
def test_workload_files_conform_to_schema(workload_path: Path) -> None:
    traces = load_jsonl_file(Trace, workload_path)
    assert traces, "Workload file must contain at least one entry"

    definitions = _definitions_by_name()

    for trace in traces:
        assert trace.solution is None
        assert trace.evaluation is None

        assert trace.definition in definitions
        definition = definitions[trace.definition]

        expected_axes = set(definition.axes.keys())
        assert set(trace.workload.axes.keys()).issubset(expected_axes)

        var_axes = {
            name for name, axis in definition.axes.items() if not isinstance(axis, AxisConst)
        }
        assert var_axes.issubset(trace.workload.axes.keys())

        for axis_name, axis_value in trace.workload.axes.items():
            axis_spec = definition.axes[axis_name]
            if isinstance(axis_spec, AxisConst):
                assert axis_value == axis_spec.value
            else:
                assert axis_value >= 0

        assert set(trace.workload.inputs.keys()) == set(definition.inputs.keys())
