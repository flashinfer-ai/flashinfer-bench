"""Tests for adapting SGLang Dumper captures into tracing workloads."""

import json
from pathlib import Path

import torch

from flashinfer_bench.data import AxisVar, Definition, TensorSpec, TraceSet
from flashinfer_bench.tracing.sglang_dumper_adapter import (
    adapt_sglang_dumper_workloads,
    build_sglang_dumper_workload_filter,
    merge_sglang_workload_shards,
)


def _write_module_definition(root: Path) -> Path:
    definition = Definition(
        name="module_scale_n",
        op_type="test_module",
        axes={"n": AxisVar()},
        inputs={
            "x": TensorSpec(shape=["n"], dtype="float32"),
            "scale": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={"y": TensorSpec(shape=["n"], dtype="float32")},
        reference="def run(x, scale):\n    return x * scale\n",
        tags=[
            "sglang_module:test.FakeModule",
            "sglang_input:x=arg:0",
            "sglang_input:scale=attr:scale",
        ],
    )
    path = root / "definitions" / "test_module" / "module_scale_n.json"
    path.parent.mkdir(parents=True)
    path.write_text(definition.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_build_sglang_dumper_workload_filter(tmp_path: Path):
    definition_path = _write_module_definition(tmp_path)
    inventory = {
        "modules": [
            {
                "class_path": "test.FakeModule",
                "module_paths": ["model.layers.1.op", "model.layers.0.op"],
            }
        ]
    }

    filter_expression, diagnostics = build_sglang_dumper_workload_filter(
        [definition_path], inventory
    )

    assert diagnostics["selected_modules"] == {
        "test.FakeModule": "model.layers.0.op"
    }
    assert "model\\\\.layers\\\\.0\\\\.op" in filter_expression
    assert "inputs" in filter_expression
    assert "output" in filter_expression


def test_build_sglang_dumper_filter_uses_first_numeric_shared_module_path(tmp_path: Path):
    definition_path = _write_module_definition(tmp_path)
    inventory = {
        "modules": [
            {
                "class_path": "test.FakeModule",
                "module_paths": [
                    "model.layers.14.op",
                    "model.layers.4.op",
                    "model.layers.9.op",
                ],
            }
        ]
    }

    filter_expression, diagnostics = build_sglang_dumper_workload_filter(
        [definition_path], inventory
    )

    assert diagnostics["selected_modules"] == {
        "test.FakeModule": "model.layers.4.op"
    }
    assert "model\\\\.layers\\\\.4\\\\.op" in filter_expression


def test_adapt_sglang_dumper_workloads(tmp_path: Path):
    definition_path = _write_module_definition(tmp_path)
    capture_root = tmp_path / "capture"
    binding_path = capture_root / "module_bindings" / "123" / "00001.pt"
    binding_path.parent.mkdir(parents=True)
    torch.save(
        {
            "definition": "module_scale_n",
            "class_path": "test.FakeModule",
            "module_path": "model.layers.0.op",
            "argument_positions": {"x": 0},
            "attributes": {"scale": 2.0},
        },
        binding_path,
    )

    dump_root = tmp_path / "sglang_dumper"
    dump_dir = dump_root / "onboarding_default" / "TP0_PP0_Rank0_pid123"
    dump_dir.mkdir(parents=True)
    torch.save(
        {
            "value": torch.arange(4, dtype=torch.float32),
            "meta": {
                "name": "non_intrusive__model.layers.0.op.inputs.0",
                "rank": 0,
                "step": 0,
                "dump_index": 1,
            },
        },
        dump_dir / "00001.pt",
    )
    torch.save(
        {
            "value": torch.arange(4, dtype=torch.float32) * 2,
            "meta": {
                "name": "non_intrusive__model.layers.0.op.output",
                "rank": 0,
                "step": 0,
                "dump_index": 2,
            },
        },
        dump_dir / "00002.pt",
    )

    diagnostics = adapt_sglang_dumper_workloads(
        dump_root,
        capture_root=capture_root,
        definition_files=[definition_path],
    )
    dataset_dir = tmp_path / "dataset"
    counts = merge_sglang_workload_shards(
        capture_root,
        dataset_dir=dataset_dir,
        definition_files=[definition_path],
        max_new_workloads=4,
    )

    assert diagnostics["summary"] == {
        "dump_files": 2,
        "module_calls": 1,
        "collect_attempts": 1,
        "collect_accepted": 1,
        "collect_rejected": 0,
        "missing_bindings": 0,
        "errors": 0,
    }
    assert diagnostics["collect_results"] == {
        "module_scale_n": {
            "attempts": 1,
            "accepted": 1,
            "rejected": 0,
            "reasons": {},
        }
    }
    assert counts == {"module_scale_n": 1}
    trace_set = TraceSet.from_path(dataset_dir)
    assert len(trace_set.workloads["module_scale_n"]) == 1
    assert trace_set.workloads["module_scale_n"][0].workload.axes == {"n": 4}


def test_adapt_sglang_dumper_reports_collect_rejection(tmp_path: Path):
    definition_path = _write_module_definition(tmp_path)
    capture_root = tmp_path / "capture"
    binding_path = capture_root / "module_bindings" / "123" / "00001.pt"
    binding_path.parent.mkdir(parents=True)
    torch.save(
        {
            "definition": "module_scale_n",
            "class_path": "test.FakeModule",
            "module_path": "model.layers.0.op",
            "argument_positions": {"x": 0},
            "attributes": {"scale": 2.0},
        },
        binding_path,
    )

    dump_dir = (
        tmp_path
        / "sglang_dumper"
        / "onboarding_default"
        / "TP0_PP0_Rank0_pid123"
    )
    dump_dir.mkdir(parents=True)
    for index, (name, value) in enumerate(
        [
            ("inputs.0", torch.zeros(2, 2)),
            ("output", torch.zeros(2, 2)),
        ],
        start=1,
    ):
        torch.save(
            {
                "value": value,
                "meta": {
                    "name": f"non_intrusive__model.layers.0.op.{name}",
                    "rank": 0,
                    "step": 0,
                    "dump_index": index,
                },
            },
            dump_dir / f"{index:05d}.pt",
        )

    diagnostics = adapt_sglang_dumper_workloads(
        tmp_path / "sglang_dumper",
        capture_root=capture_root,
        definition_files=[definition_path],
    )

    assert diagnostics["summary"]["collect_accepted"] == 0
    assert diagnostics["summary"]["collect_rejected"] == 1
    result = diagnostics["collect_results"]["module_scale_n"]
    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert next(iter(result["reasons"])).startswith("axis_inference_failed:")
