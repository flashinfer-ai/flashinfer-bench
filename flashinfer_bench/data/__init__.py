"""Data layer with strongly-typed dataclasses for FlashInfer Bench."""

from .definition import AxisConst, AxisSpec, AxisVar, Definition, TensorSpec
from .json_utils import (
    append_jsonl_file,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)
from .solution import BuildSpec, Solution, SourceFile, SupportedBindings, SupportedLanguages
from .trace import Correctness, Environment, Evaluation, EvaluationStatus, Performance, Trace
from .trace_set import TraceSet
from .workload import InputSpec, RandomInput, SafetensorsInput, ScalarInput, Workload

__all__ = [
    # Definition types
    "AxisConst",
    "AxisSpec",
    "AxisVar",
    "TensorSpec",
    "Definition",
    # Solution types
    "SourceFile",
    "BuildSpec",
    "SupportedBindings",
    "SupportedLanguages",
    "Solution",
    # Workload types
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "InputSpec",
    "Workload",
    # Trace types
    "Correctness",
    "Performance",
    "Environment",
    "Evaluation",
    "EvaluationStatus",
    "Trace",
    # TraceSet
    "TraceSet",
    # JSON functions
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
    "append_jsonl_file",
]
