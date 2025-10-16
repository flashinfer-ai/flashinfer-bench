"""Data layer with strongly-typed dataclasses for FlashInfer Bench."""

from .definition import AxisConst, AxisSpec, AxisVar, Definition, TensorSpec
from .json_utils import (
    append_jsonl_file,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)
from .solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from .trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    InputSpec,
    Performance,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    Trace,
    Workload,
)
from .trace_set import TraceSet

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
    "SupportedLanguages",
    "Solution",
    # Trace types
    "RandomInput",
    "ScalarInput",
    "SafetensorsInput",
    "InputSpec",
    "Workload",
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
