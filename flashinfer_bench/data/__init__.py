"""Data layer with strongly-typed dataclasses for FlashInfer Bench."""

from .definition import AxisConst, AxisVar, Definition, TensorSpec
from .json_codec import (
    from_json,
    load_json_file,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
    to_json,
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
from .traceset import TraceSet

__all__ = [
    # Definition types
    "AxisConst",
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
    "to_json",
    "from_json",
    "save_json_file",
    "load_json_file",
    "save_jsonl_file",
    "load_jsonl_file",
]
