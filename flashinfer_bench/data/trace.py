"""Strong-typed data definitions for traces and evaluations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Optional, Union


@dataclass
class RandomInput:
    """Random input generation descriptor."""

    type: Literal["random"] = "random"


@dataclass
class SafetensorsInput:
    """Input loaded from a safetensors file."""

    type: Literal["safetensors"] = "safetensors"
    path: str = ""
    tensor_key: str = ""

    def __post_init__(self):
        if not self.path:
            raise ValueError("SafetensorsInput path cannot be empty")
        if not self.tensor_key:
            raise ValueError("SafetensorsInput tensor_key cannot be empty")


# Union type for input descriptors
InputDesc = Union[RandomInput, SafetensorsInput]


@dataclass
class Workload:
    """Concrete workload configuration."""

    axes: Dict[str, int]
    inputs: Dict[str, InputDesc]

    def __post_init__(self):
        if not isinstance(self.axes, dict):
            raise ValueError("Workload axes must be a dictionary")

        # Validate axes values are positive integers
        for axis_name, value in self.axes.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"Workload axis '{axis_name}' must be a positive integer, got {value}"
                )

        if not isinstance(self.inputs, dict):
            raise ValueError("Workload inputs must be a dictionary")

        # Validate inputs are proper types
        for input_name, input_desc in self.inputs.items():
            if not isinstance(input_desc, (RandomInput, SafetensorsInput)):
                raise ValueError(f"Input '{input_name}' must be RandomInput or SafetensorsInput")


@dataclass
class Correctness:
    """Correctness metrics from evaluation."""

    max_relative_error: float = 0.0
    max_absolute_error: float = 0.0

    def __post_init__(self):
        if self.max_relative_error < 0:
            raise ValueError(
                f"max_relative_error must be non-negative, got {self.max_relative_error}"
            )
        if self.max_absolute_error < 0:
            raise ValueError(
                f"max_absolute_error must be non-negative, got {self.max_absolute_error}"
            )


@dataclass
class Performance:
    """Performance metrics from evaluation."""

    latency_ms: float = 0.0
    reference_latency_ms: float = 0.0
    speedup_factor: float = 0.0

    def __post_init__(self):
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if self.reference_latency_ms < 0:
            raise ValueError(
                f"reference_latency_ms must be non-negative, got {self.reference_latency_ms}"
            )
        if self.speedup_factor < 0:
            raise ValueError(f"speedup_factor must be non-negative, got {self.speedup_factor}")


@dataclass
class Environment:
    """Environment information from evaluation."""

    device: str
    libs: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.device:
            raise ValueError("Environment device cannot be empty")
        if not isinstance(self.libs, dict):
            raise ValueError("Environment libs must be a dictionary")


class EvaluationStatus(Enum):
    PASSED = "PASSED"
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILE_ERROR = "COMPILE_ERROR"


@dataclass
class Evaluation:
    """Complete evaluation result for a workload."""

    status: EvaluationStatus
    log_file: str
    environment: Environment
    timestamp: str
    correctness: Optional[Correctness] = None
    performance: Optional[Performance] = None

    def __post_init__(self):
        if not isinstance(self.status, EvaluationStatus):
            raise ValueError("Evaluation status must be of EvaluationStatus type")

        if not self.log_file:
            raise ValueError("Evaluation log_file cannot be empty")

        if self.status == EvaluationStatus.PASSED:
            if self.correctness is None:
                raise ValueError(
                    f"Evaluation must include correctness when status is {self.status}"
                )
            if self.performance is None:
                raise ValueError(
                    f"Evaluation must include performance when status is {self.status}"
                )
        elif self.status == EvaluationStatus.INCORRECT_NUMERICAL:
            if self.correctness is None:
                raise ValueError(
                    f"Evaluation must include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Evaluation must not include performance when status is {self.status}"
                )
        # The rest of the cases do not have correctness and performance
        else:
            if self.correctness is not None:
                raise ValueError(
                    f"Evaluation must not include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Evaluation must not include performance when status is {self.status}"
                )

        if not isinstance(self.environment, Environment):
            raise ValueError("Evaluation environment must be an Environment instance")

        if not self.timestamp:
            raise ValueError("Evaluation timestamp cannot be empty")


@dataclass
class Trace:
    """
    A Trace links a specific Solution to a specific Definition, details the exact
    workload configuration used for the run, and records the complete evaluation result.

    Special case: A "workload trace" only contains definition and workload fields,
    with solution and evaluation set to None. This represents a workload configuration
    without an actual benchmark run.
    """

    definition: str  # Name of the Definition
    workload: Workload
    solution: Optional[str] = None  # Name of the Solution
    evaluation: Optional[Evaluation] = None

    def __post_init__(self):
        if not self.definition:
            raise ValueError("Trace must reference a definition")

        if not isinstance(self.workload, Workload):
            raise ValueError("Trace workload must be a Workload instance")

        # Check if this is a workload-only trace
        is_workload_trace = self.solution is None and self.evaluation is None

        if not is_workload_trace:
            # Regular trace validation
            if self.solution is None:
                raise ValueError("Regular trace must reference a solution")

            if self.evaluation is None:
                raise ValueError("Regular trace must have an evaluation")

            if not isinstance(self.evaluation, Evaluation):
                raise ValueError("Trace evaluation must be an Evaluation instance")

    def is_workload(self) -> bool:
        """Check if this is a workload trace."""
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        """Check if the benchmark run was successful."""
        return (not self.is_workload()) and self.evaluation.status == EvaluationStatus.PASSED
