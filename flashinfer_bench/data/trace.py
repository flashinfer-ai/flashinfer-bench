"""Strong-typed data definitions for traces and evaluations."""

import math
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class RandomInput(BaseModel):
    """Random input generation descriptor.

    Represents a specification for generating random tensor input data
    during workload execution and benchmarking.
    """

    type: Literal["random"] = "random"
    """The input type identifier for random data generation."""


class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input specification.

    Represents a scalar value (integer, float, or boolean) that will be
    used as a direct input parameter to the computational workload.
    """

    type: Literal["scalar"] = "scalar"
    """The input type identifier for scalar values."""
    value: Union[int, float, bool]
    """The scalar value to be used as input. Must be int, float, or bool."""


class SafetensorsInput(BaseModelWithDocstrings):
    """Input specification for data loaded from safetensors files.

    Represents tensor data that will be loaded from a safetensors file
    using a specific tensor key within that file.
    """

    type: Literal["safetensors"] = "safetensors"
    """The input type identifier for safetensors data."""
    path: NonEmptyString
    """Path to the safetensors file containing the tensor data. The path is relative to the root
    path of the TraceSet."""
    tensor_key: NonEmptyString
    """Key identifier for the specific tensor within the safetensors file."""


InputSpec = Union[RandomInput, SafetensorsInput, ScalarInput]
"""Union type representing all possible input specification types."""


class Workload(BaseModelWithDocstrings):
    """Concrete workload configuration for benchmarking.

    Defines a specific instance of a computational workload with concrete
    values for all variable axes and specifications for all input data.
    This represents an executable configuration that can be benchmarked.
    """

    axes: Dict[str, NonNegativeInt]
    """Dictionary mapping axis names to their concrete integer values. All values must be
    positive."""
    inputs: Dict[str, InputSpec]
    """Dictionary mapping input names to their data specifications."""
    uuid: NonEmptyString
    """Unique identifier for this specific workload configuration."""


class Correctness(BaseModelWithDocstrings):
    """Correctness metrics from numerical evaluation.

    Contains error measurements comparing the solution output against
    a reference implementation to assess numerical accuracy.
    """

    model_config = ConfigDict(ser_json_inf_nan="strings")

    max_relative_error: float = Field(default=0.0)
    """Maximum relative error observed across all output elements."""
    max_absolute_error: float = Field(default=0.0)
    """Maximum absolute error observed across all output elements."""
    extra: Optional[Dict[str, Any]] = Field(default=None)
    """Extra metrics for correctness evaluation."""

    @field_validator("max_relative_error", "max_absolute_error")
    @classmethod
    def non_negative_or_nan(cls, v: float):
        if math.isnan(v):
            return v
        if v < 0:
            raise ValueError("must be non-negative or NaN")
        return v


class Performance(BaseModelWithDocstrings):
    """Performance metrics from timing evaluation.

    Contains timing measurements and performance comparisons from
    benchmarking the solution against reference implementations.
    """

    latency_ms: float = Field(default=0.0, ge=0.0)
    """Solution execution latency in milliseconds."""
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    """Reference implementation latency in milliseconds for comparison."""
    speedup_factor: float = Field(default=0.0, ge=0.0)
    """Performance speedup factor compared to reference (reference_time / solution_time)."""


class Environment(BaseModelWithDocstrings):
    """Environment information from evaluation execution.

    Records the hardware and software environment details from when
    the evaluation was performed, enabling reproducibility analysis.
    """

    hardware: NonEmptyString
    """Hardware identifier where the evaluation was performed (e.g., 'NVIDIA_H100')."""
    libs: Dict[str, str] = Field(default_factory=dict)
    """Dictionary of library names to version strings used during evaluation."""


class EvaluationStatus(str, Enum):
    """Status codes for evaluation results.

    Enumeration of all possible outcomes when evaluating a solution
    against a workload, covering success and various failure modes.
    """

    PASSED = "PASSED"
    """Evaluation completed successfully with correct results."""
    INCORRECT_SHAPE = "INCORRECT_SHAPE"
    """Solution produced output with incorrect tensor shape."""
    INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
    """Solution produced numerically incorrect results."""
    INCORRECT_DTYPE = "INCORRECT_DTYPE"
    """Solution produced output with incorrect data type."""
    RUNTIME_ERROR = "RUNTIME_ERROR"
    """Solution encountered a runtime error during execution."""
    COMPILE_ERROR = "COMPILE_ERROR"
    """Solution failed to compile or build successfully."""
    TIMEOUT = "TIMEOUT"
    """Evaluation did not complete within the configured timeout."""


class Evaluation(BaseModelWithDocstrings):
    """Complete evaluation result for a solution on a workload.

    Records the full outcome of benchmarking a solution implementation
    against a specific workload, including status, metrics, and environment.
    """

    status: EvaluationStatus
    """The overall evaluation status indicating success or failure mode."""
    environment: Environment
    """Environment details where the evaluation was performed."""
    timestamp: NonEmptyString
    """Timestamp when the evaluation was performed (ISO format recommended)."""
    log: str = ""
    """Captured stdout/stderr from the evaluation run."""
    correctness: Optional[Correctness] = None
    """Correctness metrics (present for PASSED and INCORRECT_NUMERICAL status)."""
    performance: Optional[Performance] = None
    """Performance metrics (present only for PASSED status)."""

    @model_validator(mode="after")
    def _validate_status_correctness_performance(self) -> "Evaluation":
        """Validate correctness and performance fields based on status.

        Ensures that correctness and performance metrics are present or absent
        based on the evaluation status, following the schema requirements.

        Raises
        ------
        ValueError
            If correctness/performance presence doesn't match status requirements.
        """
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
        else:
            # For other error statuses, neither correctness nor performance should be present
            if self.correctness is not None:
                raise ValueError(
                    f"Evaluation must not include correctness when status is {self.status}"
                )
            if self.performance is not None:
                raise ValueError(
                    f"Evaluation must not include performance when status is {self.status}"
                )
        return self


class Trace(BaseModelWithDocstrings):
    """Complete trace linking a solution to a definition with evaluation results.

    A Trace represents the complete record of benchmarking a specific solution
    implementation against a specific computational workload definition. It includes
    the workload configuration and evaluation results.

    Special case: A "workload trace" contains only definition and workload fields
    (with solution and evaluation set to None), representing a workload configuration
    without an actual benchmark execution.
    """

    definition: NonEmptyString
    """Name of the Definition that specifies the computational workload."""
    workload: Workload
    """Concrete workload configuration with specific axis values and inputs."""
    solution: Optional[str] = None
    """Name of the Solution implementation (None for workload-only traces)."""
    evaluation: Optional[Evaluation] = None
    """Evaluation results from benchmarking (None for workload-only traces)."""

    def is_workload_trace(self) -> bool:
        """Check if this is a workload-only trace.

        Returns
        -------
        bool
            True if this is a workload trace without solution/evaluation data.
        """
        return self.solution is None and self.evaluation is None

    def is_successful(self) -> bool:
        """Check if the benchmark execution was successful.

        Returns
        -------
        bool
            True if this is a regular trace with successful evaluation status.
            False for workload traces or failed evaluations.
        """
        return (not self.is_workload_trace()) and self.evaluation.status == EvaluationStatus.PASSED
