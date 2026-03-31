"""Strong-typed data definitions for traces and evaluations."""

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString
from .workload import Workload


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


class KernelProfile(BaseModelWithDocstrings):
    """NCU profiling data for a single GPU kernel invocation.

    Contains hardware-level profiling information collected via NVIDIA Nsight
    Compute (NCU) for a GPU kernel launch.
    """

    name: str
    """Kernel function name (demangled)."""
    duration_ns: float
    """Kernel execution duration in nanoseconds (gpu__time_duration.sum)."""
    grid: List[int]
    """Grid dimensions [grid_x, grid_y, grid_z]."""
    block: List[int]
    """Block dimensions [block_x, block_y, block_z]."""
    registers_per_thread: int
    """Number of registers used per thread (launch__registers_per_thread)."""
    sm_throughput_pct: float
    """SM throughput as percentage of peak (sm__throughput.avg.pct_of_peak_sustained_elapsed)."""
    dram_throughput_pct: float
    """DRAM throughput as percentage of peak (gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed)."""
    dram_bytes_read: float
    """DRAM bytes read (dram__bytes_read.sum)."""
    dram_bytes_written: float
    """DRAM bytes written (dram__bytes_write.sum)."""
    l1_hit_rate_pct: float
    """L1 cache hit rate percentage (l1tex__t_sector_hit_rate.pct)."""
    l2_hit_rate_pct: float
    """L2 cache hit rate percentage (lts__t_sector_hit_rate.pct)."""
    shared_memory_bytes: float
    """Shared memory allocated per block in bytes (launch__shared_mem_per_block_allocated)."""
    achieved_occupancy_pct: float
    """Achieved occupancy percentage (sm__warps_active.avg.pct_of_peak_sustained_active)."""
    theoretical_occupancy_pct: float
    """Theoretical occupancy percentage (sm__maximum_warps_per_active_cycle_pct)."""
    extra_metrics: Optional[Dict[str, float]] = None
    """Additional NCU metrics not covered by named fields."""


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
    profile: Optional[List[KernelProfile]] = None
    """Per-kernel NCU profiling data (present only for /profile requests)."""


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
