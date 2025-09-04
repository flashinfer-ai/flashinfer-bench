from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from triton.testing import do_bench

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import AxisConst, AxisVar, Definition
from flashinfer_bench.data.solution import Solution
from flashinfer_bench.data.trace import (
    Correctness,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)
from flashinfer_bench.utils import env_snapshot, torch_dtype_from_def


class RunnerError(RuntimeError): ...


class RunnerFatalError(RunnerError): ...


class BaselineHandle(str):
    pass


@dataclass
class DeviceBaseline:
    handle: BaselineHandle
    defn: Definition
    device: str
    inputs_dev: List[Dict[str, Any]]
    ref_outputs_dev: List[Dict[str, torch.Tensor]]
    ref_mean_latency_ms: float


class Runner(ABC):
    """Single-device runner interface."""

    def __init__(self, device: str, log_dir: str = "/tmp/flashinfer_bench") -> None:
        self.device = device
        self._log_dir = log_dir

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @abstractmethod
    def run_ref(
        self,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
    ) -> BaselineHandle:
        """Build a baseline for the given definition and workload."""
        ...

    @abstractmethod
    def run_solution(
        self,
        sol: Solution,
        baseline: BaselineHandle,
        cfg: BenchmarkConfig,
    ) -> Evaluation:
        """Run a solution against the given baseline."""
        ...
