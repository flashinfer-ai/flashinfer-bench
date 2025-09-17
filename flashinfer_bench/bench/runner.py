from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution
from flashinfer_bench.data.trace import Evaluation, Workload

from .config import BenchmarkConfig


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
    def run_ref(
        self,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        traceset_root: Optional[Path] = None,
    ) -> BaselineHandle:
        """Build a baseline for the given definition and workload."""
        ...

    @abstractmethod
    def run_solution(
        self, sol: Solution, baseline: BaselineHandle, cfg: BenchmarkConfig
    ) -> Evaluation:
        """Run a solution against the given baseline."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @abstractmethod
    def release(self, baseline: BaselineHandle) -> None:
        """Release a baseline."""
        ...
