"""Abstract base class for kernel evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.runner import DeviceBaseline
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import Runnable
from flashinfer_bench.data import (
    Correctness,
    Definition,
    Evaluation,
    EvaluationStatus,
    Performance,
    Workload,
)


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def can_evaluate(cls, definition: Definition) -> bool: ...

    @classmethod
    @abstractmethod
    def build_baseline(
        cls,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        trace_set_root: Optional[Path] = None,
    ) -> DeviceBaseline: ...

    @classmethod
    @abstractmethod
    def check_correctness(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_outputs: List[List[torch.Tensor]],
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]: ...

    @classmethod
    @abstractmethod
    def eval_performance(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Performance, Optional[Evaluation]]: ...

    @classmethod
    def evaluate(
        cls,
        definition: Definition,
        sol_runnable: Runnable,
        inputs: List[List[Any]],
        ref_outputs: List[List[torch.Tensor]],
        ref_mean_latency_ms: float,
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Evaluation:
        correctness, evaluation = cls.check_correctness(
            definition=definition,
            sol_runnable=sol_runnable,
            inputs=inputs,
            ref_outputs=ref_outputs,
            cfg=cfg,
            log_path=log_path,
            device=device,
        )
        if evaluation is not None:
            return evaluation

        performance, evaluation = cls.eval_performance(
            definition=definition,
            sol_runnable=sol_runnable,
            inputs=inputs,
            ref_mean_latency_ms=ref_mean_latency_ms,
            cfg=cfg,
            log_path=log_path,
            device=device,
        )

        if evaluation is not None:
            return evaluation

        return make_eval(
            status=EvaluationStatus.PASSED,
            device=device,
            log_path=log_path,
            correctness=correctness,
            performance=performance,
        )
