"""Abstract base class for kernel evaluators."""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
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

from .utils import allocate_outputs, normalize_result


def _first_byte(tensor: torch.Tensor) -> int:
    storage = tensor.untyped_storage()
    return storage[tensor.data_ptr() - storage.data_ptr()]


def _poison_outputs(
    outputs: List[torch.Tensor], protected: List[Any]
) -> List[Tuple[int, int, int | torch.Tensor]]:
    """Poison every element of each non-aliased output."""
    protected_storage = {
        (value.device, value.untyped_storage().data_ptr())
        for value in protected
        if isinstance(value, torch.Tensor)
    }
    poisoned: List[Tuple[int, int, int | torch.Tensor]] = []

    with torch.no_grad():
        for index, output in enumerate(outputs):
            if output.numel() == 0:
                continue
            storage_key = (output.device, output.untyped_storage().data_ptr())
            if storage_key in protected_storage:
                continue

            if output.is_contiguous():
                storage = output.untyped_storage()
                byte_view = torch.empty(0, dtype=torch.uint8, device=output.device).set_(
                    storage, output.data_ptr() - storage.data_ptr(), (output.nbytes,), (1,)
                )
                byte_view.bitwise_not_()
            else:
                first = output[(0,) * output.ndim].item()
                output.fill_(not first if output.dtype == torch.bool else (0 if first != 0 else 1))

            poison = (
                _first_byte(output)
                if output.is_contiguous()
                else output[(0,) * output.ndim].clone()
            )
            poisoned.append((index, output.untyped_storage().data_ptr(), poison))

    return poisoned


def _run_once(
    definition: Definition,
    runnable: Runnable,
    inputs: List[Any],
    device: str,
    outputs: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    if runnable.metadata.destination_passing_style:
        outputs = outputs or allocate_outputs(definition, inputs, device)
        with torch.no_grad():
            runnable(*inputs, *outputs)
    else:
        with torch.no_grad():
            result = runnable(*inputs)
        outputs = normalize_result(definition, result, device)
    torch.cuda.synchronize(device)
    return outputs


def _can_reuse_addresses(target: List[Any], source: List[Any]) -> bool:
    if len(target) != len(source):
        return False
    return all(
        (not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor) and type(a) is type(b))
        or (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and a.device == b.device
        )
        for a, b in zip(target, source)
    )


def _outputs_differ(target: List[torch.Tensor], source: List[torch.Tensor]) -> bool:
    return len(target) != len(source) or any(
        a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b)
        for a, b in zip(target, source)
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
        cfg: ResolvedEvalConfig,
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
        cfg: ResolvedEvalConfig,
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
        cfg: ResolvedEvalConfig,
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
        cfg: ResolvedEvalConfig,
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

        if len(inputs) > 1:
            target_inputs = inputs[0]
            for source_trial in range(1, len(inputs)):
                source_inputs = inputs[source_trial]
                if not _can_reuse_addresses(target_inputs, source_inputs) or not _outputs_differ(
                    ref_outputs[0], ref_outputs[source_trial]
                ):
                    continue

                swapped: List[int] = []
                try:
                    with torch.no_grad():
                        for index, source in enumerate(source_inputs):
                            if isinstance(target_inputs[index], torch.Tensor):
                                temporary = target_inputs[index].clone()
                                target_inputs[index].copy_(source)
                                source.copy_(temporary)
                            else:
                                target_inputs[index], source_inputs[index] = (
                                    source,
                                    target_inputs[index],
                                )
                            swapped.append(index)
                    torch.cuda.synchronize(device)
                    _, challenge_evaluation = cls.check_correctness(
                        definition=definition,
                        sol_runnable=sol_runnable,
                        inputs=[target_inputs],
                        ref_outputs=[ref_outputs[source_trial]],
                        cfg=cfg,
                        log_path=log_path,
                        device=device,
                    )
                finally:
                    with torch.no_grad():
                        for index in reversed(swapped):
                            if isinstance(target_inputs[index], torch.Tensor):
                                temporary = target_inputs[index].clone()
                                target_inputs[index].copy_(source_inputs[index])
                                source_inputs[index].copy_(temporary)
                            else:
                                target_inputs[index], source_inputs[index] = (
                                    source_inputs[index],
                                    target_inputs[index],
                                )
                    torch.cuda.synchronize(device)

                if challenge_evaluation is not None:
                    message = (
                        "Solution returned stale output after inputs changed at stable addresses."
                    )
                    log = challenge_evaluation.log
                    return challenge_evaluation.model_copy(
                        update={"log": f"{log}\n{message}" if log else message}
                    )
                break

        for trial, inp in enumerate(inputs):
            try:
                outputs = _run_once(definition, sol_runnable, inp, device)
                poisoned = _poison_outputs(outputs, [*inp, *ref_outputs[trial]])
                if not poisoned:
                    continue
                torch.cuda.synchronize(device)
                outputs = _run_once(definition, sol_runnable, inp, device, outputs)
            except Exception:
                traceback.print_exc()
                return make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            for index, storage_ptr, poison in poisoned:
                output = outputs[index]
                poison_unchanged = (
                    torch.equal(output[(0,) * output.ndim], poison)
                    if isinstance(poison, torch.Tensor)
                    else _first_byte(output) == poison
                )
                if output.untyped_storage().data_ptr() == storage_ptr and poison_unchanged:
                    return make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                        extra_msg="Solution did not overwrite a poisoned output on replay.",
                    )

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
