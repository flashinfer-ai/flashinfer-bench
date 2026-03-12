"""Utilities for classifying benchmark outcomes into finer-grained error buckets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from flashinfer_bench.data import Correctness, Evaluation, EvaluationStatus, Performance, Trace


@dataclass(frozen=True)
class EvaluationTaxonomy:
    """A lightweight secondary classification for benchmark results."""

    status_family: str
    secondary_bucket: str
    efficiency_bucket: Optional[str] = None


def _contains_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _normalize_log(log: str) -> str:
    return (log or "").strip().lower()


def _classify_compile_error(log_text: str) -> str:
    if _contains_any(
        log_text,
        "destination-passing style callable",
        "expected 3 parameters",
        "expected 4 parameters",
        "expected 5 parameters",
        "expected 6 parameters",
    ):
        return "compile.signature_mismatch"
    if _contains_any(log_text, "entry symbol", "exported symbol", "entry point") and _contains_any(
        log_text, "not found", "missing"
    ):
        return "compile.entry_symbol_missing"
    if _contains_any(
        log_text,
        "failed importing module",
        "modulenotfounderror",
        "importerror",
        "no module named",
    ):
        return "compile.import_failure"
    if _contains_any(log_text, "syntaxerror", "indentationerror", "invalid syntax"):
        return "compile.syntax"
    if _contains_any(log_text, "undefined reference", "ld:", "linker", "cannot find -l"):
        return "compile.linker"
    if _contains_any(
        log_text,
        "no registered builder",
        "not available in the current environment",
        "cublas",
        "cudnn",
        "cutlass",
    ):
        return "compile.dependency"
    if _contains_any(
        log_text,
        "nvcc",
        "ninja exited",
        "compilation failed",
        "error: variable",
        "error: identifier",
        "error: expected",
    ):
        return "compile.cuda_compile"
    return "compile.other"


def _classify_runtime_error(log_text: str) -> str:
    if _contains_any(log_text, "out of memory", "cuda out of memory", "std::bad_alloc"):
        return "runtime.out_of_memory"
    if _contains_any(
        log_text,
        "illegal memory access",
        "misaligned address",
        "memory access violation",
    ):
        return "runtime.illegal_memory_access"
    if _contains_any(
        log_text,
        "invalid configuration argument",
        "invalid device function",
        "too many resources requested",
        "unspecified launch failure",
        "device-side assert",
        "kernel launch failed",
    ):
        return "runtime.cuda_launch"
    if _contains_any(
        log_text,
        "must be a cuda tensor",
        "same device",
        "device mismatch",
        "cuda is not available",
    ):
        return "runtime.device_mismatch"
    if _contains_any(
        log_text,
        "attributeerror",
        "nameerror",
        "typeerror",
        "valueerror",
        "indexerror",
        "keyerror",
    ):
        return "runtime.python_exception"
    return "runtime.other"


def _classify_correctness_error(
    correctness: Optional[Correctness],
    log_text: str,
    status: EvaluationStatus,
) -> str:
    if status == EvaluationStatus.INCORRECT_SHAPE:
        return "correctness.shape"
    if status == EvaluationStatus.INCORRECT_DTYPE:
        return "correctness.dtype"

    if correctness is not None:
        if math.isnan(correctness.max_absolute_error) or math.isnan(correctness.max_relative_error):
            return "correctness.nonfinite"
        if math.isinf(correctness.max_absolute_error) or math.isinf(correctness.max_relative_error):
            return "correctness.nonfinite"

    if _contains_any(log_text, "nan", "inf"):
        return "correctness.nonfinite"
    return "correctness.numerical"


def classify_efficiency(
    performance: Optional[Performance],
    slower_than_ref_threshold: float = 1.0,
    breakout_speedup_threshold: float = 2.0,
) -> Optional[str]:
    """Classify the performance of a passed trace into coarse efficiency buckets."""
    if performance is None:
        return None

    speedup = performance.speedup_factor
    if speedup < slower_than_ref_threshold:
        return "efficiency.regression"
    if speedup < 1.1:
        return "efficiency.parity"
    if speedup < breakout_speedup_threshold:
        return "efficiency.speedup"
    return "efficiency.breakout"


def classify_evaluation(
    evaluation: Optional[Evaluation],
    slower_than_ref_threshold: float = 1.0,
    breakout_speedup_threshold: float = 2.0,
) -> EvaluationTaxonomy:
    """Map an Evaluation to a coarse status family and a finer secondary bucket."""
    if evaluation is None:
        return EvaluationTaxonomy(status_family="missing", secondary_bucket="missing.evaluation")

    log_text = _normalize_log(evaluation.log)
    status = evaluation.status

    if status == EvaluationStatus.PASSED:
        return EvaluationTaxonomy(
            status_family="passed",
            secondary_bucket="passed",
            efficiency_bucket=classify_efficiency(
                evaluation.performance,
                slower_than_ref_threshold=slower_than_ref_threshold,
                breakout_speedup_threshold=breakout_speedup_threshold,
            ),
        )

    if status == EvaluationStatus.TIMEOUT:
        return EvaluationTaxonomy(status_family="timeout", secondary_bucket="timeout.execution")

    if status == EvaluationStatus.COMPILE_ERROR:
        return EvaluationTaxonomy(
            status_family="compile_error",
            secondary_bucket=_classify_compile_error(log_text),
        )

    if status == EvaluationStatus.RUNTIME_ERROR:
        return EvaluationTaxonomy(
            status_family="runtime_error",
            secondary_bucket=_classify_runtime_error(log_text),
        )

    if status in (
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.INCORRECT_NUMERICAL,
    ):
        return EvaluationTaxonomy(
            status_family="correctness_error",
            secondary_bucket=_classify_correctness_error(
                evaluation.correctness,
                log_text,
                status,
            ),
        )

    return EvaluationTaxonomy(
        status_family="unknown",
        secondary_bucket=f"unknown.{status.value.lower()}",
    )


def classify_trace(
    trace: Trace,
    slower_than_ref_threshold: float = 1.0,
    breakout_speedup_threshold: float = 2.0,
) -> EvaluationTaxonomy:
    """Convenience wrapper for classifying a Trace."""
    return classify_evaluation(
        trace.evaluation,
        slower_than_ref_threshold=slower_than_ref_threshold,
        breakout_speedup_threshold=breakout_speedup_threshold,
    )
