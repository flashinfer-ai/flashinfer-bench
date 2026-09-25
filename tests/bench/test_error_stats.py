"""Tests for numerical error statistics."""

import math

import pytest
import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
from flashinfer_bench.bench.utils import compute_error_stats


@pytest.mark.parametrize(
    ("output_value", "reference_value"),
    [(float("inf"), float("inf")), (float("-inf"), float("-inf"))],
    ids=["positive_infinity", "negative_infinity"],
)
def test_compute_error_stats_accepts_matching_infinities(
    output_value: float, reference_value: float
) -> None:
    cfg = ResolvedEvalConfig(atol=1e-4, rtol=1e-4)

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(
        torch.tensor([output_value]), torch.tensor([reference_value]), cfg
    )

    assert not exceeds_tol
    assert matched_ratio == 1.0
    assert max_abs == 0.0
    assert max_rel == 0.0


@pytest.mark.parametrize(
    ("output_value", "reference_value"),
    [
        (float("inf"), float("-inf")),
        (float("-inf"), float("inf")),
        (0.0, float("-inf")),
        (float("-inf"), 0.0),
        (float("nan"), float("nan")),
        (float("nan"), 0.0),
        (0.0, float("nan")),
    ],
    ids=[
        "opposite_infinities_positive_negative",
        "opposite_infinities_negative_positive",
        "finite_against_infinity",
        "infinity_against_finite",
        "matching_nans",
        "nan_against_finite",
        "finite_against_nan",
    ],
)
def test_compute_error_stats_rejects_non_finite_mismatches(
    output_value: float, reference_value: float
) -> None:
    cfg = ResolvedEvalConfig(atol=1e-4, rtol=1e-4)

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(
        torch.tensor([output_value]), torch.tensor([reference_value]), cfg
    )

    assert exceeds_tol
    assert matched_ratio == 0.0
    assert math.isinf(max_abs)
    assert math.isinf(max_rel)


def test_non_finite_mismatch_is_fatal_with_relaxed_matched_ratio() -> None:
    cfg = ResolvedEvalConfig(atol=1e-2, rtol=1e-2, required_matched_ratio=0.95)
    output = torch.zeros(100)
    reference = torch.zeros(100)
    reference[0] = float("-inf")

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(output, reference, cfg)

    assert matched_ratio == pytest.approx(0.99)
    assert exceeds_tol
    assert math.isinf(max_abs)
    assert math.isinf(max_rel)


def test_matching_infinity_does_not_pollute_finite_error_metrics() -> None:
    cfg = ResolvedEvalConfig(atol=1e-2, rtol=1e-3)
    output = torch.tensor([float("-inf"), 1.005])
    reference = torch.tensor([float("-inf"), 1.0])

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(output, reference, cfg)

    assert not exceeds_tol
    assert matched_ratio == 1.0
    assert max_abs == pytest.approx(0.005, rel=1e-5)
    assert max_rel == pytest.approx(0.005, rel=1e-5)


def test_finite_tolerance_semantics_are_unchanged() -> None:
    cfg = ResolvedEvalConfig(atol=0.1, rtol=0.01)
    output = torch.tensor([1.05, 100.5, 3.0])
    reference = torch.tensor([1.0, 100.0, 1.0])

    max_abs, max_rel, exceeds_tol, matched_ratio = compute_error_stats(output, reference, cfg)

    assert exceeds_tol
    assert matched_ratio == pytest.approx(2.0 / 3.0)
    assert max_abs == pytest.approx(2.0)
    assert max_rel == pytest.approx(2.0)
