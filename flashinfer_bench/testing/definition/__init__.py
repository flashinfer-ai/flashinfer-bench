"""Definition-based testing framework.

This module provides a testing framework that automatically extracts reference
implementations from Definition JSON files and compares them against baseline
implementations (e.g., FlashInfer kernels).

Example usage:

    from flashinfer_bench.testing.definition import DefinitionTestCase

    class TestGQAPagedDecode(DefinitionTestCase):
        definition_path = "flashinfer_trace/definitions/gqa_paged/gqa_paged_decode.json"
        configs = [
            {"batch_size": 1, "num_pages": 100},
            {"batch_size": 4, "num_pages": 200},
        ]

        def baseline_fn(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
            # Wrap FlashInfer to match reference interface
            ...
"""

from flashinfer_bench.testing.definition.comparators import (
    Comparator,
    CompareResult,
    HitRatioComparator,
    MultiOutputComparator,
    TensorComparator,
)
from flashinfer_bench.testing.definition.core import DefinitionTest, TestResult
from flashinfer_bench.testing.definition.pytest_base import DefinitionTestCase

__all__ = [
    "DefinitionTest",
    "DefinitionTestCase",
    "CompareResult",
    "TestResult",
    "Comparator",
    "TensorComparator",
    "MultiOutputComparator",
    "HitRatioComparator",
]
