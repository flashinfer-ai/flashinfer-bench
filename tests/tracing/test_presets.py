"""Tests for builtin filter policies and input dump policies."""

import sys

import pytest
import torch

from flashinfer_bench.tracing import PolicyRegistry, WorkloadEntry
from flashinfer_bench.tracing.presets import (
    DumpAllPolicy,
    DumpIntPolicy,
    DumpNonePolicy,
    KeepAllPolicy,
    KeepFirstKByAxesPolicy,
    KeepFirstKPolicy,
    KeepFirstPolicy,
)

# ============================================================================
# Completeness
# ============================================================================


def test_filter_policies_registered():
    """Test all builtin filter policies are registered."""
    expected = ["keep_all", "keep_first", "keep_first_k", "keep_first_k_by_axes", "keep_none"]
    for name in expected:
        cls = PolicyRegistry.get_filter_policy(name)
        assert cls is not None, f"Filter policy '{name}' not registered"


def test_input_dump_policies_registered():
    """Test all builtin input dump policies are registered."""
    expected = ["dump_all", "dump_none", "dump_int"]
    for name in expected:
        cls = PolicyRegistry.get_input_dump_policy(name)
        assert cls is not None, f"Input dump policy '{name}' not registered"


# ============================================================================
# KeepAllPolicy
# ============================================================================


def test_keep_all():
    """Test KeepAllPolicy keeps all entries."""
    policy = KeepAllPolicy()
    entries = [
        WorkloadEntry(def_name="test", axes={"n": i}, inputs_to_dump={}, order=i) for i in range(5)
    ]
    for e in entries:
        policy.submit(e)

    drained = policy.drain()
    assert len(drained) == 5
    assert policy.drain() == []  # Empty after drain


def test_keep_all_reset():
    """Test KeepAllPolicy.reset() clears buffer."""
    policy = KeepAllPolicy()
    policy.submit(WorkloadEntry(def_name="test", axes={}, inputs_to_dump={}, order=0))
    policy.reset()
    assert policy.drain() == []


# ============================================================================
# KeepFirstKPolicy
# ============================================================================


def test_keep_first_k():
    """Test KeepFirstKPolicy respects k limit."""
    policy = KeepFirstKPolicy(k=3)
    for i in range(10):
        policy.submit(WorkloadEntry(def_name="test", axes={}, inputs_to_dump={}, order=i))

    drained = policy.drain()
    assert len(drained) == 3
    assert [e.order for e in drained] == [0, 1, 2]


def test_keep_first_k_invalid():
    """Test KeepFirstKPolicy rejects invalid k."""
    with pytest.raises(ValueError, match="k must be > 0"):
        KeepFirstKPolicy(k=0)
    with pytest.raises(ValueError, match="k must be > 0"):
        KeepFirstKPolicy(k=-1)


# ============================================================================
# KeepFirstKByAxesPolicy
# ============================================================================


def test_keep_first_k_by_axes():
    """Test KeepFirstKByAxesPolicy deduplicates by axes."""
    policy = KeepFirstKByAxesPolicy(k=2)
    entries = [
        WorkloadEntry(def_name="test", axes={"n": 10}, inputs_to_dump={}, order=0),
        WorkloadEntry(def_name="test", axes={"n": 10}, inputs_to_dump={}, order=1),
        WorkloadEntry(def_name="test", axes={"n": 10}, inputs_to_dump={}, order=2),  # Exceeds k
        WorkloadEntry(def_name="test", axes={"n": 20}, inputs_to_dump={}, order=3),
        WorkloadEntry(def_name="test", axes={"n": 20}, inputs_to_dump={}, order=4),
    ]
    for e in entries:
        policy.submit(e)

    drained = policy.drain()
    assert len(drained) == 4
    assert [e.order for e in drained] == [0, 1, 3, 4]


def test_keep_first_k_by_axes_order():
    """Test KeepFirstKByAxesPolicy is axes order invariant."""
    policy = KeepFirstKByAxesPolicy(k=1)
    policy.submit(WorkloadEntry(def_name="test", axes={"a": 1, "b": 2}, inputs_to_dump={}, order=0))
    policy.submit(WorkloadEntry(def_name="test", axes={"b": 2, "a": 1}, inputs_to_dump={}, order=1))

    drained = policy.drain()
    assert len(drained) == 1
    assert drained[0].order == 0


def test_keep_first_k_by_axes_reset():
    """Test KeepFirstKByAxesPolicy.reset() clears state."""
    policy = KeepFirstKByAxesPolicy(k=1)
    policy.submit(WorkloadEntry(def_name="test", axes={"n": 10}, inputs_to_dump={}, order=0))
    policy.reset()

    # After reset, same axes should be accepted again
    policy.submit(WorkloadEntry(def_name="test", axes={"n": 10}, inputs_to_dump={}, order=1))
    drained = policy.drain()
    assert len(drained) == 1
    assert drained[0].order == 1


# ============================================================================
# KeepFirstPolicy
# ============================================================================


def test_keep_first():
    """Test KeepFirstPolicy keeps only first entry."""
    policy = KeepFirstPolicy()
    for i in range(5):
        policy.submit(WorkloadEntry(def_name="test", axes={}, inputs_to_dump={}, order=i))

    drained = policy.drain()
    assert len(drained) == 1
    assert drained[0].order == 0


# ============================================================================
# DumpAllPolicy
# ============================================================================


def test_dump_all():
    """Test DumpAllPolicy returns all tensor names."""
    policy = DumpAllPolicy()
    inputs = {"tensor1": torch.zeros(10), "tensor2": torch.ones(5), "scalar": 42, "string": "test"}
    result = policy.dump(inputs)
    assert set(result) == {"tensor1", "tensor2"}


# ============================================================================
# DumpNonePolicy
# ============================================================================


def test_dump_none():
    """Test DumpNonePolicy returns empty list."""
    policy = DumpNonePolicy()
    result = policy.dump({"tensor": torch.zeros(10)})
    assert result == []


# ============================================================================
# DumpIntPolicy
# ============================================================================


def test_dump_int():
    """Test DumpIntPolicy returns only integer tensor names."""
    policy = DumpIntPolicy()
    inputs = {
        "int32": torch.zeros(10, dtype=torch.int32),
        "int64": torch.zeros(5, dtype=torch.int64),
        "float32": torch.zeros(3, dtype=torch.float32),
        "bool": torch.zeros(2, dtype=torch.bool),
    }
    result = policy.dump(inputs)
    assert set(result) == {"int32", "int64", "bool"}


if __name__ == "__main__":
    pytest.main(sys.argv)
