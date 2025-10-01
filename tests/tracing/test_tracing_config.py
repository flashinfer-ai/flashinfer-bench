import sys

import pytest

from flashinfer_bench.tracing.builtin_config import (
    DedupByAxesPolicy,
    KeepAllPolicy,
    KeepFirstKPolicy,
)
from flashinfer_bench.tracing.tracing_config import TracingConfig, WorkloadEntry


def test_factory_creates_independent_instances():
    """Test that dedup_policy factory creates independent instances."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy="keep_all")
    policy1 = config.create_dedup_policy()
    policy2 = config.create_dedup_policy()
    assert policy1 is not policy2


def test_factory_with_lambda():
    """Test that lambda factories work correctly."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: DedupByAxesPolicy(k=5))
    policy = config.create_dedup_policy()
    assert isinstance(policy, DedupByAxesPolicy)
    assert policy.k == 5


def test_state_isolation_between_policies():
    """Test that policy instances have isolated state."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstKPolicy(k=2))
    policy1 = config.create_dedup_policy()
    policy2 = config.create_dedup_policy()

    entry = WorkloadEntry(def_name="test", axes={}, tensors_to_dump={}, order=0)

    policy1.submit(entry)
    assert len(policy1.entries) == 1
    assert len(policy2.entries) == 0


def test_builtin_policies_create_correct_types():
    """Test that builtin policy literals create correct types."""
    configs = [
        ("keep_all", KeepAllPolicy),
        ("keep_first", KeepFirstKPolicy),
        ("dedup_by_axes", DedupByAxesPolicy),
    ]

    for literal, expected_type in configs:
        config = TracingConfig(tensors_to_dump=[], dedup_policy=literal)
        policy = config.create_dedup_policy()
        assert isinstance(policy, expected_type)


if __name__ == "__main__":
    pytest.main(sys.argv)
