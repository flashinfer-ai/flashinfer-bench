import sys

import pytest
import torch

from flashinfer_bench.tracing import (
    BUILTIN_DEDUP_POLICIES,
    BUILTIN_TENSORS_TO_DUMPS,
    KeepAllPolicy,
    KeepFirstByAxesPolicy,
    KeepFirstKPolicy,
    TracingConfig,
    WorkloadEntry,
    dump_all,
    dump_int32,
    dump_none,
)


def test_factory_creates_independent_instances():
    """Test that dedup_policy factory creates independent instances."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy="keep_all")
    policy1 = config.create_dedup_policy()
    policy2 = config.create_dedup_policy()
    assert policy1 is not policy2


def test_factory_with_lambda():
    """Test that lambda factories work correctly."""
    config = TracingConfig(tensors_to_dump=[], dedup_policy=lambda: KeepFirstByAxesPolicy(k=5))
    policy = config.create_dedup_policy()
    assert isinstance(policy, KeepFirstByAxesPolicy)
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
        ("keep_first_by_axes", KeepFirstByAxesPolicy),
    ]

    for literal, expected_type in configs:
        config = TracingConfig(tensors_to_dump=[], dedup_policy=literal)
        policy = config.create_dedup_policy()
        assert isinstance(policy, expected_type)


# ============================================================================
# Tests for Builtin DedupPolicy Implementations
# ============================================================================


def test_keep_all_policy():
    """Test KeepAllPolicy keeps all entries."""
    policy = KeepAllPolicy()
    entries = [
        WorkloadEntry(def_name="test", axes={"n": 10}, tensors_to_dump={}, order=i)
        for i in range(5)
    ]

    for entry in entries:
        policy.submit(entry)

    drained = policy.drain()
    assert len(drained) == 5
    assert drained == entries

    # After drain, buffer should be empty
    assert policy.drain() == []


def test_keep_all_policy_reset():
    """Test KeepAllPolicy reset clears buffer."""
    policy = KeepAllPolicy()
    entry = WorkloadEntry(def_name="test", axes={}, tensors_to_dump={}, order=0)
    policy.submit(entry)
    policy.reset()
    assert policy.drain() == []


def test_keep_first_k_policy():
    """Test KeepFirstKPolicy respects k limit."""
    policy = KeepFirstKPolicy(k=3)

    for i in range(10):
        entry = WorkloadEntry(def_name="test", axes={}, tensors_to_dump={}, order=i)
        policy.submit(entry)

    drained = policy.drain()
    assert len(drained) == 3
    assert [e.order for e in drained] == [0, 1, 2]


def test_keep_first_k_policy_invalid_k():
    """Test KeepFirstKPolicy raises ValueError for invalid k."""
    with pytest.raises(ValueError, match="k must be > 0"):
        KeepFirstKPolicy(k=0)

    with pytest.raises(ValueError, match="k must be > 0"):
        KeepFirstKPolicy(k=-1)


def test_keep_first_by_axes_policy():
    """Test KeepFirstByAxesPolicy deduplicates by axes."""
    policy = KeepFirstByAxesPolicy(k=2)

    # Submit entries with different axes
    entries = [
        WorkloadEntry(def_name="test", axes={"n": 10, "m": 20}, tensors_to_dump={}, order=0),
        WorkloadEntry(def_name="test", axes={"n": 10, "m": 20}, tensors_to_dump={}, order=1),
        WorkloadEntry(
            def_name="test", axes={"n": 10, "m": 20}, tensors_to_dump={}, order=2
        ),  # Exceeds k
        WorkloadEntry(def_name="test", axes={"n": 15, "m": 20}, tensors_to_dump={}, order=3),
        WorkloadEntry(def_name="test", axes={"n": 15, "m": 20}, tensors_to_dump={}, order=4),
        WorkloadEntry(
            def_name="test", axes={"n": 15, "m": 20}, tensors_to_dump={}, order=5
        ),  # Exceeds k
    ]

    for entry in entries:
        policy.submit(entry)

    drained = policy.drain()
    # Should keep first 2 for each unique axes combination
    assert len(drained) == 4
    assert [e.order for e in drained] == [0, 1, 3, 4]


def test_keep_first_by_axes_policy_axes_order_invariant():
    """Test that axes order doesn't matter for KeepFirstByAxesPolicy."""
    policy = KeepFirstByAxesPolicy(k=1)

    entry1 = WorkloadEntry(def_name="test", axes={"n": 10, "m": 20}, tensors_to_dump={}, order=0)
    entry2 = WorkloadEntry(def_name="test", axes={"m": 20, "n": 10}, tensors_to_dump={}, order=1)

    policy.submit(entry1)
    policy.submit(entry2)

    drained = policy.drain()
    # entry2 should be deduplicated since axes are the same
    assert len(drained) == 1
    assert drained[0].order == 0


def test_keep_first_by_axes_policy_reset():
    """Test KeepFirstByAxesPolicy reset clears seen counts."""
    policy = KeepFirstByAxesPolicy(k=1)

    entry = WorkloadEntry(def_name="test", axes={"n": 10}, tensors_to_dump={}, order=0)
    policy.submit(entry)
    policy.reset()

    # After reset, should be able to submit same axes again
    entry2 = WorkloadEntry(def_name="test", axes={"n": 10}, tensors_to_dump={}, order=1)
    policy.submit(entry2)
    drained = policy.drain()
    assert len(drained) == 1
    assert drained[0].order == 1


def test_builtin_dedup_policies_completeness():
    """Test that all builtin policy factories are registered."""
    expected_literals = ["keep_all", "keep_first", "keep_first_by_axes"]

    for literal in expected_literals:
        assert literal in BUILTIN_DEDUP_POLICIES
        factory = BUILTIN_DEDUP_POLICIES[literal]
        policy = factory()
        # Verify it has the protocol methods
        assert hasattr(policy, "submit")
        assert hasattr(policy, "drain")
        assert hasattr(policy, "reset")


# ============================================================================
# Tests for TensorsToDump Functions
# ============================================================================


def test_dump_all_function():
    """Test dump_all returns all tensor names."""
    inputs = {
        "tensor1": torch.zeros(10),
        "tensor2": torch.ones(5, 5),
        "scalar": 42,
        "string": "test",
    }

    result = dump_all(inputs)
    assert set(result) == {"tensor1", "tensor2"}


def test_dump_none_function():
    """Test dump_none returns empty list."""
    inputs = {"tensor1": torch.zeros(10), "tensor2": torch.ones(5)}
    result = dump_none(inputs)
    assert result == []


def test_dump_int32_function():
    """Test dump_int32 only returns int32 tensors."""
    inputs = {
        "int32_tensor": torch.zeros(10, dtype=torch.int32),
        "float32_tensor": torch.zeros(5, dtype=torch.float32),
        "int64_tensor": torch.zeros(3, dtype=torch.int64),
        "another_int32": torch.ones(2, dtype=torch.int32),
        "scalar": 42,
    }

    result = dump_int32(inputs)
    assert set(result) == {"int32_tensor", "another_int32"}


def test_builtin_tensors_to_dumps_completeness():
    """Test that all builtin dump functions are registered."""
    expected_literals = ["dump_all", "dump_none", "dump_int32"]

    for literal in expected_literals:
        assert literal in BUILTIN_TENSORS_TO_DUMPS
        func = BUILTIN_TENSORS_TO_DUMPS[literal]
        assert callable(func)


# ============================================================================
# Tests for TracingConfig Dataclass
# ============================================================================


def test_tracing_config_with_literal_tensors_to_dump():
    """Test TracingConfig with string literal for tensors_to_dump."""
    config = TracingConfig(tensors_to_dump="dump_all", dedup_policy="keep_all")

    # After __post_init__, should be resolved to function
    assert callable(config.tensors_to_dump)

    # Test it works
    inputs = {"tensor": torch.zeros(5)}
    result = config.get_tensors_to_dump(inputs)
    assert result == ["tensor"]


def test_tracing_config_with_list_tensors_to_dump():
    """Test TracingConfig with static list for tensors_to_dump."""
    config = TracingConfig(tensors_to_dump=["tensor1", "tensor2"], dedup_policy="keep_all")

    inputs = {"tensor1": torch.zeros(5), "tensor2": torch.ones(3), "tensor3": torch.zeros(2)}
    result = config.get_tensors_to_dump(inputs)
    assert result == ["tensor1", "tensor2"]


def test_tracing_config_with_callable_tensors_to_dump():
    """Test TracingConfig with custom callable for tensors_to_dump."""

    def custom_dump(inputs):
        return [name for name in inputs.keys() if name.startswith("test_")]

    config = TracingConfig(tensors_to_dump=custom_dump, dedup_policy="keep_all")

    inputs = {"test_a": torch.zeros(5), "other": torch.ones(3), "test_b": torch.zeros(2)}
    result = config.get_tensors_to_dump(inputs)
    assert set(result) == {"test_a", "test_b"}


def test_tracing_config_invalid_tensors_to_dump_literal():
    """Test TracingConfig raises error for invalid tensors_to_dump literal."""
    with pytest.raises(ValueError, match="Unknown tensors_to_dump literal"):
        TracingConfig(tensors_to_dump="invalid_literal", dedup_policy="keep_all")


def test_tracing_config_invalid_dedup_policy_literal():
    """Test TracingConfig raises error for invalid dedup_policy literal."""
    with pytest.raises(ValueError, match="Unknown dedup_policy literal"):
        TracingConfig(tensors_to_dump=[], dedup_policy="invalid_policy")


def test_tracing_config_get_tensors_to_dump_validation():
    """Test that get_tensors_to_dump validates results."""

    # Test with invalid tensor name
    def bad_dump_func(inputs):
        return ["nonexistent_tensor"]

    config = TracingConfig(tensors_to_dump=bad_dump_func, dedup_policy="keep_all")
    inputs = {"real_tensor": torch.zeros(5)}

    with pytest.raises(ValueError, match="not in runtime_args"):
        config.get_tensors_to_dump(inputs)


def test_tracing_config_all_literal_combinations():
    """Test all valid literal combinations work."""
    dump_literals = ["dump_all", "dump_none", "dump_int32"]
    policy_literals = ["keep_all", "keep_first", "keep_first_by_axes"]

    for dump_lit in dump_literals:
        for policy_lit in policy_literals:
            config = TracingConfig(tensors_to_dump=dump_lit, dedup_policy=policy_lit)
            assert config is not None
            # Should be resolved to callables
            assert callable(config.tensors_to_dump)
            assert callable(config.dedup_policy)


if __name__ == "__main__":
    pytest.main(sys.argv)
