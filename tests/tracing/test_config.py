"""Tests for TracingConfig, TracingConfigRegistry, and PolicyRegistry."""

import sys

import pytest
import torch

from flashinfer_bench.tracing import (
    PolicyRegistry,
    TracingConfig,
    TracingConfigRegistry,
    WorkloadEntry,
)

# ============================================================================
# TracingConfig
# ============================================================================


def test_config_defaults():
    """Test TracingConfig default values."""
    config = TracingConfig()
    assert config.input_dump_policy == "dump_none"
    assert config.filter_policy == "keep_all"
    assert config.filter_policy_kwargs == {}


def test_config_with_kwargs():
    """Test TracingConfig with filter_policy_kwargs."""
    config = TracingConfig(filter_policy="keep_first_k", filter_policy_kwargs={"k": 5})
    policy = config.create_filter_policy()
    assert policy.k == 5


def test_config_invalid_input_policy():
    """Test TracingConfig rejects invalid input_dump_policy."""
    with pytest.raises(ValueError, match="Unknown input_dump_policy"):
        TracingConfig(input_dump_policy="invalid")


def test_config_invalid_filter_policy():
    """Test TracingConfig rejects invalid filter_policy."""
    with pytest.raises(ValueError, match="Unknown filter_policy"):
        TracingConfig(filter_policy="invalid")


# ============================================================================
# create_filter_policy
# ============================================================================


def test_create_filter_policy():
    """Test create_filter_policy creates new instances."""
    config = TracingConfig(filter_policy="keep_all")
    p1 = config.create_filter_policy()
    p2 = config.create_filter_policy()
    assert p1 is not p2


def test_create_filter_policy_kwargs():
    """Test create_filter_policy passes kwargs."""
    config = TracingConfig(filter_policy="keep_first_k_by_axes", filter_policy_kwargs={"k": 3})
    policy = config.create_filter_policy()
    assert policy.k == 3


def test_create_filter_policy_isolation():
    """Test policy instances have isolated state."""
    config = TracingConfig(filter_policy="keep_first_k", filter_policy_kwargs={"k": 2})
    p1 = config.create_filter_policy()
    p2 = config.create_filter_policy()

    entry = WorkloadEntry(def_name="test", axes={}, inputs_to_dump={}, order=0)
    p1.submit(entry)

    assert len(p1.entries) == 1
    assert len(p2.entries) == 0


# ============================================================================
# get_inputs_to_dump
# ============================================================================


def test_get_inputs_static_list():
    """Test get_inputs_to_dump with static list."""
    config = TracingConfig(input_dump_policy=["a", "c"])
    result = config.get_inputs_to_dump(
        ["a", "b", "c"], [torch.zeros(1), torch.zeros(2), torch.zeros(3)]
    )
    assert set(result.keys()) == {"a", "c"}


def test_get_inputs_empty_list():
    """Test get_inputs_to_dump with empty list."""
    config = TracingConfig(input_dump_policy=[])
    result = config.get_inputs_to_dump(["a", "b"], [torch.zeros(1), torch.zeros(2)])
    assert result == {}


def test_get_inputs_with_policy():
    """Test get_inputs_to_dump with policy name."""
    config = TracingConfig(input_dump_policy="dump_all")
    result = config.get_inputs_to_dump(["x"], [torch.zeros(5)])
    assert "x" in result


def test_get_inputs_invalid_name():
    """Test get_inputs_to_dump rejects invalid name."""
    config = TracingConfig(input_dump_policy=["nonexistent"])
    with pytest.raises(ValueError, match="invalid input name"):
        config.get_inputs_to_dump(["a"], [torch.zeros(1)])


def test_get_inputs_non_string():
    """Test get_inputs_to_dump rejects non-string from policy."""

    @PolicyRegistry.register_input_dump_policy("_test_non_string", override=True)
    class NonStringPolicy:
        def dump(self, inputs):
            return [123]

    config = TracingConfig(input_dump_policy="_test_non_string")
    with pytest.raises(ValueError, match="invalid input name"):
        config.get_inputs_to_dump(["a"], [torch.zeros(1)])


def test_get_inputs_non_list():
    """Test get_inputs_to_dump rejects non-list from policy."""

    @PolicyRegistry.register_input_dump_policy("_test_non_list", override=True)
    class NonListPolicy:
        def dump(self, inputs):
            return "not_a_list"

    config = TracingConfig(input_dump_policy="_test_non_list")
    with pytest.raises(ValueError, match="must return a list"):
        config.get_inputs_to_dump(["a"], [torch.zeros(1)])


# ============================================================================
# TracingConfigRegistry
# ============================================================================


def test_registry_get():
    """Test TracingConfigRegistry.get() with default and per_definition."""
    default = TracingConfig(filter_policy="keep_all")
    specific = TracingConfig(filter_policy="keep_first")
    registry = TracingConfigRegistry(default=default, per_definition={"def1": specific})

    assert registry.get("def1") == specific
    assert registry.get("def2") == default
    assert registry.get("unknown") == default

    # Test with no default
    registry_no_default = TracingConfigRegistry(per_definition={"def1": specific})
    assert registry_no_default.get("def1") == specific
    assert registry_no_default.get("def2") is None


def test_registry_register():
    """Test TracingConfigRegistry.register() and override."""
    registry = TracingConfigRegistry()
    config1 = TracingConfig(filter_policy="keep_all")
    config2 = TracingConfig(filter_policy="keep_first")

    registry.register("def1", config1)
    assert registry.get("def1") == config1

    # Conflict without override
    with pytest.raises(ValueError, match="already exists"):
        registry.register("def1", config2)

    # Override
    registry.register("def1", config2, override=True)
    assert registry.get("def1") == config2


def test_registry_register_many():
    """Test TracingConfigRegistry.register_many() and conflicts."""
    registry = TracingConfigRegistry()
    config1 = TracingConfig(filter_policy="keep_all")
    config2 = TracingConfig(filter_policy="keep_first")

    registry.register_many({"def1": config1, "def2": config2})
    assert registry.get("def1") == config1
    assert registry.get("def2") == config2

    # Conflict without override
    with pytest.raises(ValueError, match="already exist"):
        registry.register_many({"def1": config1})

    # Override
    new_config = TracingConfig(filter_policy="keep_none")
    registry.register_many({"def1": new_config}, override=True)
    assert registry.get("def1") == new_config


# ============================================================================
# PolicyRegistry
# ============================================================================


def test_policy_registry():
    """Test PolicyRegistry register and get."""

    @PolicyRegistry.register_filter_policy("_test_reg")
    class TestFilterPolicy:
        def submit(self, entry):
            pass

        def drain(self):
            return []

        def reset(self):
            pass

    @PolicyRegistry.register_input_dump_policy("_test_reg")
    class TestDumpPolicy:
        def dump(self, inputs):
            return []

    # Get registered
    assert PolicyRegistry.get_filter_policy("_test_reg") is TestFilterPolicy
    assert PolicyRegistry.get_input_dump_policy("_test_reg") is TestDumpPolicy

    # Get non-existent returns None
    assert PolicyRegistry.get_filter_policy("_nonexistent") is None
    assert PolicyRegistry.get_input_dump_policy("_nonexistent") is None


def test_policy_registry_list():
    """Test PolicyRegistry list returns registered names."""
    filter_list = PolicyRegistry.list_filter_policies()
    dump_list = PolicyRegistry.list_input_dump_policies()

    assert isinstance(filter_list, list)
    assert isinstance(dump_list, list)
    assert len(filter_list) > 0
    assert len(dump_list) > 0


def test_policy_registry_duplicate():
    """Test PolicyRegistry rejects duplicate without override."""

    @PolicyRegistry.register_filter_policy("_test_dup2")
    class TestPolicy:
        def submit(self, entry):
            pass

        def drain(self):
            return []

        def reset(self):
            pass

    with pytest.raises(ValueError, match="already registered"):

        @PolicyRegistry.register_filter_policy("_test_dup2")
        class TestPolicy2:
            pass


def test_policy_registry_override():
    """Test PolicyRegistry override replaces existing."""

    @PolicyRegistry.register_filter_policy("_test_override", override=True)
    class Policy1:
        pass

    @PolicyRegistry.register_filter_policy("_test_override", override=True)
    class Policy2:
        pass

    assert PolicyRegistry.get_filter_policy("_test_override") is Policy2


if __name__ == "__main__":
    pytest.main(sys.argv)
