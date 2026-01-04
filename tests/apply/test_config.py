"""Tests for ApplyConfig and ApplyConfigRegistry."""

import pytest
from pydantic import ValidationError

from flashinfer_bench.apply import ApplyConfig, ApplyConfigRegistry


def test_apply_config():
    """Test ApplyConfig default and custom values."""
    cfg = ApplyConfig()
    assert cfg.max_atol == 1e-2
    assert cfg.max_rtol == 1e-5
    assert cfg.aot_ratio == 1.0
    assert cfg.on_miss_policy == "fallback_only"

    cfg2 = ApplyConfig(max_atol=1e-3, aot_ratio=0.5, on_miss_policy="use_def_best")
    assert cfg2.max_atol == 1e-3
    assert cfg2.aot_ratio == 0.5
    assert cfg2.on_miss_policy == "use_def_best"


def test_apply_config_validation():
    """Test ApplyConfig field validation."""
    with pytest.raises(ValidationError):
        ApplyConfig(max_atol=-1.0)
    with pytest.raises(ValidationError):
        ApplyConfig(max_rtol=0)
    with pytest.raises(ValidationError):
        ApplyConfig(aot_ratio=1.5)
    with pytest.raises(ValidationError):
        ApplyConfig(on_miss_policy="invalid")


def test_registry_get():
    """Test ApplyConfigRegistry.get() with default fallback."""
    registry = ApplyConfigRegistry()
    assert registry.get("unknown") == ApplyConfig()

    custom_default = ApplyConfig(max_atol=1e-3)
    registry2 = ApplyConfigRegistry(default=custom_default)
    assert registry2.get("any").max_atol == 1e-3


def test_registry_register():
    """Test register() and register_many()."""
    cfg1 = ApplyConfig(max_atol=1e-3)
    cfg2 = ApplyConfig(max_atol=2e-3)
    registry = ApplyConfigRegistry().register("def1", cfg1).register("def2", cfg2)
    assert registry.get("def1") == cfg1
    assert registry.get("def2") == cfg2

    registry.register_many({"a": ApplyConfig(max_atol=1e-4), "b": ApplyConfig(aot_ratio=0.0)})
    assert registry.get("a").max_atol == 1e-4
    assert registry.get("b").aot_ratio == 0.0


def test_registry_override():
    """Test register() raises on duplicate and allows override."""
    registry = ApplyConfigRegistry().register("def", ApplyConfig(max_atol=1e-3))

    with pytest.raises(ValueError, match="already exists"):
        registry.register("def", ApplyConfig(max_atol=2e-3))

    registry.register("def", ApplyConfig(max_atol=5e-3), override=True)
    assert registry.get("def").max_atol == 5e-3
