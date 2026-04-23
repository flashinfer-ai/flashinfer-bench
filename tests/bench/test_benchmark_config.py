import sys
from types import SimpleNamespace

import pytest
import yaml

from flashinfer_bench.bench import BenchmarkConfig, EvalConfig, ResolvedEvalConfig


def test_benchmark_config_defaults_valid():
    # Top-level fields are Optional and default to None so they act as CLI
    # overrides only when explicitly set. Defaults are supplied by ResolvedEvalConfig.
    cfg = BenchmarkConfig()
    assert cfg.warmup_runs is None
    assert cfg.iterations is None
    assert cfg.num_trials is None
    assert cfg.rtol is None and cfg.atol is None

    resolved = cfg.resolve_eval_config(SimpleNamespace(op_type="generic", name="anon"))
    assert resolved.warmup_runs >= 0
    assert resolved.iterations > 0
    assert resolved.num_trials > 0
    assert resolved.rtol > 0 and resolved.atol > 0


@pytest.mark.parametrize(
    "field, value",
    [("warmup_runs", -1), ("iterations", 0), ("num_trials", 0), ("rtol", 0.0), ("atol", 0.0)],
)
def test_benchmark_config_validation(field, value):
    with pytest.raises(ValueError):
        BenchmarkConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("warmup_runs", -1),
        ("iterations", 0),
        ("num_trials", 0),
        ("rtol", 0.0),
        ("atol", 0.0),
        ("required_matched_ratio", 1.1),
    ],
)
def test_eval_config_validation(field, value):
    with pytest.raises(ValueError):
        EvalConfig(**{field: value})


def test_from_yaml(tmp_path):
    data = {
        "timeout_seconds": 600,
        "rtol": 0.05,
        "op_type_config": {"moe": {"required_matched_ratio": 0.95, "rtol": 0.1}},
        "definition_config": {"my_def": {"atol": 0.5}},
    }
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml.dump(data))

    cfg = BenchmarkConfig.from_yaml(str(yaml_path))
    assert cfg.timeout_seconds == 600
    assert cfg.rtol == 0.05
    assert "moe" in cfg.op_type_config
    assert cfg.op_type_config["moe"].required_matched_ratio == 0.95
    assert cfg.op_type_config["moe"].rtol == 0.1
    assert "my_def" in cfg.definition_config
    assert cfg.definition_config["my_def"].atol == 0.5


def test_resolve_merge_priority():
    # Priority (lowest -> highest): ResolvedEvalConfig defaults ->
    # op_type_config -> definition_config -> top-level / CLI overrides.
    cfg = BenchmarkConfig(
        rtol=0.01,  # CLI-supplied override — must win over YAML layers below.
        atol=0.01,
        op_type_config={"moe": EvalConfig(rtol=0.05, required_matched_ratio=0.95)},
        definition_config={"moe_fp8_def": EvalConfig(rtol=0.1)},
    )

    definition = SimpleNamespace(op_type="moe", name="moe_fp8_def")
    resolved = cfg.resolve_eval_config(definition)

    assert isinstance(resolved, ResolvedEvalConfig)
    # CLI-supplied rtol beats both op_type_config and definition_config.
    assert resolved.rtol == 0.01
    # No CLI value for required_matched_ratio, so op_type_config applies.
    assert resolved.required_matched_ratio == 0.95
    assert resolved.atol == 0.01
    # No value at any layer -> hardcoded ResolvedEvalConfig default.
    assert resolved.warmup_runs == 10


def test_cli_override_beats_yaml_op_type():
    """Regression test: --required-matched-ratio 0.9 must not be shadowed by
    eval_config.yaml's op_type_config.moe.required_matched_ratio = 0.95."""
    cfg = BenchmarkConfig(
        required_matched_ratio=0.9,  # simulates CLI --required-matched-ratio 0.9
        op_type_config={"moe": EvalConfig(required_matched_ratio=0.95)},
    )
    definition = SimpleNamespace(op_type="moe", name="some_moe_def")
    resolved = cfg.resolve_eval_config(definition)
    assert resolved.required_matched_ratio == 0.9


def test_yaml_op_type_applies_without_cli_override():
    """When CLI doesn't supply the field, YAML op_type_config still takes effect."""
    cfg = BenchmarkConfig(op_type_config={"moe": EvalConfig(required_matched_ratio=0.95)})
    definition = SimpleNamespace(op_type="moe", name="some_moe_def")
    resolved = cfg.resolve_eval_config(definition)
    assert resolved.required_matched_ratio == 0.95


def test_definition_config_beats_op_type_config():
    """Among YAML layers, more-specific (definition) beats less-specific (op_type)."""
    cfg = BenchmarkConfig(
        op_type_config={"moe": EvalConfig(rtol=0.05)},
        definition_config={"my_def": EvalConfig(rtol=0.1)},
    )
    definition = SimpleNamespace(op_type="moe", name="my_def")
    resolved = cfg.resolve_eval_config(definition)
    assert resolved.rtol == 0.1


def test_sampling_extra_cli_override_beats_yaml():
    """Top-level sampling_* fields (deprecated CLI overrides) win over YAML layer.extra,
    matching the precedence of the other eval fields."""
    cfg = BenchmarkConfig(
        sampling_validation_trials=1,
        op_type_config={"sampling": EvalConfig(extra={"sampling_validation_trials": 100})},
    )
    definition = SimpleNamespace(op_type="sampling", name="sampling_def")
    resolved = cfg.resolve_eval_config(definition)
    assert resolved.extra["sampling_validation_trials"] == 1


def test_sampling_extra_yaml_applies_without_cli_override():
    """YAML layer.extra still applies when no top-level override is supplied."""
    cfg = BenchmarkConfig(
        op_type_config={"sampling": EvalConfig(extra={"sampling_validation_trials": 50})}
    )
    definition = SimpleNamespace(op_type="sampling", name="sampling_def")
    resolved = cfg.resolve_eval_config(definition)
    assert resolved.extra["sampling_validation_trials"] == 50


def test_sampling_extra_empty_when_nothing_set():
    """When neither CLI nor YAML sets sampling_*, resolved.extra is empty so the
    evaluator falls back to its own class-level defaults (100 / 0.2)."""
    cfg = BenchmarkConfig()
    definition = SimpleNamespace(op_type="sampling", name="sampling_def")
    resolved = cfg.resolve_eval_config(definition)
    assert "sampling_validation_trials" not in resolved.extra
    assert "sampling_tvd_threshold" not in resolved.extra


if __name__ == "__main__":
    pytest.main(sys.argv)
