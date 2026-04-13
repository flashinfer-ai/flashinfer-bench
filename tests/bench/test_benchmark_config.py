import sys
from types import SimpleNamespace

import pytest
import yaml

from flashinfer_bench.bench import BenchmarkConfig, EvalConfig, ResolvedEvalConfig


def test_benchmark_config_defaults_valid():
    cfg = BenchmarkConfig()
    assert cfg.warmup_runs >= 0
    assert cfg.iterations > 0
    assert cfg.num_trials > 0
    assert cfg.rtol > 0 and cfg.atol > 0


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
    cfg = BenchmarkConfig(
        rtol=0.01,
        atol=0.01,
        op_type_config={"moe": EvalConfig(rtol=0.05, required_matched_ratio=0.95)},
        definition_config={"moe_fp8_def": EvalConfig(rtol=0.1)},
    )

    definition = SimpleNamespace(op_type="moe", name="moe_fp8_def")
    resolved = cfg.resolve_eval_config(definition)

    assert isinstance(resolved, ResolvedEvalConfig)
    assert resolved.rtol == 0.1
    assert resolved.required_matched_ratio == 0.95
    assert resolved.atol == 0.01
    assert resolved.warmup_runs == 10


if __name__ == "__main__":
    pytest.main(sys.argv)
