import pytest

from flashinfer_bench.bench.config import BenchmarkConfig


def test_benchmark_config_defaults_valid():
    cfg = BenchmarkConfig()
    assert cfg.warmup_runs >= 0
    assert cfg.iterations > 0
    assert cfg.num_trials > 0
    assert cfg.rtol > 0 and cfg.atol > 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("warmup_runs", -1),
        ("iterations", 0),
        ("num_trials", 0),
        ("rtol", 0.0),
        ("atol", 0.0),
    ],
)
def test_benchmark_config_validation(field, value):
    kwargs = {}
    kwargs[field] = value
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)
