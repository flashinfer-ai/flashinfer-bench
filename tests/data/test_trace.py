import sys

import pytest

from flashinfer_bench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    SafetensorsInput,
    Trace,
    Workload,
)


def test_workload_validation():
    # Valid
    Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="w1")
    # Invalid axis value
    with pytest.raises(ValueError):
        Workload(axes={"M": 0}, inputs={"A": RandomInput()}, uuid="w_bad")
    # Invalid input type
    with pytest.raises(ValueError):
        Workload(axes={"M": 1}, inputs={"A": object()}, uuid="w_bad2")


def test_correctness_performance_environment_validation():
    Correctness(max_relative_error=0.0, max_absolute_error=0.0)
    with pytest.raises(ValueError):
        Correctness(max_relative_error=-1.0)
    Performance(latency_ms=0.0, reference_latency_ms=0.0, speedup_factor=0.0)
    with pytest.raises(ValueError):
        Performance(latency_ms=-1.0)
    Environment(hardware="cuda:0")
    with pytest.raises(ValueError):
        Environment(hardware="")


def test_evaluation_status_requirements():
    env = Environment(hardware="cuda")
    # PASSED requires correctness and performance
    with pytest.raises(ValueError):
        Evaluation(status=EvaluationStatus.PASSED, log_file="a", environment=env, timestamp="t")
    # INCORRECT_NUMERICAL requires correctness only
    Evaluation(
        status=EvaluationStatus.INCORRECT_NUMERICAL,
        log_file="a",
        environment=env,
        timestamp="t",
        correctness=Correctness(),
    )
    with pytest.raises(ValueError):
        Evaluation(
            status=EvaluationStatus.INCORRECT_NUMERICAL,
            log_file="a",
            environment=env,
            timestamp="t",
            performance=Performance(),
        )
    # Other statuses must not have correctness/performance
    for st in [
        EvaluationStatus.INCORRECT_SHAPE,
        EvaluationStatus.INCORRECT_DTYPE,
        EvaluationStatus.RUNTIME_ERROR,
        EvaluationStatus.COMPILE_ERROR,
    ]:
        Evaluation(status=st, log_file="a", environment=env, timestamp="t")
        with pytest.raises(ValueError):
            Evaluation(
                status=st, log_file="a", environment=env, timestamp="t", correctness=Correctness()
            )
        with pytest.raises(ValueError):
            Evaluation(
                status=st, log_file="a", environment=env, timestamp="t", performance=Performance()
            )


def test_trace_workload_and_regular():
    wl = Workload(
        axes={"M": 8},
        inputs={"A": RandomInput(), "B": SafetensorsInput(path="p", tensor_key="k")},
        uuid="w2",
    )
    # Workload-only
    t_wl = Trace(definition="def1", workload=wl)
    assert t_wl.is_workload() is True
    # Regular successful trace
    eval_ok = Evaluation(
        status=EvaluationStatus.PASSED,
        log_file="log",
        environment=Environment(hardware="cuda"),
        timestamp="t",
        correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
        performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
    )
    t_ok = Trace(definition="def1", workload=wl, solution="sol1", evaluation=eval_ok)
    assert t_ok.is_workload() is False
    assert t_ok.is_successful() is True
    # Regular missing fields
    with pytest.raises(ValueError):
        Trace(definition="def1", workload=wl, solution="sol1")  # type: ignore[call-arg]


if __name__ == "__main__":
    pytest.main(sys.argv)
