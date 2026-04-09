from flashinfer_bench.bench import classify_evaluation
from flashinfer_bench.data import Correctness, Environment, Evaluation, EvaluationStatus, Performance


def _make_evaluation(
    status: EvaluationStatus,
    *,
    log: str = "",
    correctness: Correctness | None = None,
    performance: Performance | None = None,
) -> Evaluation:
    return Evaluation(
        status=status,
        log=log,
        environment=Environment(hardware="test", libs={}),
        timestamp="2026-03-10T00:00:00",
        correctness=correctness,
        performance=performance,
    )


def test_classify_compile_signature_mismatch():
    evaluation = _make_evaluation(
        EvaluationStatus.COMPILE_ERROR,
        log="BuildError: Destination-passing style callable: expected 3 parameters, but got 2",
    )

    taxonomy = classify_evaluation(evaluation)

    assert taxonomy.status_family == "compile_error"
    assert taxonomy.secondary_bucket == "compile.signature_mismatch"


def test_classify_runtime_out_of_memory():
    evaluation = _make_evaluation(
        EvaluationStatus.RUNTIME_ERROR,
        log="RuntimeError: CUDA out of memory while launching kernel",
    )

    taxonomy = classify_evaluation(evaluation)

    assert taxonomy.status_family == "runtime_error"
    assert taxonomy.secondary_bucket == "runtime.out_of_memory"


def test_classify_correctness_nonfinite():
    evaluation = _make_evaluation(
        EvaluationStatus.INCORRECT_NUMERICAL,
        correctness=Correctness(
            max_relative_error=float("inf"),
            max_absolute_error=float("nan"),
        ),
    )

    taxonomy = classify_evaluation(evaluation)

    assert taxonomy.status_family == "correctness_error"
    assert taxonomy.secondary_bucket == "correctness.nonfinite"


def test_classify_efficiency_regression():
    evaluation = _make_evaluation(
        EvaluationStatus.PASSED,
        correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
        performance=Performance(
            latency_ms=2.0,
            reference_latency_ms=1.0,
            speedup_factor=0.5,
        ),
    )

    taxonomy = classify_evaluation(evaluation)

    assert taxonomy.status_family == "passed"
    assert taxonomy.secondary_bucket == "passed"
    assert taxonomy.efficiency_bucket == "efficiency.regression"
