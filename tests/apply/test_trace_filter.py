from __future__ import annotations

from flashinfer_bench.apply.trace_filter import (
    build_filtered_trace_set,
    collect_solution_names_by_pool,
    count_eligible_traces_by_solution,
    hardware_matches,
)
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)


def _make_definition() -> Definition:
    return Definition(
        name="add",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="float32"),
        },
        outputs={"Z": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y):\n    return X\n",
    )


def _make_solution(name: str, *, language=SupportedLanguages.PYTHON) -> Solution:
    return Solution(
        name=name,
        definition="add",
        author="tester",
        spec=BuildSpec(
            language=language,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return X\n")],
    )


def _make_trace(solution: str, hardware: str, speedup: float, uuid: str) -> Trace:
    return Trace(
        definition="add",
        workload=Workload(
            axes={"M": 2},
            inputs={"X": RandomInput(), "Y": RandomInput()},
            uuid=uuid,
        ),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="",
            environment=Environment(hardware=hardware),
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(
                latency_ms=1.0 / max(speedup, 1e-6),
                reference_latency_ms=1.0,
                speedup_factor=speedup,
            ),
        ),
    )


def _make_trace_set() -> TraceSet:
    definition = _make_definition()
    sol_a800 = _make_solution("sol_a800")
    sol_b200 = _make_solution("sol_b200")
    traces = [
        _make_trace("sol_a800", "NVIDIA A800-SXM4-80GB", 2.0, "w_a800"),
        _make_trace("sol_b200", "NVIDIA B200", 4.0, "w_b200"),
    ]
    return TraceSet(
        root=None,
        definitions={"add": definition},
        solutions={"add": [sol_a800, sol_b200]},
        traces={"add": traces},
    )


def test_hardware_matches_substring_case_insensitive():
    assert hardware_matches("NVIDIA A800-SXM4-80GB", ["a800"])
    assert hardware_matches("NVIDIA B200", ["NVIDIA", "A800"])
    assert not hardware_matches("NVIDIA B200", ["a800"])


def test_build_filtered_trace_set_filters_by_trace_hardware():
    trace_set = _make_trace_set()

    filtered = build_filtered_trace_set(
        trace_set,
        definition_names=["add"],
        trace_hardware_filters=["a800"],
    )

    assert list(filtered.definitions.keys()) == ["add"]
    assert [solution.name for solution in filtered.solutions["add"]] == ["sol_a800"]
    assert [trace.solution for trace in filtered.traces["add"]] == ["sol_a800"]


def test_build_filtered_trace_set_filters_by_solution_language():
    definition = _make_definition()
    sol_cuda = _make_solution("sol_cuda", language=SupportedLanguages.CUDA)
    sol_triton = _make_solution("sol_triton", language=SupportedLanguages.TRITON)
    trace_set = TraceSet(
        root=None,
        definitions={"add": definition},
        solutions={"add": [sol_cuda, sol_triton]},
        traces={
            "add": [
                _make_trace("sol_cuda", "NVIDIA A800-SXM4-80GB", 2.0, "w_cuda"),
                _make_trace("sol_triton", "NVIDIA A800-SXM4-80GB", 3.0, "w_triton"),
            ]
        },
    )

    filtered = build_filtered_trace_set(
        trace_set,
        definition_names=["add"],
        solution_language_filters=["cuda"],
    )

    assert [solution.name for solution in filtered.solutions["add"]] == ["sol_cuda"]
    assert [trace.solution for trace in filtered.traces["add"]] == ["sol_cuda"]


def test_collect_solution_names_by_pool_honors_solution_language_filters():
    definition = _make_definition()
    baseline = Solution(
        name="baseline_python",
        definition="add",
        author="flashinfer",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cpu"],
            entry_point="main.py::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="main.py", content="def run(X, Y):\n    return X\n")],
    )
    generated_cuda = _make_solution("generated_cuda", language=SupportedLanguages.CUDA)
    generated_triton = _make_solution("generated_triton", language=SupportedLanguages.TRITON)
    trace_set = TraceSet(
        root=None,
        definitions={"add": definition},
        solutions={"add": [baseline, generated_cuda, generated_triton]},
        traces={},
    )

    names = collect_solution_names_by_pool(
        trace_set,
        pool="generated_only",
        solution_language_filters=["cuda"],
    )

    assert names == ["generated_cuda"]


def test_count_eligible_traces_by_solution_honors_hardware_filters():
    trace_set = _make_trace_set()

    counts = count_eligible_traces_by_solution(
        trace_set,
        solution_names=["sol_a800", "sol_b200"],
        max_atol=1e-2,
        max_rtol=1e-2,
        trace_hardware_filters=["A800"],
    )

    assert counts == {"sol_a800": 1, "sol_b200": 0}
