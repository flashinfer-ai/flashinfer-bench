import sys
from pathlib import Path

import pytest

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
    save_json_file,
    save_jsonl_file,
)


def test_trace_set_from_path_and_queries(tmp_path: Path):
    # Create directory structure
    (tmp_path / "definitions").mkdir()
    (tmp_path / "solutions").mkdir()
    (tmp_path / "traces").mkdir()

    # Definition
    ref = "def run(a):\n    return a\n"
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar(), "N": AxisConst(value=2)},
        inputs={"A": TensorSpec(shape=["M", "N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference=ref,
    )
    save_json_file(definition, tmp_path / "definitions" / "d1.json")

    # Solutions
    solution1 = Solution(
        name="s1",
        definition="d1",
        author="a",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    solution2 = Solution(
        name="s2",
        definition="d1",
        author="b",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )
    save_json_file(solution1, tmp_path / "solutions" / "s1.json")
    save_json_file(solution2, tmp_path / "solutions" / "s2.json")

    # Traces JSONL
    trace_pass = Trace(
        definition="d1",
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="tw1"),
        solution="s1",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="log",
            environment=Environment(hardware="cpu"),
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )
    trace_fail = Trace(
        definition="d1",
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid="tw2"),
        solution="s2",
        evaluation=Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            log="log",
            environment=Environment(hardware="cpu"),
            timestamp="t",
        ),
    )
    trace_workload = Trace(
        definition="d1", workload=Workload(axes={"M": 3}, inputs={"A": RandomInput()}, uuid="tw3")
    )
    save_jsonl_file([trace_workload], tmp_path / "workloads" / "d1.jsonl")
    save_jsonl_file([trace_pass, trace_fail], tmp_path / "traces" / "d1.jsonl")

    # Load
    trace_set = TraceSet.from_path(str(tmp_path))

    # Queries
    assert trace_set.definitions.get("d1").name == "d1"
    assert trace_set.get_solution("s1").name == "s1"
    assert len(trace_set.workloads.get("d1", [])) == 1
    assert len(trace_set.traces.get("d1", [])) == 2  # pass + fail

    # Best trace should pick the passed one with higher speedup
    best = trace_set.get_best_trace("d1", axes={"M": 2})
    assert best is not None and best.solution == "s1"

    # Summary
    summary = trace_set.summary(baseline_author="a")
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1


def test_trace_set_ranking_uses_avg_speedup():
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )

    def make_solution(name: str, author: str) -> Solution:
        return Solution(
            name=name,
            definition="d1",
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(solution: str, uuid: str, latency_ms: float) -> Trace:
        return Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                log="log",
                environment=Environment(hardware="cpu"),
                timestamp="t",
                correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                performance=Performance(
                    latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                ),
            ),
        )

    trace_set = TraceSet(
        definitions={"d1": definition},
        solutions={
            "d1": [
                make_solution("baseline", "flashinfer"),
                make_solution("alice", "alice"),
                make_solution("bob", "bob"),
            ]
        },
        traces={
            "d1": [
                make_trace("baseline", "w1", 10.0),
                make_trace("alice", "w1", 5.0),
                make_trace("bob", "w1", 20.0),
                make_trace("baseline", "w2", 8.0),
                make_trace("alice", "w2", 8.0),
                make_trace("bob", "w2", 4.0),
            ]
        },
    )

    rankings = trace_set.rank_authors()

    assert [author for author, _ in rankings] == ["alice", "bob"]
    assert rankings[0][1].avg_speedup == pytest.approx(1.5)
    assert rankings[0][1].definitions == 1
    assert rankings[0][1].success_rate == pytest.approx(1.0)
    assert rankings[0][1].win_rate == pytest.approx(0.5)
    assert rankings[0][1].workloads == 2
    assert rankings[1][1].avg_speedup == pytest.approx(1.25)
    assert rankings[1][1].definitions == 1
    assert rankings[1][1].success_rate == pytest.approx(1.0)
    assert rankings[1][1].workloads == 2

    alice_score = trace_set.get_author_score("alice")
    assert alice_score is not None
    assert alice_score.avg_speedup == pytest.approx(1.5)
    assert alice_score.win_rate == pytest.approx(0.5)

    bob_score = trace_set.get_solution_score("bob")
    assert bob_score is not None
    assert bob_score.avg_speedup == pytest.approx(1.25)
    assert bob_score.success_rate == pytest.approx(1.0)
    assert bob_score.win_rate == pytest.approx(0.5)


def test_trace_set_ranking_counts_missing_pass_as_zero():
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )

    def make_solution(name: str, author: str) -> Solution:
        return Solution(
            name=name,
            definition="d1",
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(
        solution: str, uuid: str, status: EvaluationStatus, latency_ms: float | None
    ) -> Trace:
        if status == EvaluationStatus.PASSED:
            return Trace(
                definition="d1",
                workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
                solution=solution,
                evaluation=Evaluation(
                    status=status,
                    log="log",
                    environment=Environment(hardware="cpu"),
                    timestamp="t",
                    correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                    performance=Performance(
                        latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                    ),
                ),
            )
        return Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=status, log="log", environment=Environment(hardware="cpu"), timestamp="t"
            ),
        )

    trace_set = TraceSet(
        definitions={"d1": definition},
        solutions={
            "d1": [
                make_solution("baseline", "flashinfer"),
                make_solution("alice", "alice"),
                make_solution("bob", "bob"),
            ]
        },
        traces={
            "d1": [
                make_trace("baseline", "w1", EvaluationStatus.PASSED, 10.0),
                make_trace("alice", "w1", EvaluationStatus.PASSED, 5.0),
                make_trace("bob", "w1", EvaluationStatus.RUNTIME_ERROR, None),
                make_trace("baseline", "w2", EvaluationStatus.PASSED, 8.0),
                make_trace("alice", "w2", EvaluationStatus.RUNTIME_ERROR, None),
                make_trace("bob", "w2", EvaluationStatus.PASSED, 4.0),
            ]
        },
    )

    rankings = trace_set.rank_authors()

    assert [author for author, _ in rankings] == ["alice", "bob"]
    assert rankings[0][1].avg_speedup == pytest.approx(0.0)
    assert rankings[0][1].definitions == 1
    assert rankings[0][1].success_rate == pytest.approx(0.0)
    assert rankings[0][1].win_rate == pytest.approx(0.0)
    assert rankings[0][1].workloads == 2
    assert rankings[1][1].avg_speedup == pytest.approx(0.0)
    assert rankings[1][1].definitions == 1
    assert rankings[1][1].success_rate == pytest.approx(0.0)
    assert rankings[1][1].win_rate == pytest.approx(0.0)
    assert rankings[1][1].workloads == 2


def test_trace_set_ranking_uses_best_solution_per_author_and_definition():
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )

    def make_solution(name: str, author: str) -> Solution:
        return Solution(
            name=name,
            definition="d1",
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(
        solution: str, uuid: str, status: EvaluationStatus, latency_ms: float | None
    ) -> Trace:
        if status == EvaluationStatus.PASSED:
            return Trace(
                definition="d1",
                workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
                solution=solution,
                evaluation=Evaluation(
                    status=status,
                    log="log",
                    environment=Environment(hardware="cpu"),
                    timestamp="t",
                    correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                    performance=Performance(
                        latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                    ),
                ),
            )
        return Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=status, log="log", environment=Environment(hardware="cpu"), timestamp="t"
            ),
        )

    trace_set = TraceSet(
        definitions={"d1": definition},
        solutions={
            "d1": [
                make_solution("baseline", "flashinfer"),
                make_solution("alice_bad", "alice"),
                make_solution("alice_good", "alice"),
                make_solution("bob", "bob"),
            ]
        },
        traces={
            "d1": [
                make_trace("baseline", "w1", EvaluationStatus.PASSED, 10.0),
                make_trace("alice_bad", "w1", EvaluationStatus.PASSED, 5.0),
                make_trace("alice_good", "w1", EvaluationStatus.PASSED, 6.0),
                make_trace("bob", "w1", EvaluationStatus.PASSED, 20.0),
                make_trace("baseline", "w2", EvaluationStatus.PASSED, 8.0),
                make_trace("alice_bad", "w2", EvaluationStatus.RUNTIME_ERROR, None),
                make_trace("alice_good", "w2", EvaluationStatus.PASSED, 4.0),
                make_trace("bob", "w2", EvaluationStatus.PASSED, 8.0),
            ]
        },
    )

    rankings = trace_set.rank_authors()

    assert [author for author, _ in rankings] == ["alice", "bob"]
    assert rankings[0][1].avg_speedup == pytest.approx((10.0 / 6.0 + 8.0 / 4.0) / 2)
    assert rankings[0][1].definitions == 1
    assert rankings[0][1].success_rate == pytest.approx(1.0)
    assert rankings[0][1].win_rate == pytest.approx(1.0)
    assert rankings[0][1].workloads == 2
    assert rankings[1][1].avg_speedup == pytest.approx((10.0 / 20.0 + 8.0 / 8.0) / 2)
    assert rankings[1][1].definitions == 1
    assert rankings[1][1].success_rate == pytest.approx(1.0)
    assert rankings[1][1].win_rate == pytest.approx(0.0)
    assert rankings[1][1].workloads == 2


def test_trace_set_ranking_raises_when_baseline_missing():
    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )

    def make_solution(name: str, author: str) -> Solution:
        return Solution(
            name=name,
            definition="d1",
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(solution: str, uuid: str, latency_ms: float) -> Trace:
        return Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                log="log",
                environment=Environment(hardware="cpu"),
                timestamp="t",
                correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                performance=Performance(
                    latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                ),
            ),
        )

    trace_set = TraceSet(
        definitions={"d1": definition},
        solutions={"d1": [make_solution("alice", "alice")]},
        traces={"d1": [make_trace("alice", "w1", 5.0)]},
    )

    with pytest.raises(ValueError, match="No baseline solution from author .* definition 'd1'"):
        trace_set.rank_authors()


def test_multi_definition_scope_and_best_solutions():
    def make_def(name: str, op_type: str) -> Definition:
        return Definition(
            name=name,
            op_type=op_type,
            axes={"M": AxisVar()},
            inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
            outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
            reference="def run(a):\n    return a\n",
        )

    def make_solution(name: str, author: str, definition: str) -> Solution:
        return Solution(
            name=name,
            definition=definition,
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(solution: str, definition: str, uuid: str, latency_ms: float) -> Trace:
        return Trace(
            definition=definition,
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                log="log",
                environment=Environment(hardware="cpu"),
                timestamp="t",
                correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                performance=Performance(
                    latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                ),
            ),
        )

    trace_set = TraceSet(
        definitions={"d1": make_def("d1", "attn"), "d2": make_def("d2", "norm")},
        solutions={
            "d1": [
                make_solution("bl_d1", "flashinfer", "d1"),
                make_solution("alice_d1_slow", "alice", "d1"),
                make_solution("alice_d1_fast", "alice", "d1"),
            ],
            "d2": [
                make_solution("bl_d2", "flashinfer", "d2"),
                make_solution("alice_d2", "alice", "d2"),
            ],
        },
        traces={
            "d1": [
                make_trace("bl_d1", "d1", "w1", 10.0),
                make_trace("alice_d1_slow", "d1", "w1", 20.0),
                make_trace("alice_d1_fast", "d1", "w1", 5.0),
            ],
            "d2": [make_trace("bl_d2", "d2", "w2", 8.0), make_trace("alice_d2", "d2", "w2", 4.0)],
        },
    )

    # d1: best solution = alice_d1_fast, speedup = 10/5 = 2.0
    # d2: speedup = 8/4 = 2.0
    # author score = mean(2.0, 2.0) = 2.0
    score = trace_set.get_author_score("alice")
    assert score is not None
    assert score.avg_speedup == pytest.approx(2.0)
    assert score.definitions == 2
    assert score.best_solutions == {"d1": "alice_d1_fast", "d2": "alice_d2"}

    # Scope by op_type: only d1 (attn)
    score_attn = trace_set.get_author_score("alice", op_type="attn")
    assert score_attn is not None
    assert score_attn.definitions == 1
    assert score_attn.best_solutions == {"d1": "alice_d1_fast"}

    # Scope by definition_name
    score_d2 = trace_set.get_author_score("alice", definition_name="d2")
    assert score_d2 is not None
    assert score_d2.definitions == 1
    assert score_d2.best_solutions == {"d2": "alice_d2"}

    # rank_authors returns tuples and respects scope
    all_rankings = trace_set.rank_authors()
    assert len(all_rankings) == 1
    assert all_rankings[0][0] == "alice"
    assert all_rankings[0][1].definitions == 2

    attn_rankings = trace_set.rank_authors(op_type="attn")
    assert len(attn_rankings) == 1
    assert attn_rankings[0][1].definitions == 1


def test_summary_includes_rankings():
    def make_solution(name: str, author: str) -> Solution:
        return Solution(
            name=name,
            definition="d1",
            author=author,
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cpu"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
        )

    def make_trace(solution: str, uuid: str, latency_ms: float) -> Trace:
        return Trace(
            definition="d1",
            workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
            solution=solution,
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                log="log",
                environment=Environment(hardware="cpu"),
                timestamp="t",
                correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
                performance=Performance(
                    latency_ms=latency_ms, reference_latency_ms=latency_ms, speedup_factor=1.0
                ),
            ),
        )

    definition = Definition(
        name="d1",
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )

    trace_set = TraceSet(
        definitions={"d1": definition},
        solutions={
            "d1": [make_solution("baseline", "flashinfer"), make_solution("alice", "alice")]
        },
        traces={"d1": [make_trace("baseline", "w1", 10.0), make_trace("alice", "w1", 5.0)]},
    )

    s = trace_set.summary()
    assert s.total == 2
    assert s.passed == 2
    assert s.failed == 0
    assert len(s.rankings) == 1
    assert s.rankings[0][0] == "alice"
    assert s.rankings[0][1].avg_speedup == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main(sys.argv)
