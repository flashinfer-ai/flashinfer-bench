import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    BuildSpec,
    Definition,
    EvaluationStatus,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    load_jsonl_file,
    save_json_file,
    save_jsonl_file,
)


def test_run_all_empty_traceset(tmp_path: Path):
    """Test run_all with completely empty trace set."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)
    result = benchmark.run_all()

    assert len(benchmark._traces_to_dump) == 0
    assert len(result.definitions) == 0
    assert len(result.solutions) == 0
    assert len(result.workloads) == 0
    assert len(result.traces) == 0


def test_run_all_no_solutions(tmp_path: Path, caplog):
    """Test run_all with definitions but no solutions."""
    # Create definition
    definition = Definition(
        name="test_def",
        op_type="test_op",
        axes={"M": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A + 1\n",
    )

    trace_set = TraceSet(
        root=str(tmp_path),
        definitions={"test_def": definition},
        solutions={},  # No solutions
        workloads={},
        traces={},
    )

    benchmark = Benchmark(trace_set)

    # Capture log messages from the benchmark's logger
    with caplog.at_level(logging.WARNING, logger=benchmark._logger.name):
        result = benchmark.run_all()

    assert "No solutions found for def=test_def, skipping definition" in caplog.text
    assert len(benchmark._traces_to_dump) == 0
    assert len(result.traces) == 0


def test_run_all_no_workloads(tmp_path: Path):
    """Test run_all with definitions and solutions but no workloads."""
    # Create definition
    definition = Definition(
        name="test_def",
        op_type="test_op",
        axes={"M": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A + 1\n",
    )

    # Create solution
    solution = Solution(
        name="test_sol",
        definition="test_def",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="test.py::run"
        ),
        sources=[
            SourceFile(path="test.py", content="import torch\n\ndef run(A):\n    return A + 1\n")
        ],
    )

    trace_set = TraceSet(
        root=str(tmp_path),
        definitions={"test_def": definition},
        solutions={"test_def": [solution]},
        workloads={},  # No workloads
        traces={},
    )

    benchmark = Benchmark(trace_set)
    result = benchmark.run_all()

    assert len(benchmark._traces_to_dump) == 0
    assert len(result.traces) == 0


def test_dump_traces_false(tmp_path: Path):
    """Test run_all with dump_traces=False."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)
    result = benchmark.run_all(dump_traces=False)

    # Should not add traces to dump list
    assert len(benchmark._traces_to_dump) == 0


@patch("flashinfer_bench.bench.benchmark.MultiProcessRunner")
def test_runner_runtime_error(mock_runner_class, tmp_path: Path, caplog):
    """Test handling of RuntimeError from runner."""
    # Setup mock runner to raise RuntimeError
    mock_runner = MagicMock()
    mock_runner.run_workload.side_effect = RuntimeError("Simulated runner error")
    mock_runner_class.return_value = mock_runner

    # Create test data
    definition = Definition(
        name="test_def",
        op_type="test_op",
        axes={"M": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A + 1\n",
    )

    solution = Solution(
        name="test_sol",
        definition="test_def",
        author="tester",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="test.py::run"
        ),
        sources=[
            SourceFile(path="test.py", content="import torch\n\ndef run(A):\n    return A + 1\n")
        ],
    )

    workload = Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="test_uuid")

    workload_trace = Trace(definition="test_def", workload=workload)

    trace_set = TraceSet(
        root=str(tmp_path),
        definitions={"test_def": definition},
        solutions={"test_def": [solution]},
        workloads={"test_def": [workload_trace]},
        traces={},
    )

    benchmark = Benchmark(trace_set)

    # Capture log messages from the benchmark's logger
    with caplog.at_level(logging.ERROR, logger=benchmark._logger.name):
        result = benchmark.run_all()

    assert "Failed to run workload test_uuid: Simulated runner error" in caplog.text
    assert len(benchmark._traces_to_dump) == 0
    assert len(result.traces) == 0


def test_flush_behavior(tmp_path: Path):
    """Test flush behavior and backup mechanism."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)

    # Add some dummy traces to dump
    dummy_trace = Trace(
        definition="dummy",
        workload=Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="test"),
    )

    # Mock the trace_set methods with side_effect to capture arguments
    captured_traces = []

    def capture_add_traces(traces):
        captured_traces.append(traces.copy())  # Make a copy to avoid reference issues

    with patch.object(benchmark._trace_set, "backup_traces") as mock_backup, patch.object(
        benchmark._trace_set, "add_traces", side_effect=capture_add_traces
    ) as mock_add_traces:

        # Add trace and flush
        benchmark._traces_to_dump.append(dummy_trace)

        # First flush should trigger backup
        benchmark.flush()

        mock_backup.assert_called_once()
        mock_add_traces.assert_called_once()
        assert len(captured_traces) == 1
        assert captured_traces[0] == [dummy_trace]
        assert benchmark._is_traces_backed_up is True
        assert len(benchmark._traces_to_dump) == 0

        # Second flush should not trigger backup again
        mock_backup.reset_mock()
        mock_add_traces.reset_mock()
        captured_traces.clear()

        # Add another trace and flush again
        dummy_trace2 = Trace(
            definition="dummy2",
            workload=Workload(axes={"M": 8}, inputs={"A": RandomInput()}, uuid="test2"),
        )
        benchmark._traces_to_dump.append(dummy_trace2)  # Add another trace
        benchmark.flush()

        mock_backup.assert_not_called()  # Should not backup again
        mock_add_traces.assert_called_once()
        assert len(captured_traces) == 1
        assert captured_traces[0] == [dummy_trace2]


def test_multiple_run_and_flush(tmp_path: Path):
    """Test multiple run_all and flush cycles."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)

    with patch.object(benchmark._trace_set, "backup_traces") as mock_backup, patch.object(
        benchmark._trace_set, "add_traces"
    ) as mock_add_traces:

        # First cycle
        benchmark.run_all()
        benchmark.flush()

        # Second cycle
        benchmark.run_all()
        benchmark.flush()

        # Backup should only be called once
        assert mock_backup.call_count == 1
        # add_traces should be called twice
        assert mock_add_traces.call_count == 2


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_benchmark_with_mixed_results(tmp_path: Path):
    """Test benchmark with solutions that have different outcomes."""
    # Build dataset structure
    (tmp_path / "definitions").mkdir(parents=True)
    (tmp_path / "solutions").mkdir(parents=True)
    (tmp_path / "workloads").mkdir(parents=True)

    # Create definition
    definition = Definition(
        name="simple_add",
        op_type="op",
        axes={"N": AxisConst(value=8)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A + 1\n",
    )
    save_json_file(definition, tmp_path / "definitions" / "simple_add.json")

    # Create solutions with different behaviors
    solutions = [
        Solution(
            name="correct_sol",
            definition="simple_add",
            author="tester",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="correct.py::run",
            ),
            sources=[
                SourceFile(
                    path="correct.py", content="import torch\n\ndef run(A):\n    return A + 1\n"
                )
            ],
        ),
        Solution(
            name="wrong_sol",
            definition="simple_add",
            author="tester",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="wrong.py::run",
            ),
            sources=[
                SourceFile(
                    path="wrong.py",
                    content="import torch\n\ndef run(A):\n    return A + 2\n",  # Wrong result
                )
            ],
        ),
    ]

    for sol in solutions:
        save_json_file(sol, tmp_path / "solutions" / f"{sol.name}.json")

    # Create workload
    workload = Workload(axes={"N": 8}, inputs={"A": RandomInput()}, uuid="test_workload")

    workload_trace = Trace(definition="simple_add", workload=workload)
    save_jsonl_file([workload_trace], tmp_path / "workloads" / "op" / "simple_add.jsonl")

    # Load trace set and run benchmark
    trace_set = TraceSet.from_path(str(tmp_path))
    config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    benchmark = Benchmark(trace_set, config)

    result = benchmark.run_all()
    result_traces = result.traces["simple_add"]

    # Verify results
    assert len(result_traces) == 2  # One per solution

    # Should have both correct and incorrect results
    statuses = [t.evaluation.status for t in result_traces]
    assert EvaluationStatus.PASSED in statuses
    assert EvaluationStatus.INCORRECT_NUMERICAL in statuses

    # Flush
    benchmark.flush()

    # Check that the traces were flushed
    assert (tmp_path / "traces" / "op" / "simple_add.jsonl").exists()
    traces_loaded = load_jsonl_file(Trace, tmp_path / "traces" / "op" / "simple_add.jsonl")
    assert traces_loaded == result_traces


if __name__ == "__main__":
    pytest.main(sys.argv)
