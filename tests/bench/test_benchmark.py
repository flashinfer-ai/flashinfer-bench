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


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_run_all_empty_traceset(tmp_path: Path):
    """Test run_all with completely empty trace set."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)
    result = benchmark.run_all()

    assert len(result.definitions) == 0
    assert len(result.solutions) == 0
    assert len(result.workloads) == 0
    assert len(result.traces) == 0


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
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
    from flashinfer_bench.bench.benchmark import logger

    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = benchmark.run_all()

    assert "No solutions found for def=test_def, skipping definition" in caplog.text
    assert len(result.traces) == 0


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
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

    assert len(result.traces) == 0


@pytest.mark.skipif(
    __import__("torch").cuda.device_count() == 0, reason="CUDA devices not available"
)
def test_dump_traces_false(tmp_path: Path):
    """Test run_all with dump_traces=False."""
    trace_set = TraceSet(root=str(tmp_path), definitions={}, solutions={}, workloads={}, traces={})

    benchmark = Benchmark(trace_set)
    result = benchmark.run_all(dump_traces=False)

    # Should not add traces to the trace set
    assert len(result.traces) == 0


@patch("flashinfer_bench.bench.benchmark.IsolatedRunner")
def test_isolated_runner_runtime_error(mock_runner_class, tmp_path: Path, caplog):
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

    benchmark = Benchmark(trace_set, BenchmarkConfig(use_isolated_runner=True))

    # Capture log messages from the benchmark's logger
    from flashinfer_bench.bench.benchmark import logger

    with caplog.at_level(logging.ERROR, logger=logger.name):
        result = benchmark.run_all()

    assert "Failed to run workload test_uuid: Simulated runner error" in caplog.text
    assert len(result.traces) == 0


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
                target_hardware=["cuda"],
                entry_point="correct.py::run",
                destination_passing_style=False,
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
                target_hardware=["cuda"],
                entry_point="wrong.py::run",
                destination_passing_style=False,
            ),
            sources=[
                SourceFile(
                    path="wrong.py",
                    content="import torch\n\ndef run(A):\n    return A + 2\n",  # Wrong result
                )
            ],
        ),
    ]

    for solution in solutions:
        save_json_file(solution, tmp_path / "solutions" / f"{solution.name}.json")

    # Create workload
    workload = Workload(axes={"N": 8}, inputs={"A": RandomInput()}, uuid="test_workload")

    workload_trace = Trace(definition="simple_add", workload=workload)
    save_jsonl_file([workload_trace], tmp_path / "workloads" / "op" / "simple_add.jsonl")

    # Load trace set and run benchmark
    trace_set = TraceSet.from_path(str(tmp_path))
    config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    benchmark = Benchmark(trace_set, config)

    result = benchmark.run_all(dump_traces=True)
    result_traces = result.traces["simple_add"]

    # Verify results
    assert len(result_traces) == 2  # One per solution

    # Should have both correct and incorrect results
    statuses = [t.evaluation.status for t in result_traces]
    assert EvaluationStatus.PASSED in statuses
    assert EvaluationStatus.INCORRECT_NUMERICAL in statuses

    # Check that the traces were stored to the disk
    assert (tmp_path / "traces" / "op" / "simple_add.jsonl").exists()
    traces_loaded = load_jsonl_file(Trace, tmp_path / "traces" / "op" / "simple_add.jsonl")
    assert traces_loaded == result_traces


if __name__ == "__main__":
    pytest.main(sys.argv)
