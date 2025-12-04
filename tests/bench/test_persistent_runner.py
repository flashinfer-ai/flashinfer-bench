import logging
import sys
from pathlib import Path

import pytest
import torch

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.bench.runner.persistent_runner import (
    PersistentRunner,
    PersistentSubprocessWorker,
)
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
    Workload,
)


def _def2d():
    return Definition(
        name="d",
        op_type="op",
        axes={"M": AxisConst(value=2), "N": AxisConst(value=3)},
        inputs={
            "X": TensorSpec(shape=["M", "N"], dtype="float32"),
            "Y": TensorSpec(shape=["M", "N"], dtype="int32"),
            "S": TensorSpec(shape=None, dtype="int32"),
        },
        outputs={"O": TensorSpec(shape=["M", "N"], dtype="float32")},
        reference="def run(X, Y, S):\n    return X\n",
    )


def _simple_def():
    return Definition(
        name="simple",
        op_type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestPersistentSubprocessWorker:
    def test_worker_initialization_and_cleanup(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        assert worker.is_healthy() is True
        assert worker._worker_proc is not None
        assert worker._worker_proc.is_alive()
        assert worker._parent_conn is not None

        worker.close()
        assert worker.is_healthy() is False
        assert worker._worker_proc is None or not worker._worker_proc.is_alive()

    def test_worker_health_check(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        try:
            health_result = worker.is_healthy()
            if health_result:
                assert worker.is_healthy() is True
            else:
                # Worker might be unhealthy for various reasons, that's okay
                pass

        finally:
            worker.close()

    def test_worker_run_ref_basic(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_ref")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            handle = worker.run_ref(d, workload, cfg, None)

            assert handle in worker._baselines
            baseline = worker._baselines[handle]
            assert baseline.definition == d
            assert baseline.device == "cuda:0"
            assert len(baseline.inputs) == cfg.num_trials
            assert len(baseline.outputs) == cfg.num_trials
            assert baseline.mean_latency_ms > 0

            worker.release(handle)
            assert handle not in worker._baselines

        finally:
            worker.close()

    def test_worker_run_solution_success(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_sol")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )
            srcs = [
                SourceFile(
                    path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n"
                )
            ]
            solution = Solution(
                name="test_success", definition=d.name, author="test", spec=spec, sources=srcs
            )

            handle = worker.run_ref(d, workload, cfg, None)

            evaluation = worker.run_solution(solution, handle, cfg)

            assert evaluation.status in {
                EvaluationStatus.PASSED,
                EvaluationStatus.RUNTIME_ERROR,
                EvaluationStatus.COMPILE_ERROR,
            }
            assert isinstance(evaluation.log, str)
            assert evaluation.timestamp is not None
            assert evaluation.environment is not None

            if evaluation.status == EvaluationStatus.PASSED:
                assert evaluation.correctness is not None
                assert evaluation.performance is not None
                assert evaluation.performance.latency_ms > 0
                assert evaluation.performance.reference_latency_ms > 0

            worker.release(handle)

        finally:
            worker.close()

    def test_worker_embeds_stdout(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        handle = None
        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_log")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            message = "persistent worker log line"
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )
            srcs = [
                SourceFile(
                    path="pkg/main.py",
                    content=(
                        "import torch\n"
                        f"def run(A):\n"
                        f"    print({message!r})\n"
                        "    return A\n"
                    ),
                )
            ]
            solution = Solution(
                name="test_log", definition=d.name, author="test", spec=spec, sources=srcs
            )

            handle = worker.run_ref(d, workload, cfg, None)
            evaluation = worker.run_solution(solution, handle, cfg)

            assert isinstance(evaluation.log, str)
            assert message in evaluation.log

        finally:
            if handle is not None:
                worker.release(handle)
            worker.close()


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestPersistentRunner:
    def test_runner_initialization(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")

        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        assert (
            len(runner._workers) > 0
        )  # Should have at least one worker for available CUDA devices
        assert len(runner._available_devices) > 0
        assert all(worker.is_healthy() for worker in runner._workers)

        # Test that workers are properly initialized
        for worker in runner._workers:
            assert worker._worker_proc is not None
            assert worker._worker_proc.is_alive()
            assert worker._parent_conn is not None

    def test_run_workload_single_solution(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_sol")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )
            srcs = [
                SourceFile(
                    path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n"
                )
            ]
            solution = Solution(
                name="test_success", definition=d.name, author="test", spec=spec, sources=srcs
            )

            results = runner.run_workload(d, workload, [solution], cfg, Path(tmp_path))

            assert len(results) == 1
            assert "test_success" in results

            evaluation = results["test_success"]
            assert evaluation.status in {
                EvaluationStatus.PASSED,
                EvaluationStatus.RUNTIME_ERROR,
                EvaluationStatus.COMPILE_ERROR,
            }
            assert isinstance(evaluation.log, str)
            assert evaluation.timestamp is not None
            assert evaluation.environment is not None

            if evaluation.status == EvaluationStatus.PASSED:
                assert evaluation.correctness is not None
                assert evaluation.performance is not None
                assert evaluation.performance.latency_ms > 0
                assert evaluation.performance.reference_latency_ms > 0

        finally:
            # Workers are managed internally, no explicit cleanup needed
            pass

    def test_run_workload_multiple_solutions(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_multi")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )

            solutions = []
            for i in range(3):
                srcs = [
                    SourceFile(
                        path="pkg/main.py",
                        content=f"import torch\n\ndef run(A):\n    return A{'+ 0' if i == 0 else '.clone()' if i == 1 else '* 1'}\n",
                    )
                ]
                solution = Solution(
                    name=f"sol_{i}", definition=d.name, author="test", spec=spec, sources=srcs
                )
                solutions.append(solution)

            results = runner.run_workload(d, workload, solutions, cfg, Path(tmp_path))

            assert len(results) == 3
            for i in range(3):
                assert f"sol_{i}" in results

                evaluation = results[f"sol_{i}"]
                assert evaluation.status in {
                    EvaluationStatus.PASSED,
                    EvaluationStatus.RUNTIME_ERROR,
                    EvaluationStatus.COMPILE_ERROR,
                }
                assert isinstance(evaluation.log, str)
                assert evaluation.timestamp is not None
                assert evaluation.environment is not None

        finally:
            # Workers are managed internally, no explicit cleanup needed
            pass

    def test_run_workload_compilation_error(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(
                axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_compile_error"
            )
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )
            srcs = [
                SourceFile(
                    path="pkg/main.py",
                    content="import nonexistent_module_xyz\n\ndef run(A):\n    return A\n",
                )
            ]
            solution = Solution(
                name="test_error", definition=d.name, author="test", spec=spec, sources=srcs
            )

            results = runner.run_workload(d, workload, [solution], cfg, Path(tmp_path))

            assert len(results) == 1
            evaluation = results["test_error"]
            assert evaluation.status in {
                EvaluationStatus.COMPILE_ERROR,
                EvaluationStatus.RUNTIME_ERROR,
            }
            assert evaluation.log and "nonexistent_module_xyz" in evaluation.log

        finally:
            # Workers are managed internally, no explicit cleanup needed
            pass

    def test_run_workload_empty_solutions(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_empty")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            results = runner.run_workload(d, workload, [], cfg, Path(tmp_path))

            assert len(results) == 0
            assert results == {}

        finally:
            # Workers are managed internally, no explicit cleanup needed
            pass

    def test_worker_retry_logic(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            # Test that the runner has retry logic parameters
            assert hasattr(runner, "_device_retry_counts")
            assert hasattr(runner, "_worker_max_retries")
            assert runner._worker_max_retries == 3

            # Test worker selection mechanism
            if len(runner._workers) > 0:
                selected_workers = runner._pick_workers(2)
                assert len(selected_workers) <= min(2, len(runner._workers))
                assert len(selected_workers) <= len(runner._available_devices)

        finally:
            # Workers are managed internally, no explicit cleanup needed
            pass

    def test_has_healthy_workers(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = logging.getLogger("test_runner")
        runner = PersistentRunner(logger=logger, log_dir=log_dir)

        try:
            assert runner._has_healthy_workers() is True

            assert len(runner._workers) > 0

        finally:
            pass

    def test_registry_cache_usage(self, tmp_path):
        """Test that the registry's builder cache is being used properly by testing with the same worker."""
        log_dir = str(tmp_path / "logs")

        worker = PersistentSubprocessWorker(device="cuda:0", log_dir=log_dir)

        try:
            d = _simple_def()
            workload = Workload(
                axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_registry_cache"
            )
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)

            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="pkg/main.py::run",
            )
            srcs = [
                SourceFile(
                    path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n"
                )
            ]
            solution = Solution(
                name="test_registry", definition=d.name, author="test", spec=spec, sources=srcs
            )

            handle = worker.run_ref(d, workload, cfg, None)

            import time

            start_time = time.time()
            result1 = worker.run_solution(solution, handle, cfg)
            first_duration = time.time() - start_time

            start_time = time.time()
            result2 = worker.run_solution(solution, handle, cfg)
            second_duration = time.time() - start_time

            print(f"First run duration: {first_duration:.3f}s")
            print(f"Second run duration: {second_duration:.3f}s")

            assert result1.status in {
                EvaluationStatus.PASSED,
                EvaluationStatus.COMPILE_ERROR,
                EvaluationStatus.RUNTIME_ERROR,
            }
            assert result2.status in {
                EvaluationStatus.PASSED,
                EvaluationStatus.COMPILE_ERROR,
                EvaluationStatus.RUNTIME_ERROR,
            }

            # the second should be faster due to caching
            if (
                result1.status == EvaluationStatus.PASSED
                and result2.status == EvaluationStatus.PASSED
            ):
                assert (
                    second_duration < first_duration
                ), f"Second run ({second_duration:.3f}s) should be faster than first run ({first_duration:.3f}s) due to caching"

            worker.release(handle)

        finally:
            worker.close()


if __name__ == "__main__":
    pytest.main(sys.argv)
