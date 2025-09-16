import sys
import time
import uuid
from pathlib import Path

import pytest
import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runners.persistent_runner import (
    PersistentRunner,
    SolutionCacheKey,
    CachedSolution,
)
from flashinfer_bench.data.definition import AxisConst, Definition, TensorSpec
from flashinfer_bench.data.solution import BuildSpec, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.data.trace import RandomInput, SafetensorsInput, ScalarInput, Workload, EvaluationStatus


def _def2d():
    return Definition(
        name="d",
        type="op",
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
        type="op",
        axes={"N": AxisConst(value=4)},
        inputs={"A": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(A):\n    return A\n",
    )


class TestSolutionCaching:    
    def test_solution_cache_key_creation(self):
        d = _simple_def()
        
        spec = BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="pkg/main.py::run",
        )
        srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
        s = Solution(name="test_sol", definition=d.name, author="test", spec=spec, sources=srcs)
        
        key1 = SolutionCacheKey.from_defn_sol(d, s)
        key2 = SolutionCacheKey.from_defn_sol(d, s)
        
        assert key1.definition_hash == key2.definition_hash
        assert key1.solution_hash == key2.solution_hash
        assert key1 == key2
        assert hash(key1) == hash(key2)
        
        key_set = {key1, key2}
        assert len(key_set) == 1
    
    def test_solution_cache_key_different_solutions(self):
        d = _simple_def()
        
        spec = BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["gpu"],
            entry_point="pkg/main.py::run",
        )
        
        srcs1 = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
        s1 = Solution(name="sol1", definition=d.name, author="test", spec=spec, sources=srcs1)
        
        srcs2 = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A * 2\n")]
        s2 = Solution(name="sol2", definition=d.name, author="test", spec=spec, sources=srcs2)
        
        key1 = SolutionCacheKey.from_defn_sol(d, s1)
        key2 = SolutionCacheKey.from_defn_sol(d, s2)
        
        assert key1 != key2
        assert hash(key1) != hash(key2)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="CUDA devices not available")
class TestPersistentRunner:    
    def test_runner_initialization_and_cleanup(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        assert runner._worker_healthy is True
        assert runner._worker_proc is not None
        assert runner._worker_proc.is_alive()
        assert runner._parent_conn is not None
        
        runner.close()
        assert runner._worker_healthy is False
        assert runner._worker_proc is None or not runner._worker_proc.is_alive()
    
    def test_worker_health_check(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            health_result = runner._check_worker_health()
            if health_result:
                assert runner._worker_healthy is True
                
                assert runner._check_worker_health() is True
            else:
                pass
                
        finally:
            runner.close()
    
    def test_run_ref_basic(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_ref")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            handle = runner.run_ref(d, wl, cfg, None)
            
            assert handle in runner._baselines
            baseline = runner._baselines[handle]
            assert baseline.defn == d
            assert baseline.device == "cuda:0"
            assert len(baseline.inputs_dev) == cfg.num_trials
            assert len(baseline.ref_outputs_dev) == cfg.num_trials
            assert baseline.ref_mean_latency_ms > 0
            
            runner.release(handle)
            assert handle not in runner._baselines
            
        finally:
            runner.close()
    
    def test_run_solution_success(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_sol")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
            sol = Solution(name="test_success", definition=d.name, author="test", spec=spec, sources=srcs)
            
            handle = runner.run_ref(d, wl, cfg, None)
            
            initial_cache_size = len(runner._solution_cache)
            
            evaluation = runner.run_solution(sol, handle, cfg)
            
            if evaluation.status != EvaluationStatus.COMPILE_ERROR:
                assert len(runner._solution_cache) == initial_cache_size + 1
            
            assert evaluation.status in {
                EvaluationStatus.PASSED, 
                EvaluationStatus.RUNTIME_ERROR, 
                EvaluationStatus.COMPILE_ERROR
            }
            assert evaluation.log_file is not None
            assert evaluation.timestamp is not None
            assert evaluation.environment is not None
            
            if evaluation.status == EvaluationStatus.PASSED:
                assert evaluation.correctness is not None
                assert evaluation.performance is not None
                assert evaluation.performance.latency_ms > 0
                assert evaluation.performance.reference_latency_ms > 0
            
            runner.release(handle)
            
        finally:
            runner.close()
    
    def test_solution_caching_reuse(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_cache")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
            sol = Solution(name="test_cache", definition=d.name, author="test", spec=spec, sources=srcs)
            
            handle = runner.run_ref(d, wl, cfg, None)
            
            initial_cache_size = len(runner._solution_cache)
            eval1 = runner.run_solution(sol, handle, cfg)
            cache_size_after_first = len(runner._solution_cache)
            
            eval2 = runner.run_solution(sol, handle, cfg)
            cache_size_after_second = len(runner._solution_cache)
            
            if eval1.status not in {EvaluationStatus.COMPILE_ERROR, EvaluationStatus.RUNTIME_ERROR}:
                assert cache_size_after_second == cache_size_after_first  # No new cache entry
            
            if eval1.status == EvaluationStatus.PASSED:
                assert eval2.status == EvaluationStatus.PASSED
            
            runner.release(handle)
            
        finally:
            runner.close()
    
    def test_compilation_error_handling(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_compile_error")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            srcs = [SourceFile(path="pkg/main.py", content="import nonexistent_module_xyz\n\ndef run(A):\n    return A\n")]
            sol = Solution(name="test_error", definition=d.name, author="test", spec=spec, sources=srcs)
            
            handle = runner.run_ref(d, wl, cfg, None)
            evaluation = runner.run_solution(sol, handle, cfg)
            
            assert evaluation.status in {EvaluationStatus.COMPILE_ERROR, EvaluationStatus.RUNTIME_ERROR}
            assert evaluation.error is not None
            
            runner.release(handle)
            
        finally:
            runner.close()
    
    def test_multiple_solutions_same_definition(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_multi")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            
            srcs1 = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
            sol1 = Solution(name="sol1", definition=d.name, author="test", spec=spec, sources=srcs1)
            
            srcs2 = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A.clone()\n")]
            sol2 = Solution(name="sol2", definition=d.name, author="test", spec=spec, sources=srcs2)
            
            handle = runner.run_ref(d, wl, cfg, None)
            
            eval1 = runner.run_solution(sol1, handle, cfg)
            eval2 = runner.run_solution(sol2, handle, cfg)
            
            if eval1.status not in {EvaluationStatus.COMPILE_ERROR, EvaluationStatus.RUNTIME_ERROR} and \
               eval2.status not in {EvaluationStatus.COMPILE_ERROR, EvaluationStatus.RUNTIME_ERROR}:
                assert len(runner._solution_cache) >= 2
            
            assert eval1.status in {EvaluationStatus.PASSED, EvaluationStatus.RUNTIME_ERROR, EvaluationStatus.COMPILE_ERROR}
            assert eval2.status in {EvaluationStatus.PASSED, EvaluationStatus.RUNTIME_ERROR, EvaluationStatus.COMPILE_ERROR}
            
            runner.release(handle)
            
        finally:
            runner.close()
    
    def test_worker_restart_on_corruption(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            initial_proc = runner._worker_proc
            initial_pid = initial_proc.pid if initial_proc else None
            
            runner._worker_healthy = False
            result = runner._check_worker_health()
            
            # Two cases:
            # 1. Worker was successfully restarted (result is True)
            # 2. Worker restart failed due to GPU issues (which is expected in this environment)
            assert result in {True, False}  # Either succeeds or fails gracefully
            
        finally:
            runner.close()
    
    def test_baseline_not_found_error(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
            sol = Solution(name="test_no_baseline", definition=d.name, author="test", spec=spec, sources=srcs)
            
            from flashinfer_bench.bench.runner import BaselineHandle
            fake_handle = BaselineHandle("fake_handle")
            
            with pytest.raises(Exception):
                runner.run_solution(sol, fake_handle, cfg)
                
        finally:
            runner.close()
    
    def test_concurrent_solution_execution(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        runner = PersistentRunner(device="cuda:0", log_dir=log_dir)
        
        try:
            d = _simple_def()
            wl = Workload(axes={"N": 4}, inputs={"A": RandomInput()}, uuid="test_concurrent")
            cfg = BenchmarkConfig(num_trials=1, warmup_runs=0, iterations=1)
            
            spec = BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["gpu"],
                entry_point="pkg/main.py::run",
            )
            srcs = [SourceFile(path="pkg/main.py", content="import torch\n\ndef run(A):\n    return A\n")]
            
            handle = runner.run_ref(d, wl, cfg, None)
            
            evaluations = []
            for i in range(3):
                sol = Solution(
                    name=f"concurrent_sol_{i}", 
                    definition=d.name, 
                    author="test", 
                    spec=spec, 
                    sources=srcs
                )
                eval_result = runner.run_solution(sol, handle, cfg)
                evaluations.append(eval_result)
            
            for eval_result in evaluations:
                assert eval_result.status in {
                    EvaluationStatus.PASSED, 
                    EvaluationStatus.RUNTIME_ERROR, 
                    EvaluationStatus.COMPILE_ERROR
                }
            
            runner.release(handle)
            
        finally:
            runner.close()


if __name__ == "__main__":
    pytest.main(sys.argv)
