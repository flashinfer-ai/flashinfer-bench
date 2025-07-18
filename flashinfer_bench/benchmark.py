import importlib.util
import logging
import multiprocessing as mp
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .benchmark_config import BenchmarkConfig
from .builders.base import BuilderRegistry, BuildError
from .definition import Definition
from .solution import Solution
from .trace import Trace
from .trace_set import TraceSet
from .utils.benchmark_utils import CorrectnessChecker, DeviceManager, SeedManager
from .utils.validation import (
    validate_axis,
    validate_constraints,
    validate_reference_code,
    validate_tensor,
    validate_workload_axes,
)


def _compile_python_code(code_string: str, entry_point: str) -> Callable:
    """Compile Python code string and return the specified entry point callable"""
    namespace = {}
    exec(code_string, namespace)

    if entry_point not in namespace:
        raise ValueError(f"Entry point '{entry_point}' not found in code")

    return namespace[entry_point]


def _compile_triton_code(code_string: str, entry_point: str) -> Callable:
    """
    Compile Triton code using tempfile approach to handle @triton.jit decorator
    Inspired by KernelBench-Triton https://github.com/ScalingIntelligence/KernelBench/pull/35/files#diff-7c33c37dd2ca3f92b111b25d2c1168f5c98f308c1f2d0e2add9e4b8b240e5918
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code_string)
        tempfile_path = tmp_file.name

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    if not hasattr(temp_module, entry_point):
        os.unlink(tempfile_path)
        raise ValueError(f"Entry point '{entry_point}' not found in Triton code")

    return getattr(temp_module, entry_point), tempfile_path


def _generate_test_inputs(
    definition: Definition, workload_axes: Dict[str, int], device_manager: DeviceManager
) -> List[torch.Tensor]:
    """Generate test inputs based on definition and workload axes"""
    inputs = []

    input_shapes = definition.get_input_shapes(workload_axes)

    for input_name, input_spec in definition.inputs.items():
        shape = input_shapes[input_name]

        dtype_str = input_spec["dtype"]
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        if dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, dtype=dtype, device=device_manager.device)
        else:
            tensor = torch.randn(shape, dtype=dtype, device=device_manager.device)
        inputs.append(tensor)

    return inputs


def _time_kernel(callable_func: Callable, inputs: List[Any], warmup: int, iterations: int) -> float:
    """Time a callable execution using CUDA events"""
    if isinstance(callable_func, nn.Module):
        callable_func.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = callable_func(*inputs)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(iterations):
            _ = callable_func(*inputs)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    return elapsed_time_ms / iterations


def _format_environment(device_manager: DeviceManager) -> Dict[str, Any]:
    device_info = device_manager.get_device_info()
    try:
        from triton import __version__ as triton_version
    except ImportError:
        triton_version = None

    return {
        "device": device_info.get("device_name", None),
        "libs": {
            "torch": torch.__version__,
            "triton": triton_version,
            "cuda": device_info.get("cuda", None),
        },
    }


def _run_single_benchmark(
    definition: Definition, solution: Solution, workload: Dict, config: BenchmarkConfig
) -> Trace:
    """
    Run a single benchmark in an isolated subprocess.
    This function is the entry point for the subprocess.
    """

    logger = logging.getLogger(f"BenchmarkWorker-{os.getpid()}")
    temp_files = []

    try:
        ref_callable = _compile_python_code(definition.reference, "run")

        entry_point = solution.spec.get("entry_point", "run")
        language = solution.spec.get("language", "Python").lower()

        if not solution.sources or len(solution.sources) == 0:
            raise BuildError(f"No source code found for solution: {solution.name}")

        source_content = solution.sources[0]["content"]

        if language == "triton":
            impl_callable, temp_file = _compile_triton_code(source_content, entry_point)
            temp_files.append(temp_file)
        else:
            impl_callable = _compile_python_code(source_content, entry_point)

        seed_manager = SeedManager()
        device_manager = DeviceManager(config.device)

        # Correctness check
        num_trials = config.correctness_trials
        abs_diffs = []
        rel_diffs = []
        pass_count = 0

        with torch.no_grad():
            for _ in range(num_trials):
                inputs = _generate_test_inputs(definition, workload["axes"], device_manager)

                trial_data_ref = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                trial_data_impl = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

                ref_output = ref_callable(*trial_data_ref)
                torch.cuda.synchronize()

                impl_output = impl_callable(*trial_data_impl)
                torch.cuda.synchronize()

                shape_correct, err_msg = CorrectnessChecker.validate_shapes(ref_output, impl_output)

                if not shape_correct:
                    status = "INCORRECT"
                    break

                max_abs_diff = CorrectnessChecker.max_absolute_diff(ref_output, impl_output)
                max_rel_diff = CorrectnessChecker.max_relative_diff(ref_output, impl_output)

                if max_abs_diff <= config.max_diff_limit:
                    pass_count += 1
                abs_diffs.append(max_abs_diff)
                rel_diffs.append(max_rel_diff)

        seed_manager.reset_seed()

        max_abs_diff = max(abs_diffs, default=None) # None if list is empty, i.e. shape mismatch
        max_rel_diff = max(rel_diffs, default=None)

        status = "PASSED" if pass_count == num_trials else "INCORRECT"

        if status == "INCORRECT":
            evaluation = {
                "status": status,
                "log_file": f"{solution.name}_{hash(str(workload))}.log",
                "correctness": {
                    "max_relative_error": max_rel_diff,
                    "max_absolute_error": max_abs_diff,
                },
                "performance": None,
                "environment": _format_environment(device_manager),
                "timestamp": datetime.now().isoformat(),
            }
            return Trace(
                definition=definition.name,
                solution=solution.name,
                workload=workload,
                evaluation=evaluation,
            )

        # Performance testing if correct
        impl_latency = _time_kernel(impl_callable, inputs, config.warmup_runs, config.iterations)
        ref_latency = _time_kernel(ref_callable, inputs, config.warmup_runs, config.iterations)
        speedup = ref_latency / impl_latency if impl_latency > 0 else 0.0

        evaluation = {
            "status": status,
            "log_file": f"{solution.name}_{hash(str(workload))}.log",
            "correctness": {
                "max_relative_error": max_rel_diff,
                "max_absolute_error": max_abs_diff,
            },
            "performance": {
                "latency_ms": impl_latency,
                "reference_latency_ms": ref_latency,
                "speedup_factor": speedup,
            },
            "environment": _format_environment(device_manager),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}")
        logger.error(traceback.format_exc())

        evaluation = {
            "status": "RUNTIME_ERROR",
            "log_file": f"{solution.name}_{hash(str(workload))}.log",
            "correctness": None,
            "performance": None,
            "environment": _format_environment(device_manager),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    return Trace(
        definition=definition.name,
        solution=solution.name,
        workload=workload,
        evaluation=evaluation,
    )


class Benchmark:
    """
    The Benchmark class is a stateless runner of benchmarks.
    It takes a TraceSet and executes benchmarks on the solutions,
    updating the TraceSet with new trace results.
    """

    def __init__(self, trace_set: TraceSet):
        """Initialize Benchmark with a TraceSet containing definitions, solutions, and workloads"""
        self.trace_set = trace_set
        self.logger = logging.getLogger(__name__)

    @property
    def traces(self) -> Dict[str, List[Trace]]:
        """Get all traces from the associated TraceSet"""
        return self.trace_set.traces

    @property
    def definitions(self) -> Dict[str, Definition]:
        """Get definitions from the associated TraceSet"""
        return self.trace_set.definitions

    @property
    def solutions(self) -> Dict[str, List[Solution]]:
        """Get solutions from the associated TraceSet"""
        return self.trace_set.solutions

    @property
    def workload(self) -> Dict[str, List[Trace]]:
        """Get workload traces from the associated TraceSet"""
        return self.trace_set.workload

    def _validate(self) -> None:
        """Validate loaded definitions, solutions, and workloads for consistency"""
        for def_name, definition in self.definitions.items():
            for axis_name, axis_def in definition.axes.items():
                validate_axis(axis_def)

            for tensor_name, tensor_def in definition.inputs.items():
                validate_tensor(tensor_def, definition.axes)
            for tensor_name, tensor_def in definition.outputs.items():
                validate_tensor(tensor_def, definition.axes)

            validate_reference_code(definition.reference)

            if hasattr(definition, "constraints") and definition.constraints:
                validate_constraints(definition.constraints, definition.axes)

        for def_name, solution_list in self.solutions.items():
            if def_name not in self.definitions:
                raise ValueError(f"Solutions reference undefined definition: {def_name}")

            for solution in solution_list:
                if solution.definition != def_name:
                    raise ValueError(
                        f"Solution '{solution.name}' has mismatched definition reference"
                    )

        for def_name, workload_list in self.workload.items():
            if def_name not in self.definitions:
                raise ValueError(f"Workloads reference undefined definition: {def_name}")

            definition = self.definitions[def_name]
            for workload_trace in workload_list:
                if "axes" in workload_trace.workload:
                    validate_workload_axes(workload_trace.workload["axes"], definition.axes)

    def run_solution(self, solution_name: str, config: BenchmarkConfig = BenchmarkConfig()) -> None:
        """Run benchmarks for a specific solution

        Args:
            solution_name: Name of the solution to run
            config: Benchmark configuration
        """
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))

        # Find the solution and its definition
        target_solution = None
        target_def_name = None

        for def_name, solution_list in self.solutions.items():
            for solution in solution_list:
                if solution.name == solution_name:
                    target_solution = solution
                    target_def_name = def_name
                    break
            if target_solution:
                break

        if not target_solution:
            raise ValueError(f"Solution '{solution_name}' not found")

        if target_def_name not in self.definitions:
            raise ValueError(
                f"Solution '{solution_name}' references undefined definition: {target_def_name}"
            )

        definition = self.definitions[target_def_name]

        # Get workloads
        workload_list = [trace.workload for trace in self.workload.get(target_def_name, [])]
        if not workload_list:
            raise ValueError(f"No workloads found for definition {target_def_name}")

        self.logger.info(f"Evaluating {target_solution.name} on {len(workload_list)} workloads")

        # Run each workload in a subprocess
        for i, workload in enumerate(workload_list):
            self.logger.info(f"Running workload {i+1}/{len(workload_list)}")

            # Use multiprocessing to run in isolation
            # Note that we don't catch multiprocessing errors
            ctx = mp.get_context("spawn")  # Use spawn to get clean CUDA context
            with ctx.Pool(processes=1) as pool:
                result = pool.apply(
                    _run_single_benchmark, args=(definition, target_solution, workload, config)
                )
                self.trace_set.add_trace(result)

                # Log result
                if result.evaluation["status"] == "PASSED":
                    self.logger.info(
                        f"Workload {i+1}: PASSED "
                        f"(speedup: {result.evaluation['performance']['speedup_factor']:.2f}x)"
                    )
                else:
                    self.logger.warning(f"Workload {i+1}: {result.evaluation['status']}")
                self.logger.debug(result)

    def run(self, config: BenchmarkConfig = BenchmarkConfig()) -> None:
        """
        Updates the associated TraceSet with benchmark results.
        This method simply iterates through all solutions and calls run_solution for each.

        Args:
            config: Benchmark configuration
        """
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))

        self._validate()

        self.logger.info("Starting benchmark run...")

        # Collect all solutions
        solutions = []
        for def_name, solution_list in self.solutions.items():
            for solution in solution_list:
                solutions.append((def_name, solution))

        total_solutions = len(solutions)
        self.logger.info(f"Found {total_solutions} solutions to benchmark")

        # Run each solution
        for i, (def_name, solution) in enumerate(solutions):
            self.logger.info(
                f"Running solution {i+1}/{total_solutions}: {solution.name} for definition {def_name}"
            )

            try:
                self.run_solution(solution.name, config)
            except Exception as e:
                self.logger.error(f"Failed to run solution {solution.name}: {str(e)}")
                continue

        # Final summary
        trace_count = sum(len(traces) for traces in self.traces.values())
        self.logger.info(f"Benchmark completed with {trace_count} traces")
