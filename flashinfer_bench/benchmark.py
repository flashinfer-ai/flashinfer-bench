import importlib.util
import linecache
import logging
import multiprocessing as mp
import os
import tempfile
import traceback
import types
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import safetensors.torch as safetensors
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from triton.testing import do_bench

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


def build_solution(solution: Solution) -> Callable:
    """Build a solution into a callable function"""
    language = solution.spec.get("language", "python").lower()
    entry_point = solution.spec.get("entry_point", "run")

    if language == "triton":
        return _compile_triton_code(
            solution.sources[0]["content"], entry_point
        )
    elif language == "cuda":
        return _compile_cuda_code(
            solution.sources, entry_point
        )
    else: # python
        return _compile_python_code(
            solution.sources, entry_point
        )


def _compile_python_code(sources: List[Dict[str, str]], entry_point: str) -> Callable:
    """Compile Python code string and return the specified entry point callable"""
    namespace = {}
    
    for source in sources:
        path = source['path']
        content = source['content']
        annotated_content = f"# Source: {path}\n{content}"
        
        try:
            exec(annotated_content, namespace)
        except Exception as e:
            raise RuntimeError(f"Failed to execute code from '{path}': {e}")

    if entry_point not in namespace:
        available_callables = [name for name, obj in namespace.items() if callable(obj) and not name.startswith('_')]
        raise ValueError(
            f"Entry point '{entry_point}' not found in code. "
        )

    return namespace[entry_point]


def _compile_triton_code(code_string: str, entry_point: str) -> Callable:
    """
    Compile Triton code string and return the specified entry point callable
    """
    mod_name = f"_fi_bench_triton_tmp_{uuid.uuid4().hex}"
    fake_filename = f"<{mod_name}>"
    linecache.cache[fake_filename] = (
        len(code_string),
        None,
        code_string.splitlines(True),
        fake_filename,
    )

    mod = types.ModuleType(mod_name)
    exec(compile(code_string, fake_filename, "exec"), mod.__dict__)

    if not hasattr(mod, entry_point):
        raise ValueError(f"{entry_point!r} not found")
    return getattr(mod, entry_point)

def _compile_cuda_code(sources: List[Dict[str, str]], entry_point: str) -> Callable:
    """
    Compile CUDA code from sources and return the specified entry point callable.
    Note: .cpp wrapper is required, do NOT include header files, pybind11 is optional.
    """
    cpp_sources = []
    cuda_sources = []
    has_pybind = False
    
    for source in sources:
        path = source['path']
        content = source['content']
        
        if 'pybind11' in content or 'PYBIND11_MODULE' in content:
            has_pybind = True

        if path.endswith('.cu'):
            cuda_sources.append(content)
        elif path.endswith('.cpp') or path.endswith('.cc') or path.endswith('.cxx'):
            cpp_sources.append(content)
    
    module_name = f"_fi_bench_cuda_tmp_{uuid.uuid4().hex[:8]}"
    
    load_args = {
        'name': module_name,
        'cpp_sources': cpp_sources if cpp_sources else None,
        'cuda_sources': cuda_sources if cuda_sources else None,
        'verbose': False,
        'with_cuda': True,
        'extra_include_paths': ['./3rdparty/cutlass/include'],
        'extra_ldflags': ['-lcublas', '-lcudnn'],
    }
    
    if not has_pybind:
        load_args['functions'] = [entry_point]
    
    load_args = {k: v for k, v in load_args.items() if v is not None}
    
    try:
        module = load_inline(**load_args)
    except Exception as e:
        raise RuntimeError(f"Failed to compile CUDA code: {e}")
    
    if not hasattr(module, entry_point):
        raise ValueError(f"{entry_point!r} not found")
    return getattr(module, entry_point)

def _generate_test_inputs(
    definition: Definition, workload: Dict[str, Any], device_manager: DeviceManager
) -> List[torch.Tensor]:
    """Generate test inputs based on definition and workload specification"""
    inputs = []
    workload_axes = workload["axes"]
    workload_inputs = workload["inputs"]

    input_shapes = definition.get_input_shapes(workload_axes)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
        "bool": torch.bool,
    }

    for input_name, input_spec in definition.inputs.items():
        shape = input_shapes[input_name]
        dtype_str = input_spec["dtype"]
        dtype = dtype_map.get(dtype_str, torch.float32)

        if input_name in workload_inputs:
            input_desc = workload_inputs[input_name]
            input_type = input_desc["type"]

            if input_type == "random":
                if dtype == torch.bool:
                    tensor = torch.randint(0, 2, shape, dtype=dtype, device=device_manager.device)
                else:
                    tensor = torch.randn(shape, dtype=dtype, device=device_manager.device)
            elif input_type == "scalar":
                value = input_desc["value"]
                tensor = value
            elif input_type == "safetensors":
                if safetensors is None:
                    raise ImportError("safetensors library is required but not installed")

                path = input_desc["path"]
                tensor_key = input_desc["tensor_key"]
                tensors = safetensors.load_file(path)
                if tensor_key not in tensors:
                    raise ValueError(
                        f"Tensor key '{tensor_key}' not found in safetensors file '{path}'"
                    )

                tensor = tensors[tensor_key].to(device=device_manager.device, dtype=dtype)

                if list(tensor.shape) != shape:
                    raise ValueError(
                        f"Tensor '{input_name}' shape mismatch. Expected {shape}, got {list(tensor.shape)}"
                    )
            else:
                raise ValueError(f"Unsupported input type '{input_type}' for input '{input_name}'")

        else:
            # Default to random generation if not specified in workload
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

    # One-time pre-run
    with torch.no_grad():
        callable_func(*inputs)
    torch.cuda.synchronize()

    return do_bench(
        lambda: callable_func(*inputs),
        warmup=warmup,
        rep=iterations,
    )


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

    try:
        ref_callable = _compile_python_code(definition.reference, "run")
        try:
            impl_callable = build_solution(solution)
        except Exception as compile_err:
            logger.error(f"Compilation error: {compile_err}")
            logger.error(traceback.format_exc())
            evaluation = {
                "status": "COMPILE_ERROR",
                "log_file": f"{solution.name}_{hash(str(workload))}.log",
                "correctness": None,
                "performance": None,
                "environment": _format_environment(DeviceManager(config.device)),
                "timestamp": datetime.now().isoformat(),
            }
            return Trace(
                definition=definition.name,
                solution=solution.name,
                workload=workload,
                evaluation=evaluation,
            )

        seed_manager = SeedManager()
        device_manager = DeviceManager(config.device)

        # Correctness check
        num_trials = config.correctness_trials
        abs_diffs = []
        rel_diffs = []
        pass_count = 0

        with torch.no_grad():
            for _ in range(num_trials):
                inputs = _generate_test_inputs(definition, workload, device_manager)

                trial_data_ref = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                trial_data_impl = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]

                ref_output = ref_callable(*trial_data_ref)
                torch.cuda.synchronize()

                impl_output = impl_callable(*trial_data_impl)
                torch.cuda.synchronize()

                shape_correct, err_msg = CorrectnessChecker.validate_shapes(ref_output, impl_output)

                if not shape_correct:
                    logger.error(f"Shape mismatch for trial {_}: {err_msg}")
                    status = "SHAPE_MISMATCH"
                    break

                max_abs_diff = CorrectnessChecker.max_absolute_diff(ref_output, impl_output)
                max_rel_diff = CorrectnessChecker.max_relative_diff(ref_output, impl_output)

                if max_abs_diff <= config.max_diff_limit:
                    pass_count += 1
                abs_diffs.append(max_abs_diff)
                rel_diffs.append(max_rel_diff)

        seed_manager.reset_seed()

        max_abs_diff = max(abs_diffs, default=None)  # None if list is empty, i.e. shape mismatch
        max_rel_diff = max(rel_diffs, default=None)

        status = "PASSED" if pass_count == num_trials else "INCORRECT_RESULT"

        if status == "INCORRECT_RESULT" or status == "SHAPE_MISMATCH":
            evaluation = {
                "status": status,
                "log_file": f"{solution.name}_{hash(str(workload))}.log",
                "correctness": {
                    "max_relative_error": max_rel_diff,
                    "max_absolute_error": max_abs_diff,
                },
                "performance": {
                    "latency_ms": 0.0,
                    "reference_latency_ms": 0.0,
                    "speedup_factor": 0.0,
                },
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
        impl_latencies = []
        ref_latencies = []
        
        for _ in range(5): # Hardcoded 5 for now, can maybe add to BenchmarkConfig in the future
            timing_inputs = _generate_test_inputs(definition, workload, device_manager)
            impl_timing_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in timing_inputs]
            ref_timing_inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in timing_inputs]
            
            impl_latency = _time_kernel(impl_callable, impl_timing_inputs, config.warmup_runs, config.iterations)
            ref_latency = _time_kernel(ref_callable, ref_timing_inputs, config.warmup_runs, config.iterations)
            
            impl_latencies.append(impl_latency)
            ref_latencies.append(ref_latency)

        impl_latency = sum(impl_latencies) / len(impl_latencies)
        ref_latency = sum(ref_latencies) / len(ref_latencies)
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
            # TODO: differentiate between compile/runtime errors
            "status": "RUNTIME_ERROR",
            # TODO: save actual log file
            "log_file": f"{solution.name}_{hash(str(workload))}.log",
            "correctness": None,
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
