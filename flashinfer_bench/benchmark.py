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
    
    # Hardcoded to compile cutlass, cublas, and cudnn for now, will allow user specification in the future.
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


def _run_reference_benchmark(
    definition: Definition, config: BenchmarkConfig, 
    correctness_inputs: List[List[torch.Tensor]], performance_inputs: List[List[torch.Tensor]]
) -> Dict[str, Any]:
    """
    Run reference implementation benchmark in an isolated subprocess.
    """
    logger = logging.getLogger(f"ReferenceWorker-{os.getpid()}")
    
    try:
        ref_callable = _compile_python_code(
            [{"path": "main.py", "content": definition.reference}], 
            "run"
        )
    except Exception as e:
        logger.error(f"Reference compilation error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "COMPILE_ERROR",
            "error": str(e),
            "environment": _format_environment(DeviceManager(config.device)),
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        device_manager = DeviceManager(config.device)
        
        ref_outputs = []
        ref_latencies = []
        
        with torch.no_grad():
            for inputs in correctness_inputs:
                device_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        device_inputs.append(inp.to(device_manager.device))
                    else:
                        device_inputs.append(inp)
                
                ref_output = ref_callable(*device_inputs)
                torch.cuda.synchronize()
                if isinstance(ref_output, torch.Tensor):
                    ref_output = ref_output.cpu()
                elif isinstance(ref_output, dict):
                    ref_output = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in ref_output.items()}
                ref_outputs.append(ref_output)
        
        # Performance testing
        for inputs in performance_inputs:
            device_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    device_inputs.append(inp.to(device_manager.device))
                else:
                    device_inputs.append(inp)
            
            ref_latency = _time_kernel(ref_callable, device_inputs, config.warmup_runs, config.iterations)
            ref_latencies.append(ref_latency)
                
        return {
            "status": "PASSED",
            "outputs": ref_outputs,
            "latencies": ref_latencies,
            "environment": _format_environment(device_manager),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error in reference benchmark execution: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "RUNTIME_ERROR",
            "error": str(e),
            "environment": _format_environment(DeviceManager(config.device)),
            "timestamp": datetime.now().isoformat(),
        }


def _run_solution_benchmark(
    definition: Definition, solution: Solution, config: BenchmarkConfig,
    correctness_inputs: List[List[torch.Tensor]], performance_inputs: List[List[torch.Tensor]]
) -> Dict[str, Any]:
    """
    Run solution implementation benchmark in an isolated subprocess.
    """
    logger = logging.getLogger(f"SolutionWorker-{os.getpid()}")
    
    try:
        impl_callable = build_solution(solution)
    except Exception as e:
        logger.error(f"Solution compilation error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "COMPILE_ERROR",
            "error": str(e),
            "environment": _format_environment(DeviceManager(config.device)),
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        device_manager = DeviceManager(config.device)
        
        impl_outputs = []
        impl_latencies = []
        
        with torch.no_grad():
            for inputs in correctness_inputs:
                device_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        device_inputs.append(inp.to(device_manager.device))
                    else:
                        device_inputs.append(inp)
                
                impl_output = impl_callable(*device_inputs)
                torch.cuda.synchronize()
                if isinstance(impl_output, torch.Tensor):
                    impl_output = impl_output.cpu()
                elif isinstance(impl_output, dict):
                    impl_output = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in impl_output.items()}
                impl_outputs.append(impl_output)
        
        # Performance testing
        for inputs in performance_inputs:
            device_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    device_inputs.append(inp.to(device_manager.device))
                else:
                    device_inputs.append(inp)
            
            impl_latency = _time_kernel(impl_callable, device_inputs, config.warmup_runs, config.iterations)
            impl_latencies.append(impl_latency)
                
        return {
            "status": "PASSED",
            "outputs": impl_outputs,
            "latencies": impl_latencies,
            "environment": _format_environment(device_manager),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error in solution benchmark execution: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "RUNTIME_ERROR",
            "error": str(e),
            "environment": _format_environment(DeviceManager(config.device)),
            "timestamp": datetime.now().isoformat(),
        }


def _format_evaluation(
    status: str,
    solution_name: str,
    workload: Dict[str, Any],
    correctness: Optional[Dict[str, Any]] = None,
    performance: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    default_correctness = {
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
    }
    
    default_performance = {
        "latency_ms": 0.0,
        "reference_latency_ms": 0.0,
        "speedup_factor": 0.0,
    }
    
    evaluation = {
        "status": status,
        "log_file": f"{solution_name}_{hash(str(workload))}.log",
        "correctness": correctness if correctness is not None else default_correctness,
        "performance": performance if performance is not None else default_performance,
        "environment": environment if environment is not None else {},
        "timestamp": datetime.now().isoformat(),
    }
    
    if error_message:
        evaluation["error"] = error_message
    
    return evaluation


def _run_single_benchmark(
    definition: Definition, solution: Solution, workload: Dict, config: BenchmarkConfig
) -> Trace:
    """
    Run a single benchmark by coordinating separate reference and solution subprocesses.
    This function orchestrates the comparison between reference and solution implementations.
    """
    logger = logging.getLogger(f"BenchmarkCoordinator-{os.getpid()}")
    
    try:
        seed_manager = SeedManager()
        device_manager = DeviceManager(config.device)
        
        correctness_inputs = []
        for trial in range(config.correctness_trials):
            inputs = _generate_test_inputs(definition, workload, device_manager)
            cpu_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    cpu_inputs.append(inp.cpu())
                else:
                    cpu_inputs.append(inp)
            correctness_inputs.append(cpu_inputs)
        
        performance_inputs = []
        for trial in range(config.performance_trials):
            inputs = _generate_test_inputs(definition, workload, device_manager)
            cpu_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    cpu_inputs.append(inp.cpu())
                else:
                    cpu_inputs.append(inp)
            performance_inputs.append(cpu_inputs)
        
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=1) as pool:
            ref_result = pool.apply(
                _run_reference_benchmark, args=(definition, config, correctness_inputs, performance_inputs)
            )
        
        if ref_result["status"] != "PASSED":
            logger.error(f"Reference benchmark failed with status {ref_result['status']}: {ref_result.get('error', 'Unknown error')}")
            evaluation = _format_evaluation(
                status=ref_result["status"],
                solution_name=solution.name,
                workload=workload,
                environment=ref_result.get("environment", {}),
                error_message=ref_result.get("error")
            )
            return Trace(
                definition=definition.name,
                solution=solution.name,
                workload=workload,
                evaluation=evaluation,
            )
        
        with ctx.Pool(processes=1) as pool:
            impl_result = pool.apply(
                _run_solution_benchmark, args=(definition, solution, config, correctness_inputs, performance_inputs)
            )
        
        if impl_result["status"] == "COMPILE_ERROR":
            evaluation = _format_evaluation(
                status="COMPILE_ERROR",
                solution_name=solution.name,
                workload=workload,
                environment=impl_result.get("environment", {}),
                error_message=impl_result.get("error")
            )
            return Trace(
                definition=definition.name,
                solution=solution.name,
                workload=workload,
                evaluation=evaluation,
            )
        
        if impl_result["status"] != "PASSED":
            evaluation = _format_evaluation(
                status="RUNTIME_ERROR",
                solution_name=solution.name,
                workload=workload,
                environment=impl_result.get("environment", {}),
                error_message=impl_result.get("error")
            )
            return Trace(
                definition=definition.name,
                solution=solution.name,
                workload=workload,
                evaluation=evaluation,
            )
        
        ref_outputs = ref_result["outputs"]
        impl_outputs = impl_result["outputs"]
        
        if len(ref_outputs) != len(impl_outputs):
            raise ValueError(f"Mismatch in number of outputs: ref={len(ref_outputs)}, impl={len(impl_outputs)}")
        
        pass_count = 0
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        error_details = {}
        
        for i, (ref_output, impl_output) in enumerate(zip(ref_outputs, impl_outputs)):
            is_correct, trial_error_details = CorrectnessChecker.check_correctness(
                ref_output, impl_output, rtol=config.rtol, atol=config.atol
            )
            
            if not is_correct:
                error_type = trial_error_details.get("error_type", "comparison_failed")
                if error_type in ["shape_mismatch", "dtype_mismatch", "device_mismatch"]:
                    logger.error(f"Structure mismatch for trial {i}: {trial_error_details.get('error_message', 'Unknown error')}")
                    
                    avg_impl_latency = sum(impl_result["latencies"]) / len(impl_result["latencies"]) if impl_result["latencies"] else 0.0
                    avg_ref_latency = sum(ref_result["latencies"]) / len(ref_result["latencies"]) if ref_result["latencies"] else 0.0
                    
                    correctness_info = {
                        "error_type": error_type,
                        "error_message": trial_error_details.get("error_message", ""),
                        "max_absolute_error": trial_error_details.get("max_absolute_error", 0.0),
                        "max_relative_error": trial_error_details.get("max_relative_error", 0.0),
                    }
                    
                    performance_info = {
                        "latency_ms": avg_impl_latency,
                        "reference_latency_ms": avg_ref_latency,
                        "speedup_factor": 0.0,
                    }
                    
                    evaluation = _format_evaluation(
                        status="INCORRECT",
                        solution_name=solution.name,
                        workload=workload,
                        correctness=correctness_info,
                        performance=performance_info,
                        environment=impl_result.get("environment", {})
                    )
                    return Trace(
                        definition=definition.name,
                        solution=solution.name,
                        workload=workload,
                        evaluation=evaluation,
                    )
                
                # This is a numerical comparison failure
                trial_abs_diff = trial_error_details.get("max_absolute_error", 0.0)
                trial_rel_diff = trial_error_details.get("max_relative_error", 0.0)
                max_abs_diff = max(max_abs_diff, trial_abs_diff)
                max_rel_diff = max(max_rel_diff, trial_rel_diff)
                
                # Store the most detailed error message for reporting
                if not error_details or trial_abs_diff > error_details.get("max_absolute_error", 0.0):
                    error_details = trial_error_details
            else:
                pass_count += 1
        
        status = "PASSED" if pass_count == len(ref_outputs) else "INCORRECT"
        
        impl_latencies = torch.tensor(impl_result["latencies"])
        ref_latencies = torch.tensor(ref_result["latencies"])
        speedups = ref_latencies / impl_latencies
        speedup = torch.exp(torch.log(speedups).mean()).item()
        
        correctness_info = {
            "max_absolute_error": max_abs_diff,
            "max_relative_error": max_rel_diff,
        }
        
        if status != "PASSED" and error_details:
            correctness_info.update({
                "error_type": error_details.get("error_type", "comparison_failed"),
                "error_message": error_details.get("error_message", ""),
            })

        performance_info = {
            "latency_ms": impl_latencies.mean().item(),
            "reference_latency_ms": ref_latencies.mean().item(),
            "speedup_factor": speedup,
        }

        evaluation = _format_evaluation(
            status=status,
            solution_name=solution.name,
            workload=workload,
            correctness=correctness_info,
            performance=performance_info,
            environment=impl_result.get("environment", {})
        )

    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}")
        logger.error(traceback.format_exc())

        try:
            environment = impl_result.get("environment", {})
        except NameError:
            environment = _format_environment(DeviceManager(config.device))

        evaluation = _format_evaluation(
            status="RUNTIME_ERROR",
            solution_name=solution.name,
            workload=workload,
            environment=environment,
            error_message=str(e)
        )

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

        for i, workload in enumerate(workload_list):
            self.logger.info(f"Running workload {i+1}/{len(workload_list)}")

            result = _run_single_benchmark(definition, target_solution, workload, config)
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