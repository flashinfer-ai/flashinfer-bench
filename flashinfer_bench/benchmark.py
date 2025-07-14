import importlib.util
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .benchmark_config import BenchmarkConfig
from .builders.base import BaseBuilder, BuilderRegistry, BuildError
from .definition import Definition
from .solution import Solution
from .trace import Trace
from .trace_set import TraceSet
from .utils.benchmark_utils import CorrectnessChecker, DeviceManager, SeedManager
from .utils.json_utils import load_json, load_jsonl
from .utils.validation import (
    validate_axis,
    validate_constraints,
    validate_reference_code,
    validate_tensor,
    validate_workload_axes,
)


class Benchmark:
    """
    The Benchmark class is a stateless runner of benchmarks.
    It is responsible for loading definitions, solutions, and specific workload
    configurations, building and validating the code, executing the benchmarks,
    and finally outputting a series of Trace objects as results.
    """

    def __init__(self):
        self.definitions: Dict[str, Definition] = {}
        self.solutions: Dict[str, List[Solution]] = {}
        self.workloads: Dict[str, List[Dict]] = {}
        self._reference_callables: Dict[str, Callable] = {}
        self._solution_callables: Dict[str, Dict[str, Callable]] = {}
        self._temp_files: List[Any] = []  # Track temp files for cleanup

        self.builder_registry = BuilderRegistry()
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        """Cleanup temp files when object is destroyed"""
        self._cleanup_temp_files()

    @classmethod
    def from_path(cls, path: Union[str, List[str]]) -> "Benchmark":
        """Create Benchmark instance from local path(s)

        Expected directory structure:
        <path>/
        ├── definitions/
        │   └── gemm.json
        ├── solutions/
        │   └── gemm_triton_gemma.json
        └── traces/
            ├── workloads/
            │   ├── gemm_b128.jsonl
            └── gemm.jsonl
        """
        benchmark = cls()

        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        for p in paths:
            benchmark._load_from_path(Path(p))

        benchmark._validate()
        return benchmark

    @classmethod
    def from_hub(cls) -> "Benchmark":
        """Create Benchmark instance from FlashInfer Hub"""
        benchmark = cls()
        # TODO: Implement hub integration
        benchmark.logger.warning("Hub integration not implemented yet")
        return benchmark

    def _load_from_path(self, path: Path) -> None:
        """Load definitions, solutions, and workloads from directory of given path"""
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        definitions_dir = path / "definitions"
        if definitions_dir.exists():
            for json_file in definitions_dir.glob("*.json"):
                definition = load_json(json_file, Definition)
                self.definitions[definition.name] = definition

        for dir_name in ["solutions", "implementations"]:
            solutions_dir = path / dir_name
            if solutions_dir.exists():
                for json_file in solutions_dir.glob("*.json"):
                    solution = load_json(json_file, Solution)
                    if solution.definition not in self.solutions:
                        self.solutions[solution.definition] = []
                    self.solutions[solution.definition].append(solution)

        traces_dir = path / "traces"
        if traces_dir.exists():
            # Load workload-only traces from workloads subdirectory
            workloads_dir = traces_dir / "workloads"
            if workloads_dir.exists():
                for jsonl_file in workloads_dir.glob("*.jsonl"):
                    workload_traces = load_jsonl(jsonl_file, Trace)
                    for trace in workload_traces:
                        if trace.definition not in self.workloads:
                            self.workloads[trace.definition] = []
                        self.workloads[trace.definition].append(trace.workload)

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

        for def_name, workload_list in self.workloads.items():
            if def_name not in self.definitions:
                raise ValueError(f"Workloads reference undefined definition: {def_name}")

            definition = self.definitions[def_name]
            for workload in workload_list:
                if "axes" in workload:
                    validate_workload_axes(workload["axes"], definition.axes)

    def _build_reference_testing(self, definition_name: str) -> Callable:
        """TODO: In development, kernel builder system"""
        if definition_name in self._reference_callables:
            return self._reference_callables[definition_name]

        if definition_name not in self.definitions:
            raise ValueError(f"Definition not found: {definition_name}")

        definition = self.definitions[definition_name]

        try:
            validate_reference_code(definition.reference)

            builder = self.builder_registry.get_reference_builder()
            callable_obj = builder.build_reference(definition)
            self._reference_callables[definition_name] = callable_obj
            return callable_obj
        except Exception as e:
            raise BuildError(f"Failed to build reference for {definition_name}: {str(e)}")

    def _build_solution_testing(self, solution: Solution) -> Callable:
        """TODO: In development, kernel builder system"""
        if solution.definition in self._solution_callables:
            if solution.name in self._solution_callables[solution.definition]:
                return self._solution_callables[solution.definition][solution.name]

        try:
            builder = self.builder_registry.get_builder(solution)

            definition = self.definitions[solution.definition]
            if not builder.validate_signature(definition, solution):
                raise BuildError(f"Signature mismatch for solution: {solution.name}")

            callable_obj = builder.build_implementation(solution)

            if solution.definition not in self._solution_callables:
                self._solution_callables[solution.definition] = {}
            self._solution_callables[solution.definition][solution.name] = callable_obj

            return callable_obj
        except Exception as e:
            raise BuildError(f"Failed to build solution {solution.name}: {str(e)}")

    def _build_reference(self, definition_name: str) -> Callable:
        """Build reference callable for definition using direct compilation"""
        if definition_name in self._reference_callables:
            return self._reference_callables[definition_name]

        if definition_name not in self.definitions:
            raise ValueError(f"Definition not found: {definition_name}")

        definition = self.definitions[definition_name]

        try:
            validate_reference_code(definition.reference)

            callable_obj = self._compile_python_code(definition.reference, "run")
            self._reference_callables[definition_name] = callable_obj
            return callable_obj
        except Exception as e:
            raise BuildError(f"Failed to build reference for {definition_name}: {str(e)}")

    def _build_solution(self, solution: Solution) -> Callable:
        """Simple build implementation callable from solution using direct compilation"""
        if solution.definition in self._solution_callables:
            if solution.name in self._solution_callables[solution.definition]:
                return self._solution_callables[solution.definition][solution.name]

        try:
            # Get the entry point from solution spec
            entry_point = solution.spec.get("entry_point", "run")
            language = solution.spec.get("language", "Python").lower()

            # Get the source code (assuming single source file for now)
            if not solution.sources or len(solution.sources) == 0:
                raise BuildError(f"No source code found for solution: {solution.name}")

            source_content = solution.sources[0]["content"]

            if language == "triton":
                # Use tempfile approach for Triton code
                callable_obj = self._compile_triton_code(source_content, entry_point)
            else:
                # Use compile/exec for Python code
                callable_obj = self._compile_python_code(source_content, entry_point)

            # Cache the callable
            if solution.definition not in self._solution_callables:
                self._solution_callables[solution.definition] = {}
            self._solution_callables[solution.definition][solution.name] = callable_obj

            return callable_obj
        except Exception as e:
            raise BuildError(f"Failed to build solution {solution.name}: {str(e)}")

    def _compile_python_code(self, code_string: str, entry_point: str) -> Callable:
        """Compile Python code string and return the specified entry point callable"""
        try:
            namespace = {}

            exec(code_string, namespace)

            if entry_point not in namespace:
                raise ValueError(f"Entry point '{entry_point}' not found in code")

            return namespace[entry_point]
        except Exception as e:
            raise BuildError(f"Failed to compile Python code: {str(e)}")

    def _compile_triton_code(self, code_string: str, entry_point: str) -> Callable:
        """
        Compile Triton code using tempfile approach to handle @triton.jit decorator
        Inspired by KernelBench-Triton https://github.com/ScalingIntelligence/KernelBench/pull/35/files#diff-7c33c37dd2ca3f92b111b25d2c1168f5c98f308c1f2d0e2add9e4b8b240e5918
        """
        try:
            # Create a temporary named file with a .py extension
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_string)
                tempfile_path = tmp_file.name

            self._temp_files.append(tempfile_path)

            spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            if not hasattr(temp_module, entry_point):
                raise ValueError(f"Entry point '{entry_point}' not found in Triton code")

            return getattr(temp_module, entry_point)
        except Exception as e:
            raise BuildError(f"Failed to compile Triton code: {str(e)}")

    def _cleanup_temp_files(self):
        """Clean up temporary files created during compilation"""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    self.logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
        self._temp_files.clear()

    def _format_environment(self, device_manager: DeviceManager) -> Dict[str, Any]:
        device_info = device_manager.get_device_info()
        return {
            "device": device_info.get("device_str", "unknown"),
            "libs": {
                "torch": torch.__version__,
                "driver_version": device_info.get("driver_version", "unknown"),
                "device_name": device_info.get("device_name", "unknown"),
                "compute_capability": str(device_info.get("compute_capability", "unknown")),
                "backend": device_info.get("backend", "cuda"),
            },
        }

    def _time_kernel(
        self, callable_func: Callable, inputs: List[Any], warmup: int, iterations: int
    ) -> float:
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

    def _generate_test_inputs(
        self, definition: Definition, workload_axes: Dict[str, int], device_manager: DeviceManager
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

    def _evaluate_single_workload(
        self,
        solution: Solution,
        workload: Dict,
        ref_callable: Callable,
        impl_callable: Callable,
        config: BenchmarkConfig,
        seed_manager: SeedManager,
        device_manager: DeviceManager,
    ) -> Trace:
        """Evaluate a solution on a single workload"""
        definition = self.definitions[solution.definition]

        inputs = self._generate_test_inputs(definition, workload["axes"], device_manager)

        is_correct, max_abs_diff, max_rel_diff, _ = self._run_correctness_trials(
            ref_callable, impl_callable, inputs, config, seed_manager, device_manager
        )

        status = "PASSED" if is_correct else "INCORRECT"

        if not is_correct:
            evaluation = {
                "status": status,
                "log_file": f"{solution.name}_{hash(str(workload))}.log",
                "correctness": {
                    "max_relative_error": max_rel_diff,
                    "max_absolute_error": max_abs_diff,
                },
                "performance": {
                    "latency_ms": float("inf"),
                    "reference_latency_ms": float("inf"),
                    "speedup_factor": 0.0,
                },
                "environment": self._format_environment(device_manager),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            try:
                # Performance testing
                impl_latency = self._time_kernel(
                    impl_callable, inputs, config.warmup_runs, config.iterations
                )
                ref_latency = self._time_kernel(
                    ref_callable, inputs, config.warmup_runs, config.iterations
                )

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
                    "environment": self._format_environment(device_manager),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                evaluation = {
                    "status": "RUNTIME_ERROR",
                    "log_file": f"{solution.name}_{hash(str(workload))}.log",
                    "correctness": {
                        "max_relative_error": float("inf"),
                        "max_absolute_error": float("inf"),
                    },
                    "performance": {
                        "latency_ms": float("inf"),
                        "reference_latency_ms": float("inf"),
                        "speedup_factor": 0.0,
                    },
                    "environment": self._format_environment(device_manager),
                    "timestamp": datetime.now().isoformat(),
                }

        return Trace(
            definition=solution.definition,
            solution=solution.name,
            workload=workload,
            evaluation=evaluation,
        )

    def _run_correctness_trials(
        self,
        ref_callable: Callable,
        impl_callable: Callable,
        inputs: List[Any],
        config: BenchmarkConfig,
        seed_manager: SeedManager,
        device_manager: DeviceManager,
        num_trials: int = 1,
    ) -> Tuple[bool, float, float, Dict]:
        """Run correctness trials between reference and implementation"""
        max_diffs = []
        rel_diffs = []
        pass_count = 0
        metadata = {
            "correctness_trials_attempted": num_trials,
            "max_differences": [],
            "rel_differences": [],
            "trial_results": [],
        }

        original_seed = seed_manager.seed
        seed_manager.reset_seed()

        with torch.no_grad():
            for trial in range(num_trials):
                trial_result = {"trial": trial}

                try:
                    # Clone inputs to prevent aliasing
                    trial_data_ref = [
                        x.clone() if isinstance(x, torch.Tensor) else x for x in inputs
                    ]
                    trial_data_impl = [
                        x.clone() if isinstance(x, torch.Tensor) else x for x in inputs
                    ]

                    ref_output = ref_callable(*trial_data_ref)
                    torch.cuda.synchronize()

                    impl_output = impl_callable(*trial_data_impl)
                    torch.cuda.synchronize()

                    max_abs_diff = CorrectnessChecker.max_absolute_diff(ref_output, impl_output)
                    max_rel_diff = CorrectnessChecker.max_relative_diff(ref_output, impl_output)

                    shape_valid, shape_error = CorrectnessChecker.validate_shapes(
                        ref_output, impl_output
                    )
                    if not shape_valid:
                        trial_result["status"] = "shape_mismatch"
                        trial_result["error"] = shape_error
                        metadata["trial_results"].append(trial_result)
                        seed_manager.seed = original_seed
                        seed_manager.reset_seed()
                        return False, float("inf"), float("inf"), metadata

                    is_trial_correct = max_abs_diff <= config.max_diff_limit

                    max_diffs.append(max_abs_diff)
                    rel_diffs.append(max_rel_diff)

                    if is_trial_correct:
                        pass_count += 1
                        trial_result["status"] = "pass"
                    else:
                        trial_result["status"] = "fail"

                    trial_result["max_abs_diff"] = max_abs_diff
                    trial_result["max_rel_diff"] = max_rel_diff
                    metadata["trial_results"].append(trial_result)

                except Exception as e:
                    trial_result["status"] = "runtime_error"
                    trial_result["error"] = str(e)
                    metadata["trial_results"].append(trial_result)
                    max_diffs.append(float("inf"))
                    rel_diffs.append(float("inf"))

        seed_manager.seed = original_seed
        seed_manager.reset_seed()

        overall_max_abs_diff = max(max_diffs) if max_diffs else float("inf")
        overall_max_rel_diff = max(rel_diffs) if rel_diffs else float("inf")
        is_correct = pass_count == num_trials and overall_max_abs_diff <= config.max_diff_limit

        return is_correct, overall_max_abs_diff, overall_max_rel_diff, metadata

    def run(self, config: BenchmarkConfig = BenchmarkConfig()) -> TraceSet:
        """Orchestrate the entire end-to-end benchmark process

        Returns:
            TraceSet containing all benchmark traces
        """
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))

        all_traces = []

        # Phase 1: Build - Collect all workloads to evaluate and build callables
        self.logger.info("Phase 1: Building reference implementations...")

        for def_name in self.definitions:
            try:
                self._build_reference(def_name)
                self.logger.info(f"Built reference for {def_name}")
            except Exception as e:
                self.logger.error(f"Failed to build reference for {def_name}: {e}")
                # Skip all solutions for this definition
                if def_name in self.solutions:
                    del self.solutions[def_name]

        self.logger.info("Building solution implementations...")
        failed_solutions = []

        for def_name, solution_list in self.solutions.items():
            for solution in solution_list:
                try:
                    self._build_solution(solution)
                    self.logger.info(f"Built solution {solution.name}")
                except Exception as e:
                    self.logger.error(f"Failed to build solution {solution.name}: {e}")
                    failed_solutions.append(solution.name)

        # Remove failed solutions
        for def_name, solution_list in self.solutions.items():
            self.solutions[def_name] = [s for s in solution_list if s.name not in failed_solutions]

        # Phase 2: Run - Execute benchmarks for all valid workloads
        self.logger.info("Phase 2: Running benchmarks...")

        seed_manager = SeedManager()
        device_manager = DeviceManager(config.device)

        for def_name, solution_list in self.solutions.items():
            if def_name not in self._reference_callables:
                continue

            ref_callable = self._reference_callables[def_name]

            workload_list = self.workloads.get(def_name, [])

            # If no workloads, create a default one
            if not workload_list:
                definition = self.definitions[def_name]
                default_workload = {"axes": {}, "inputs": {}}

                for axis_name, axis_def in definition.axes.items():
                    if axis_def["type"] == "var":
                        default_workload["axes"][axis_name] = 32

                for input_name in definition.inputs:
                    default_workload["inputs"][input_name] = {"type": "random"}

                workload_list = [default_workload]

            for solution in solution_list:
                if solution.name not in self._solution_callables.get(def_name, {}):
                    continue

                impl_callable = self._solution_callables[def_name][solution.name]

                self.logger.info(f"Evaluating {solution.name} on {len(workload_list)} workloads")

                for workload in workload_list:
                    try:
                        trace = self._evaluate_single_workload(
                            solution,
                            workload,
                            ref_callable,
                            impl_callable,
                            config,
                            seed_manager,
                            device_manager,
                        )
                        all_traces.append(trace)
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate {solution.name} on workload: {e}")
                        error_trace = Trace(
                            definition=def_name,
                            solution=solution.name,
                            workload=workload,
                            evaluation={
                                "status": "EVALUATION_ERROR",
                                "log_file": f"{solution.name}_error.log",
                                "correctness": {
                                    "max_relative_error": float("inf"),
                                    "max_absolute_error": float("inf"),
                                },
                                "performance": {
                                    "latency_ms": float("inf"),
                                    "reference_latency_ms": float("inf"),
                                    "speedup_factor": 0.0,
                                },
                                "environment": {"error": str(e)},
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        all_traces.append(error_trace)

        # Phase 3: Aggregate - Create and return TraceSet
        self.logger.info(f"Phase 3: Benchmark completed with {len(all_traces)} traces")

        traces_by_definition = {}
        workload_by_definition = {}
        
        for trace in all_traces:
            if trace.is_workload():
                if trace.definition not in workload_by_definition:
                    workload_by_definition[trace.definition] = []
                workload_by_definition[trace.definition].append(trace)
            else:
                if trace.definition not in traces_by_definition:
                    traces_by_definition[trace.definition] = []
                traces_by_definition[trace.definition].append(trace)

        trace_set = TraceSet(
            definitions=self.definitions,
            solutions=self.solutions,
            workload=workload_by_definition,
            traces=traces_by_definition,
        )

        self._cleanup_temp_files()

        return trace_set
