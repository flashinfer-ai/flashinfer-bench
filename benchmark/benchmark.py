import argparse
import json
import os
import sys
import time
import asyncio
import importlib.util
import types
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import AsyncExitStack
except ImportError:
    print("Warning: MCP not available, falling back to subprocess")
    ClientSession = None

from kernel_loaders import create_kernel_loader


@dataclass
class KernelExecResult:
    """single kernel execution result"""
    round: int
    compiled: bool = False
    correctness: bool = False
    max_diff: float = float('inf')
    avg_diff: float = float('inf')
    runtime: Optional[float] = None
    speedup: Optional[float] = None
    generated_code: str = ""
    ncu_results: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)    
    compilation_error: Optional[Exception] = None
    runtime_error: Optional[Exception] = None
    generation_error: Optional[Exception] = None


@dataclass
class BenchmarkResults:
    """results across multiple rounds"""
    device_info: Dict
    kernel_spec: Dict
    baseline_time: float
    generations: List[KernelExecResult] = field(default_factory=list)
    
    success_rate: float = 0.0
    avg_speedup: float = 0.0
    max_speedup: float = 0.0
    compilation_success_rate: float = 0.0
    correctness_success_rate: float = 0.0
    
    def compute_statistics(self):
        if not self.generations:
            return
        
        total_rounds = len(self.generations)
        compiled_count = sum(1 for g in self.generations if g.compiled)
        correct_count = sum(1 for g in self.generations if g.correctness)
        
        self.compilation_success_rate = compiled_count / total_rounds if total_rounds > 0 else 0.0
        self.correctness_success_rate = correct_count / total_rounds if total_rounds > 0 else 0.0
        self.success_rate = self.correctness_success_rate
        
        successful_speedups = [g.speedup for g in self.generations if g.speedup is not None]
        if successful_speedups:
            self.avg_speedup = np.mean(successful_speedups)
            self.max_speedup = np.max(successful_speedups)
        
    def to_dict(self) -> Dict:
        return {
            "device_info": self.device_info,
            "kernel_spec": self.kernel_spec,
            "baseline_time": self.baseline_time,
            "success_rate": self.success_rate,
            "avg_speedup": self.avg_speedup,
            "max_speedup": self.max_speedup,
            "compilation_success_rate": self.compilation_success_rate,
            "correctness_success_rate": self.correctness_success_rate,
            "generations": [
                {
                    "round": g.round,
                    "compiled": g.compiled,
                    "correctness": g.correctness,
                    "max_diff": g.max_diff,
                    "avg_diff": g.avg_diff,
                    "runtime": g.runtime,
                    "speedup": g.speedup,
                    "generated_code": g.generated_code,
                    "ncu_results": g.ncu_results,
                    "metadata": g.metadata,
                    "compilation_error": str(g.compilation_error) if g.compilation_error else None,
                    "runtime_error": str(g.runtime_error) if g.runtime_error else None,
                    "generation_error": str(g.generation_error) if g.generation_error else None,
                }
                for g in self.generations
            ]
        }


class SeedManager:
    """for reproducibility"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset_seed()
    
    def reset_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


class DeviceManager:
    def __init__(self, device: Union[int, torch.device, str] = None, backend: str = "cuda"):
        self.backend = backend
        self.is_triton = backend == "triton"
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        if device is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        if self.device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid device id: {self.device.index}. Only {torch.cuda.device_count()} devices available.")
        
        torch.cuda.set_device(self.device)
        
        if self.is_triton:
            if device is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    def get_device_info(self) -> Dict:
        device_id = self.device.index
        return {
            "device_name": torch.cuda.get_device_name(device_id),
            "device_id": device_id,
            "device_str": str(self.device),
            "compute_capability": torch.cuda.get_device_capability(device_id),
            "total_memory": torch.cuda.get_device_properties(device_id).total_memory,
            "driver_version": torch.version.cuda,
            "backend": self.backend,
        }


def get_error_name(error: Exception) -> str:
    return type(error).__name__


class CorrectnessChecker:    
    @staticmethod
    def max_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            max_diffs = []
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    max_diffs.append(torch.max(torch.abs(o1 - o2)).item())
            return max(max_diffs) if max_diffs else float('inf')
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.max(torch.abs(output1 - output2)).item()
        else:
            return float('inf')
    
    @staticmethod
    def avg_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            avg_diffs = []
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    avg_diffs.append(torch.mean(torch.abs(o1 - o2)).item())
            return np.mean(avg_diffs) if avg_diffs else float('inf')
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.mean(torch.abs(output1 - output2)).item()
        else:
            return float('inf')
    
    @staticmethod
    def validate_shapes(output1: torch.Tensor, output2: torch.Tensor) -> Tuple[bool, str]:
        if isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            if len(output1) != len(output2):
                return False, f"Output length mismatch: Expected {len(output1)}, got {len(output2)}"
            
            for i, (o1, o2) in enumerate(zip(output1, output2)):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    if o1.shape != o2.shape:
                        return False, f"Output[{i}] shape mismatch: Expected {o1.shape}, got {o2.shape}"
                elif type(o1) != type(o2):
                    return False, f"Output[{i}] type mismatch: Expected {type(o1)}, got {type(o2)}"
            return True, ""
            
        elif isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            if output1.shape != output2.shape:
                return False, f"Output shape mismatch: Expected {output1.shape}, got {output2.shape}"
            return True, ""
        elif type(output1) != type(output2):
            return False, f"Output type mismatch: Expected {type(output1)}, got {type(output2)}"
        else:
            return True, ""
    
    @staticmethod
    def check_correctness(output1: torch.Tensor, output2: torch.Tensor, max_diff_limit: float, 
                         validate_shapes: bool = True) -> Dict[str, Any]:
        result = {
            "correct": False,
            "max_diff": float('inf'),
            "avg_diff": float('inf'),
            "shape_valid": True,
            "shape_error": "",
            "error": None
        }
        
        try:
            # shape validation
            if validate_shapes:
                shape_valid, shape_error = CorrectnessChecker.validate_shapes(output1, output2)
                result["shape_valid"] = shape_valid
                result["shape_error"] = shape_error
                
                if not shape_valid:
                    result["error"] = shape_error
                    return result
            
            # differences
            max_diff = CorrectnessChecker.max_diff(output1, output2)
            avg_diff = CorrectnessChecker.avg_diff(output1, output2)
            
            result["max_diff"] = max_diff
            result["avg_diff"] = avg_diff
            
            result["correct"] = max_diff <= max_diff_limit
            
        except Exception as e:
            result["error"] = str(e)
            result["correct"] = False
        
        return result
    
    @staticmethod
    def is_correct(output1: torch.Tensor, output2: torch.Tensor, max_diff_limit: float) -> bool:
        """Simple correctness check - returns only boolean result"""
        result = CorrectnessChecker.check_correctness(output1, output2, max_diff_limit)
        return result["correct"]


class NCUProfiler:
    """Wrapper for ncu profiler mcp server"""
    
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.use_mcp = ClientSession is not None
        
    async def start_mcp(self):
        if self.use_mcp:
            try:
                server_params = StdioServerParameters(
                    command="python",
                    args=[os.path.join(os.path.dirname(__file__), "ncu.py")],
                    env=None
                )
                
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                
                await self.session.initialize()
                print("Connected to NCU profiler server")
                
            except Exception as e:
                print(f"Failed to connect to NCU server: {e}")
                self.session = None
    
    async def profile_kernel(self, code: str, inputs: List[Any], init_inputs: List[Any] = None) -> Dict:
        if self.use_mcp and self.session:
            try:
                result = await self.session.call_tool(
                    "ncu_profiler",
                    {
                        "code": code,
                        "inputs": inputs,
                        "init_inputs": init_inputs or []
                    }
                )
                return result.content[0].text if result.content else {}
            except Exception as e:
                print(f"MCP profiling failed: {e}")
                return {"status": "error", "error": str(e)}
        else:
            return {"status": "error", "error": "MCP not available"}
    
    async def close(self):
        if self.exit_stack:
            await self.exit_stack.aclose()


class BenchmarkRunner:    
    def __init__(self, args):
        self.args = args
        self.use_ncu = getattr(args, 'use_ncu', 'true').lower() in ['true', 'yes']
        self.profiler = NCUProfiler() if self.use_ncu else None
        
        self.seed_manager = SeedManager(getattr(args, 'seed', 42))
        self.device_manager = DeviceManager(
            getattr(args, 'device', None), 
            getattr(args, 'backend', 'cuda')
        )
        
        self.kernel_loader = create_kernel_loader(
            args.kernel_description, 
            getattr(args, 'loader_type', 'auto')
        )
        
    def _time_kernel(self, model: nn.Module, inputs: List[Any], warmup: int, iterations: int) -> float:
        """Time a kernel execution using CUDA events"""
        model.eval()
        
        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*inputs)
        
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # actual timing
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(*inputs)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        
        # return avg time in ms
        return elapsed_time_ms / iterations

    async def _load_kernel_generator(self) -> Any:
        spec = importlib.util.spec_from_file_location("kernel_generator", self.args.kernel_generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'generate_kernel'):
            raise ValueError(f"No 'generate_kernel' function found in {self.args.kernel_generator}")
        
        return module.generate_kernel
    
    def _run_correctness_trials(self, original_model, custom_model, get_inputs_fn, num_trials: int = 1) -> Tuple[bool, float, Dict]:
        max_diffs = []
        avg_diffs = []
        pass_count = 0
        metadata = {
            "correctness_trials_attempted": num_trials,
            "max_differences": [],
            "avg_differences": [],
            "trial_results": []
        }
        
        original_seed = self.seed_manager.seed
        
        self.seed_manager.reset_seed()
        correctness_trial_seeds = [
            torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)
        ]
        
        with torch.no_grad():
            for trial in range(num_trials):
                trial_seed = correctness_trial_seeds[trial]
                trial_result = {"trial": trial, "seed": trial_seed}
                
                try:
                    torch.manual_seed(trial_seed)
                    torch.cuda.manual_seed_all(trial_seed)
                    np.random.seed(trial_seed)
                    random.seed(trial_seed)
                    
                    trial_data = get_inputs_fn()
                    trial_data = [x.to(self.device_manager.device) if isinstance(x, torch.Tensor) else x 
                                 for x in trial_data]
                    
                    # clone inputs to prevent aliasing
                    trial_data_ref = [x.clone() if isinstance(x, torch.Tensor) else x for x in trial_data]
                    trial_data_custom = [x.clone() if isinstance(x, torch.Tensor) else x for x in trial_data]
                    
                    original_output = original_model(*trial_data_ref)
                    torch.cuda.synchronize()
                    
                    custom_output = custom_model(*trial_data_custom)
                    torch.cuda.synchronize()
                    
                    # clone outputs to prevent aliasing
                    original_output = original_output.clone() if isinstance(original_output, torch.Tensor) else [o.clone() if isinstance(o, torch.Tensor) else o for o in original_output] if isinstance(original_output, (list, tuple)) else original_output
                    custom_output = custom_output.clone() if isinstance(custom_output, torch.Tensor) else [o.clone() if isinstance(o, torch.Tensor) else o for o in custom_output] if isinstance(custom_output, (list, tuple)) else custom_output
                    
                    trial_result["original_output_type"] = str(type(original_output))
                    trial_result["custom_output_type"] = str(type(custom_output))
                    if isinstance(original_output, torch.Tensor):
                        trial_result["original_output_shape"] = str(original_output.shape)
                        trial_result["original_output_dtype"] = str(original_output.dtype)
                    if isinstance(custom_output, torch.Tensor):
                        trial_result["custom_output_shape"] = str(custom_output.shape)
                        trial_result["custom_output_dtype"] = str(custom_output.dtype)
                    
                    correctness_result = CorrectnessChecker.check_correctness(
                        original_output, custom_output, self.args.max_diff_limit
                    )
                    
                    if correctness_result.get("error"):
                        trial_result["correctness_error"] = correctness_result["error"]
                    
                    # if shape invalid return directly
                    if not correctness_result["shape_valid"]:
                        trial_result["status"] = "shape_mismatch"
                        trial_result["error"] = correctness_result["shape_error"]
                        metadata["trial_results"].append(trial_result)
                        
                        self.seed_manager.seed = original_seed
                        self.seed_manager.reset_seed()
                        return False, float('inf'), metadata
                    
                    max_diff = correctness_result["max_diff"]
                    avg_diff = correctness_result["avg_diff"]
                    is_trial_correct = correctness_result["correct"]
                    
                    max_diffs.append(max_diff)
                    avg_diffs.append(avg_diff)
                    metadata["max_differences"].append(f"{max_diff:.6f}")
                    metadata["avg_differences"].append(f"{avg_diff:.6f}")

                    if is_trial_correct:
                        pass_count += 1
                        trial_result["status"] = "pass"
                    else:
                        trial_result["status"] = "fail"
                    
                    trial_result["max_diff"] = max_diff
                    trial_result["avg_diff"] = avg_diff
                    metadata["trial_results"].append(trial_result)
                    
                except Exception as e:
                    trial_result["status"] = "runtime_error"
                    trial_result["error"] = str(e)
                    trial_result["error_type"] = get_error_name(e)
                    metadata["trial_results"].append(trial_result)
                    
                    max_diffs.append(float('inf'))
                    avg_diffs.append(float('inf'))
                    metadata["max_differences"].append("inf")
                    metadata["avg_differences"].append("inf")
        
        self.seed_manager.seed = original_seed
        self.seed_manager.reset_seed()
        
        overall_max_diff = max(max_diffs) if max_diffs else float('inf')
        overall_avg_diff = np.mean([d for d in avg_diffs if d != float('inf')]) if avg_diffs else float('inf')
        is_correct = pass_count == num_trials and overall_max_diff <= self.args.max_diff_limit
        
        metadata["pass_count"] = pass_count
        metadata["total_trials"] = len(max_diffs)
        metadata["correctness_summary"] = f"({pass_count} / {len(max_diffs)})"
        metadata["overall_max_diff"] = overall_max_diff
        metadata["overall_avg_diff"] = overall_avg_diff
        
        return is_correct, overall_max_diff, metadata

    async def run_benchmark(self) -> BenchmarkResults:
        """main benchmarking script that runs the agent on a single reference kernel, use run_benchmark script for whole dataset"""
        print("Loading kernel description...")
        code_string, Model_class, get_init_inputs_fn, get_inputs_fn = self.kernel_loader.load_original_model_and_inputs(self.args.kernel_description)
        
        print("Generating test data...")
        self.seed_manager.reset_seed()
        test_data = get_inputs_fn()
        test_data = [x.to(self.device_manager.device) if isinstance(x, torch.Tensor) else x 
                    for x in test_data]
        
        print("Getting initialization inputs...")
        self.seed_manager.reset_seed()
        init_inputs = get_init_inputs_fn() if get_init_inputs_fn else []
        init_inputs = [x.to(self.device_manager.device) if isinstance(x, torch.Tensor) else x 
                      for x in init_inputs]
        
        print("Running baseline reference...")
        self.seed_manager.reset_seed()
        baseline_model = Model_class(*init_inputs)
        baseline_model = baseline_model.to(self.device_manager.device)
        
        with torch.no_grad():
            baseline_output = baseline_model(*test_data)
        
        # Time baseline
        baseline_time = self._time_kernel(baseline_model, test_data, self.args.warmup, self.args.iter)
        print(f"Baseline time: {baseline_time:.3f}ms")
        
        print("Loading kernel generator...")
        generate_kernel = await self._load_kernel_generator()
        
        if self.use_ncu and self.profiler:
            await self.profiler.start_mcp()
        
        results = BenchmarkResults(
            device_info=self.device_manager.get_device_info(),
            kernel_spec={
                "description_file": self.args.kernel_description,
                "generator_file": self.args.kernel_generator,
                "warmup": self.args.warmup,
                "iterations": self.args.iter,
                "max_diff_limit": self.args.max_diff_limit,
                "report_n": self.args.report_n,
                "correctness_trials": getattr(self.args, 'correctness_trials', 1),
                "seed": getattr(self.args, 'seed', 42),
                "backend": self.device_manager.backend,
                "use_ncu": self.use_ncu,
            },
            baseline_time=baseline_time
        )
        
        # generation rounds
        for round_num in range(self.args.report_n):
            print(f"Running generation {round_num + 1}/{self.args.report_n}...")
            
            generation_result = KernelExecResult(round=round_num)
            
            try:
                generated_code = generate_kernel(code_string)
                generation_result.generated_code = generated_code
                
            except Exception as e:
                print(f"Round {round_num + 1}: Generation failed: {e}")
                generation_result.generation_error = e
                generation_result.metadata["generation_error_name"] = get_error_name(e)
                results.generations.append(generation_result)
                continue
            
            try:
                # agent generated kernl compilation
                torch.cuda.synchronize()
                generated_model_class = self.kernel_loader.load_generated_model(generated_code)
                
                self.seed_manager.reset_seed()
                generated_model = generated_model_class(*init_inputs)
                generated_model = generated_model.to(self.device_manager.device)
                generation_result.compiled = True
                
            except Exception as e:
                print(f"Round {round_num + 1}: Compilation failed: {e}")
                generation_result.compilation_error = e
                generation_result.metadata["compilation_error_name"] = get_error_name(e)
                
                if "lock" in str(e) or "No such file or directory" in str(e):
                    generation_result.metadata["lock_file_error"] = True
                
                results.generations.append(generation_result)
                continue
            
            try:
                torch.cuda.synchronize()
                
                is_correct, max_diff, metadata = self._run_correctness_trials(
                    baseline_model, generated_model, get_inputs_fn,
                    getattr(self.args, 'correctness_trials', 1)
                )
                
                generation_result.max_diff = max_diff
                generation_result.avg_diff = metadata.get("overall_avg_diff", float('inf'))
                generation_result.correctness = is_correct
                
                if is_correct:
                    kernel_time = self._time_kernel(generated_model, test_data, self.args.warmup, self.args.iter)
                    generation_result.runtime = kernel_time
                    generation_result.speedup = baseline_time / kernel_time
                    
                    # NCU profiling
                    if self.use_ncu and self.profiler:
                        test_data_cloned = [x.clone() if isinstance(x, torch.Tensor) else x for x in test_data]
                        ncu_result = await self.profiler.profile_kernel(
                            generated_code, test_data_cloned, init_inputs=init_inputs
                        )
                        generation_result.ncu_results = ncu_result
                    
                    print(f"Round {round_num + 1}: Speedup = {generation_result.speedup:.2f}x "
                          f"(time: {kernel_time:.3f}ms)")
                else:
                    print(f"Round {round_num + 1}: Correctness failed (max_diff = {max_diff:.2e})")
                
                generation_result.metadata.update(metadata)
                
            except Exception as e:
                print(f"Round {round_num + 1}: Runtime error: {e}")
                generation_result.runtime_error = e
                generation_result.metadata["runtime_error_name"] = get_error_name(e)
            
            results.generations.append(generation_result)
        
        results.compute_statistics()
        
        print(f"\nFinal Results:")
        print(f"  Compilation success rate: {results.compilation_success_rate*100:.1f}%")
        print(f"  Correctness success rate: {results.correctness_success_rate*100:.1f}%")
        if results.avg_speedup > 0:
            print(f"  Average speedup: {results.avg_speedup:.2f}x")
            print(f"  Best speedup: {results.max_speedup:.2f}x")
        else:
            print(f"  No successful generations out of {self.args.report_n} attempts")
        
        if self.profiler:
            await self.profiler.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA kernels using AI generation")
    parser.add_argument("kernel_description", help="Path to kernel description file (.py)")
    parser.add_argument("kernel_generator", help="Path to kernel generator file (.py)")
    
    # Basic benchmarking parameters
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Number of timing iterations per round")
    parser.add_argument("--report-n", type=int, default=16, help="Number of generation rounds")
    parser.add_argument("--max-diff-limit", type=float, default=1e-5, help="Maximum difference for correctness")
    
    # Correctness and reproducibility
    parser.add_argument("--correctness-trials", type=int, default=1, help="Number of correctness trials with different inputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Device and backend management
    parser.add_argument("--device", help="CUDA device to use (e.g., 'cuda:0', 0, or 'auto')")
    parser.add_argument("--backend", choices=["cuda", "triton"], default="cuda", help="Backend to use")
    parser.add_argument("--loader-type", choices=["auto", "kernelbench", "triton", "flashinfer"], default="auto", 
                        help="Type of kernel loader to use (auto-detects if not specified)")
    
    # Profiling
    parser.add_argument("--use-ncu", type=str, default="true", help="Enable NCU profiling (default: true)")
    
    # Output
    parser.add_argument("--output", default="result.json", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.kernel_description):
        print(f"Error: Kernel description file '{args.kernel_description}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.kernel_generator):
        print(f"Error: Kernel generator file '{args.kernel_generator}' not found")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    if args.loader_type == "triton" and args.backend == "cuda":
        print("Info: Auto-setting backend to 'triton' based on loader type")
        args.backend = "triton"
    
    print(f"Benchmark Configuration:")
    print(f"  Kernel description: {args.kernel_description}")
    print(f"  Kernel generator: {args.kernel_generator}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Backend: {args.backend}")
    print(f"  Loader type: {args.loader_type}")
    print(f"  Generation rounds: {args.report_n}")
    print(f"  Timing iterations: {args.iter}")
    print(f"  Correctness trials: {args.correctness_trials}")
    print(f"  Seed: {args.seed}")
    print(f"  Max diff limit: {args.max_diff_limit}")
    print(f"  Use NCU profiling: {args.use_ncu}")
    print()
    
    runner = BenchmarkRunner(args)
    
    async def run():
        try:
            results = await runner.run_benchmark()
            
            with open(args.output, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
            
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nBenchmark failed with error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        finally:
            # cleanup tmp files for triton backend
            if hasattr(runner.kernel_loader, 'cleanup'):
                runner.kernel_loader.cleanup()
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
