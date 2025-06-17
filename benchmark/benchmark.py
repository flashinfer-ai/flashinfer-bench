import argparse
import json
import os
import sys
import time
import asyncio
import importlib.util
import types
from typing import List, Tuple, Any, Dict, Optional
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


class KernelLoader:    
    @staticmethod
    def load_from_file(filepath: str) -> Tuple[str, Any, Any]:
        """Returns tuple of (code_string, run_pytorch_func, gen_data_func)"""
        spec = importlib.util.spec_from_file_location("kernel_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        with open(filepath, 'r') as f:
            code_string = f.read()
        
        run_pytorch = getattr(module, 'run_pytorch', None)
        gen_data = getattr(module, 'gen_data', None)
        
        if run_pytorch is None:
            raise ValueError(f"No 'run_pytorch' function found in {filepath}")
        if gen_data is None:
            raise ValueError(f"No 'gen_data' function found in {filepath}")
            
        return code_string, run_pytorch, gen_data
    
    @staticmethod
    def load_from_string(code: str) -> Tuple[str, Any, Any]:
        """Returns tuple of (code_string, run_pytorch_func, gen_data_func)"""
        module = types.ModuleType("kernel_module")
        exec(code, module.__dict__)
        
        run_pytorch = getattr(module, 'run_pytorch', None)
        gen_data = getattr(module, 'gen_data', None)
        
        if run_pytorch is None:
            raise ValueError("No 'run_pytorch' function found in code")
        if gen_data is None:
            raise ValueError("No 'gen_data' function found in code")
            
        return code, run_pytorch, gen_data


class CorrectnessChecker:    
    @staticmethod
    def max_diff(output1: torch.Tensor, output2: torch.Tensor) -> float:
        """Calculate max diff between two tensors"""
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
    def is_correct(output1: torch.Tensor, output2: torch.Tensor, max_diff_limit: float) -> bool:
        return CorrectnessChecker.max_diff(output1, output2) <= max_diff_limit


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
        self.profiler = NCUProfiler()
        self.device_info = self._get_device_info()
        
    def _get_device_info(self) -> Dict:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return {
                "device_name": torch.cuda.get_device_name(device),
                "device_id": device,
                "compute_capability": torch.cuda.get_device_capability(device),
                "total_memory": torch.cuda.get_device_properties(device).total_memory,
                "driver_version": torch.version.cuda,
            }
        else:
            return {"error": "CUDA not available"}
    
    def _time_kernel(self, model: nn.Module, inputs: List[Any], warmup: int, iterations: int) -> float:
        """Time a kernel execution using CUDA events, TODO: add nsys mcp suppport"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(*inputs)
        
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Actual timing
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(*inputs)
        end_event.record()
        
        # Wait for events to complete and get elapsed time
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        
        # Return average time per iteration in milliseconds
        return elapsed_time_ms / iterations
    
    async def _load_kernel_generator(self) -> Any:
        spec = importlib.util.spec_from_file_location("kernel_generator", self.args.kernel_generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'generate_kernel'):
            raise ValueError(f"No 'generate_kernel' function found in {self.args.kernel_generator}")
        
        return module.generate_kernel
    
    async def run_benchmark(self) -> Dict:
        """Main benchmarking script"""        
        print("Loading kernel description...")
        if self.args.kernel_description.endswith('.py'):
            code_string, run_pytorch, gen_data = KernelLoader.load_from_file(self.args.kernel_description)
        else:
            code_string, run_pytorch, gen_data = KernelLoader.load_from_string(self.args.kernel_description)
        
        print("Generating test data...")
        test_data = gen_data()
        
        # Get baseline reference
        print("Running baseline reference!!!")
        baseline_output = run_pytorch(test_data)
        
        baseline_module = types.ModuleType("baseline_module")
        exec(code_string, baseline_module.__dict__)
        baseline_model = baseline_module.ModelNew()
        baseline_model = baseline_model.cuda()
        baseline_time = self._time_kernel(baseline_model, test_data, self.args.warmup, self.args.iter)
        
        print(f"Baseline time: {baseline_time:.6f}s")
        
        print("Loading kernel generator...")
        generate_kernel = await self._load_kernel_generator()
        
        # Start profiler
        await self.profiler.start_mcp()
        
        results = {
            "device": self.device_info,
            "kernel_spec": {
                "description_file": self.args.kernel_description,
                "generator_file": self.args.kernel_generator,
                "warmup": self.args.warmup,
                "iterations": self.args.iter,
                "max_diff_limit": self.args.max_diff_limit,
                "report_n": self.args.report_n
            },
            "baseline_time": baseline_time,
            "generations": [],
            "avg_of_n_speedup": 0.0,
            "max_of_n_speedup": 0.0
        }
        
        speedups = []
        
        for round_num in range(self.args.report_n):
            print(f"Running generation {round_num + 1}/{self.args.report_n}...")
            
            generation_result = {
                "round": round_num,
                "generated_code": "",
                "compiled": False,
                "max_diff": float('inf'),
                "time": float('inf'),
                "relative_time": float('inf'),
                "ncu_results": {}
            }
            
            try:
                generated_code = await generate_kernel(code_string)
                generation_result["generated_code"] = generated_code
                
                try:
                    # Execute generated code
                    generated_module = types.ModuleType("generated_module")
                    exec(generated_code, generated_module.__dict__)
                    
                    if hasattr(generated_module, 'ModelNew'):
                        generated_model = generated_module.ModelNew()
                        generated_model = generated_model.cuda()
                        generation_result["compiled"] = True
                        
                        #correctness check
                        with torch.no_grad():
                            generated_output = generated_model(*test_data)
                        
                        max_diff = CorrectnessChecker.max_diff(baseline_output, generated_output)
                        generation_result["max_diff"] = max_diff
                        
                        if CorrectnessChecker.is_correct(baseline_output, generated_output, self.args.max_diff_limit):
                            kernel_time = self._time_kernel(generated_model, test_data, self.args.warmup, self.args.iter)
                            generation_result["time"] = kernel_time
                            generation_result["relative_time"] = kernel_time / baseline_time
                            speedups.append(baseline_time / kernel_time)
                            
                            # NCU profiling
                            ncu_result = await self.profiler.profile_kernel(
                                generated_code, 
                                test_data, 
                                init_inputs=None
                            )
                            generation_result["ncu_results"] = ncu_result
                            
                            print(f"Round {round_num + 1}: Speedup = {baseline_time / kernel_time:.2f}x")
                        else:
                            print(f"Round {round_num + 1}: Correctness failed (max_diff = {max_diff:.2e})")
                    else:
                        print(f"Round {round_num + 1}: No ModelNew class found in generated code")
                        
                except Exception as e:
                    print(f"Round {round_num + 1}: Compilation/execution failed: {e}")
                    
            except Exception as e:
                print(f"Round {round_num + 1}: Generation failed: {e}")
                generation_result["generation_error"] = str(e)
            
            results["generations"].append(generation_result)
        
        if speedups:
            results["avg_of_n_speedup"] = np.mean(speedups)
            results["max_of_n_speedup"] = np.max(speedups)
            print(f"\nFinal Results:")
            print(f"  Average speedup: {results['avg_of_n_speedup']:.2f}x")
            print(f"  Best speedup: {results['max_of_n_speedup']:.2f}x")
            print(f"  Success rate: {len(speedups)}/{self.args.report_n} ({len(speedups)/self.args.report_n*100:.1f}%)")
        else:
            print(f"\nNo successful generations out of {self.args.report_n} attempts")
        
        await self.profiler.close()
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA kernels using AI generation")
    parser.add_argument("kernel_description", help="Path to kernel description file (.py)")
    parser.add_argument("kernel_generator", help="Path to kernel generator file (.py)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Number of timing iterations")
    parser.add_argument("--report-n", type=int, default=16, help="Number of generation rounds") # I combined the best_of_n and average_n into one, maybe could separete?
    parser.add_argument("--max-diff-limit", type=float, default=1e-5, help="Maximum difference for correctness")
    parser.add_argument("--output", default="result.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.kernel_description):
        print(f"Error: Kernel description file '{args.kernel_description}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.kernel_generator):
        print(f"Error: Kernel generator file '{args.kernel_generator}' not found")
        sys.exit(1)
    
    runner = BenchmarkRunner(args)
    
    async def run():
        results = await runner.run_benchmark()
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
