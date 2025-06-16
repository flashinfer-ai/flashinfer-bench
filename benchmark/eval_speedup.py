import os
import sys
import json
import torch
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import re
from dataclasses import dataclass

from utils import (
    load_original_model_and_inputs,
    load_custom_model,
    set_seed,
    graceful_eval_cleanup,
    check_metadata_serializable_all_types
)

@dataclass
class SpeedupResult:
    """Result of speedup evaluation"""
    problem_name: str
    compiled: bool = False
    profiled_successfully: bool = False
    reference_kernel_time: Optional[float] = None  # microseconds
    custom_kernel_time: Optional[float] = None     # microseconds
    speedup: Optional[float] = None
    reference_total_time: Optional[float] = None
    custom_total_time: Optional[float] = None
    error: Optional[str] = None
    nsys_reference_file: Optional[str] = None
    nsys_custom_file: Optional[str] = None
    reference_nvtx_info: Dict = None 
    custom_nvtx_info: Dict = None 
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_name": self.problem_name,
            "compiled": self.compiled,
            "profiled_successfully": self.profiled_successfully,
            "reference_kernel_time_us": self.reference_kernel_time,
            "custom_kernel_time_us": self.custom_kernel_time,
            "speedup": self.speedup,
            "reference_total_time_ms": self.reference_total_time,
            "custom_total_time_ms": self.custom_total_time,
            "error": self.error,
            "nsys_reference_file": self.nsys_reference_file,
            "nsys_custom_file": self.nsys_custom_file,
            "reference_nvtx_info": self.reference_nvtx_info or {},
            "custom_nvtx_info": self.custom_nvtx_info or {},
            "metadata": self.metadata or {}
        }

class NSysProfiler:
    """Helper class for NSys profiling operations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def profile_model(self, model_name: str, model_func, *args, warmup_runs: int = 3, 
                     profile_runs: int = 10) -> Tuple[str, bool]:
        """Profile a model using NSys and return the output file path"""
        
        output_file = self.output_dir / f"{model_name}_profile.nsys-rep"
        
        # temporary script to run the model
        script_content = f"""
import torch
import sys
import os

# Warmup runs
print("Starting warmup runs...")
for i in range({warmup_runs}):
    result = model_func(*args)
    torch.cuda.synchronize()
    print(f"Warmup {{i+1}}/{warmup_runs} completed")

print("Starting profiled runs...")
# Profiled runs
for i in range({profile_runs}):
    result = model_func(*args)
    torch.cuda.synchronize()
    print(f"Profile run {{i+1}}/{profile_runs} completed")

print("Profiling completed")
"""
        
        script_file = self.output_dir / f"{model_name}_profile_script.py"
        
        return str(output_file), False
    
    def run_nsys_command(self, script_file: str, output_file: str, 
                        warmup_runs: int = 3, profile_runs: int = 10) -> bool:
        """Run NSys profiling command"""
        
        nsys_cmd = [
            "nsys", "profile",
            "--output", str(output_file),
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--duration", "30",  # Max 30 seconds
            "--sample", "none",  # Disable CPU sampling for faster profiling
            "--capture-range", "cudaProfilerApi",  # Use CUDA profiler API
            "--stop-on-range-end", "true",  # Stop when profiling range ends
            "python", str(script_file)
        ]
        
        try:
            print(f"Running NSys command: {' '.join(nsys_cmd)}")
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"NSys profiling successful, output: {output_file}")
                return True
            else:
                print(f"NSys profiling failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("NSys profiling timed out")
            return False
        except Exception as e:
            print(f"Error running NSys: {e}")
            return False
    
    def extract_nvtx_info(self, nsys_file: str) -> Dict:
        """Extract NVTX ranges and timing information from NSys report"""
        if not Path(nsys_file).exists():
            return {}
            
        try:
            nvtx_cmd = [
                "nsys", "stats", 
                "--report", "nvtx",
                "--format", "csv",
                str(nsys_file)
            ]
            
            result = subprocess.run(nvtx_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Failed to extract NVTX stats from {nsys_file}: {result.stderr}")
                return {}
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return {}
            
            header = lines[0].split(',')
            try:
                duration_idx = header.index('"Duration (nsec)"')
                name_idx = header.index('"Name"')
            except ValueError:
                print("Could not find required columns in NVTX output")
                return {}
            
            nvtx_info = {
                "warmup_phase_time": None,
                "profiling_phase_time": None,
                "warmup_runs": [],
                "profile_runs": [],
                "model_forward_passes": [],
                "total_nvtx_ranges": 0
            }
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                fields = line.split(',')
                if len(fields) <= max(duration_idx, name_idx):
                    continue
                
                try:
                    duration_ns = float(fields[duration_idx].strip('"'))
                    range_name = fields[name_idx].strip('"')
                    
                    nvtx_info["total_nvtx_ranges"] += 1
                    
                    if range_name == "Warmup Phase":
                        nvtx_info["warmup_phase_time"] = duration_ns / 1000000 
                    elif range_name == "Profiling Phase":
                        nvtx_info["profiling_phase_time"] = duration_ns / 1000000 #ms
                    elif range_name.startswith("Warmup Run"):
                        nvtx_info["warmup_runs"].append({
                            "run": range_name,
                            "duration_ms": duration_ns / 1000000
                        })
                    elif range_name.startswith("Profile Run"):
                        nvtx_info["profile_runs"].append({
                            "run": range_name,
                            "duration_ms": duration_ns / 1000000
                        })
                    elif range_name == "Model Forward Pass":
                        nvtx_info["model_forward_passes"].append({
                            "duration_ms": duration_ns / 1000000
                        })
                        
                except (ValueError, IndexError):
                    continue
            
            if nvtx_info["profile_runs"]:
                profile_times = [run["duration_ms"] for run in nvtx_info["profile_runs"]]
                nvtx_info["profile_run_stats"] = {
                    "mean_ms": sum(profile_times) / len(profile_times),
                    "min_ms": min(profile_times),
                    "max_ms": max(profile_times),
                    "count": len(profile_times)
                }
            
            if nvtx_info["model_forward_passes"]:
                forward_times = [fp["duration_ms"] for fp in nvtx_info["model_forward_passes"]]
                nvtx_info["forward_pass_stats"] = {
                    "mean_ms": sum(forward_times) / len(forward_times),
                    "min_ms": min(forward_times),
                    "max_ms": max(forward_times),
                    "count": len(forward_times)
                }
            
            return nvtx_info
                
        except subprocess.TimeoutExpired:
            print(f"Timeout extracting NVTX stats from {nsys_file}")
            return {}
        except Exception as e:
            print(f"Error extracting NVTX info: {e}")
            return {}

    def extract_kernel_times(self, nsys_file: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract kernel execution times from NSys report
        
        Returns:
            (kernel_time_us, total_time_ms): Tuple of kernel time and total time
        """
        if not Path(nsys_file).exists():
            return None, None
            
        try:
            stats_cmd = [
                "nsys", "stats", 
                "--report", "gputrace",
                "--format", "csv",
                str(nsys_file)
            ]
            
            result = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Failed to extract stats from {nsys_file}: {result.stderr}")
                return None, None
            
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return None, None
            
            header = lines[0].split(',')
            try:
                duration_idx = header.index('"Duration (nsec)"')
                name_idx = header.index('"Name"')
            except ValueError:
                print("Could not find required columns in NSys output")
                return None, None
            
            total_kernel_time_ns = 0
            kernel_count = 0
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                fields = line.split(',')
                if len(fields) <= max(duration_idx, name_idx):
                    continue
                
                try:
                    duration_ns = float(fields[duration_idx].strip('"'))
                    kernel_name = fields[name_idx].strip('"')
                    
                    # Filter for actual CUDA kernels (exclude memory operations)
                    if not any(x in kernel_name.lower() for x in ['memcpy', 'memset', 'memory']):
                        total_kernel_time_ns += duration_ns
                        kernel_count += 1
                        
                except (ValueError, IndexError):
                    continue
            
            if kernel_count == 0:
                return None, None
            
            kernel_time_us = total_kernel_time_ns / 1000
            
            summary_cmd = [
                "nsys", "stats", 
                "--report", "gpusum",
                "--format", "csv",
                str(nsys_file)
            ]
            
            summary_result = subprocess.run(summary_cmd, capture_output=True, text=True, timeout=30)
            total_time_ms = None
            
            if summary_result.returncode == 0:
                summary_lines = summary_result.stdout.strip().split('\n')
                if len(summary_lines) >= 2:
                    try:
                        for line in summary_lines[1:]:
                            if line.strip():
                                fields = line.split(',')
                                if len(fields) > 1:
                                    duration_field = fields[1].strip('"')
                                    if duration_field.replace('.', '').isdigit():
                                        total_time_ms = float(duration_field) / 1000000
                                        break
                    except:
                        pass
            
            return kernel_time_us, total_time_ms
            
        except subprocess.TimeoutExpired:
            print(f"Timeout extracting stats from {nsys_file}")
            return None, None
        except Exception as e:
            print(f"Error extracting kernel times: {e}")
            return None, None

class SpeedupEvaluator:
    def __init__(self, level1_dir: str = "level1", solutions_dir: str = "kernel_agent_solutions", 
                 results_dir: str = "speedup_results"):
        self.level1_dir = Path(level1_dir)
        self.solutions_dir = Path(solutions_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.nsys_dir = self.results_dir / "nsys_profiles"
        self.profiler = NSysProfiler(str(self.nsys_dir))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for kernel evaluation")
        
        print(f"Using device: {self.device} ({torch.cuda.get_device_name(self.device)})")
        
    def load_problem_files(self) -> Dict[str, Tuple[str, str]]:
        problems = {}
        
        for ref_file in self.level1_dir.glob("*.py"):
            problem_name = ref_file.stem
            solution_file = self.solutions_dir / f"{problem_name}_solution.py"
            
            if not solution_file.exists():
                print(f"Warning: No solution found for {problem_name}")
                continue
                
            with open(ref_file, 'r', encoding='utf-8') as f:
                reference_code = f.read()
            
            with open(solution_file, 'r', encoding='utf-8') as f:
                solution_code = f.read()
            
            problems[problem_name] = (reference_code, solution_code)
            
        print(f"Found {len(problems)} problem-solution pairs")
        return problems
    
    def create_profile_script(self, script_path: str, reference_code: str, solution_code: str,
                        is_custom: bool = False, problem_name: str = "unknown",
                        warmup_runs: int = 3, profile_runs: int = 10) -> None:        
        script_path_obj = Path(script_path)
        ref_code_file = script_path_obj.parent / f"{script_path_obj.stem}_ref_code.py"
        with open(ref_code_file, 'w') as f:
            f.write(reference_code)
        
        custom_code_file = None
        if is_custom:
            custom_code_file = script_path_obj.parent / f"{script_path_obj.stem}_custom_code.py"
            with open(custom_code_file, 'w') as f:
                f.write(solution_code)
        
        ref_code_file_str = str(ref_code_file.absolute())
        custom_code_file_str = str(custom_code_file.absolute()) if custom_code_file else ""
        
        script_content = f'''
import torch
import os
import sys
import tempfile
import importlib.util
import torch.cuda.nvtx as nvtx

# Set device
device = torch.device("cuda:0")
torch.cuda.set_device(device)

def load_model_from_file(file_path):
    """Load model from Python file"""
    spec = importlib.util.spec_from_file_location("model_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    Model = getattr(module, "Model", None)
    get_init_inputs = getattr(module, "get_init_inputs", None)
    get_inputs = getattr(module, "get_inputs", None)
    
    return Model, get_init_inputs, get_inputs

def load_custom_model_from_file(file_path, build_dir):
    """Load custom model from Python file"""
    # Set build directory environment
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    spec = importlib.util.spec_from_file_location("custom_model_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    ModelNew = getattr(module, "ModelNew", None)
    return ModelNew

# Load model
print("Loading model...")
ref_code_file = "{ref_code_file_str}"
Model, get_init_inputs, get_inputs = load_model_from_file(ref_code_file)

if Model is None or get_init_inputs is None or get_inputs is None:
    print("Error: Could not load required functions from reference code")
    sys.exit(1)

# Initialize model
print("Initializing model...")
with nvtx.range("Model Initialization"):
    torch.manual_seed(42)
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device) if isinstance(x, torch.Tensor) else x 
        for x in init_inputs
    ]

    with torch.no_grad():
        torch.manual_seed(42)
        if {is_custom}:
            # Load custom model
            with nvtx.range("Custom Model Loading"):
                build_dir = tempfile.mkdtemp()
                custom_code_file = "{custom_code_file_str}"
                ModelNew = load_custom_model_from_file(custom_code_file, build_dir)
                if ModelNew is None:
                    print("Error: Could not load custom model")
                    sys.exit(1)
                model = ModelNew(*init_inputs)
        else:
            # Load reference model
            with nvtx.range("Reference Model Loading"):
                model = Model(*init_inputs)
        
        model = model.cuda(device)

print("Preparing inputs...")
with nvtx.range("Input Preparation"):
    torch.manual_seed(42)
    inputs = get_inputs()
    inputs = [
        x.cuda(device) if isinstance(x, torch.Tensor) else x 
        for x in inputs
    ]

problem_name = "{problem_name}"
model_type = "custom" if {is_custom} else "reference"

nvtx.mark(f"Problem: {{problem_name}}")
nvtx.mark(f"Model Type: {{model_type}}")
nvtx.mark(f"Device: {{device}}")

print("Starting warmup runs...")
# Warmup phase with NVTX markers
with nvtx.range("Warmup Phase"):
    for i in range({warmup_runs}):
        with nvtx.range(f"Warmup Run {{i+1}}"):
            with torch.no_grad():
                with nvtx.range("Model Forward Pass"):
                    result = model(*inputs)
                torch.cuda.synchronize(device)
        print(f"Warmup {{i+1}}/{warmup_runs} completed")

print("Starting profiled runs...")
# Profiled runs with NVTX markers
with nvtx.range("Profiling Phase"):
    for i in range({profile_runs}):
        with nvtx.range(f"Profile Run {{i+1}}"):
            with torch.no_grad():
                with nvtx.range("Model Forward Pass"):
                    result = model(*inputs)
                torch.cuda.synchronize(device)
        print(f"Profile run {{i+1}}/{profile_runs} completed")

nvtx.mark("Profiling Completed")
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
    def evaluate_single_speedup(self, problem_name: str, reference_code: str, 
                               solution_code: str) -> SpeedupResult:
        print(f"\n{'='*60}")
        print(f"Evaluating speedup: {problem_name}")
        print(f"{'='*60}")
        
        result = SpeedupResult(problem_name=problem_name)
        
        try:
            print("Verifying custom kernel compilation...")
            with tempfile.TemporaryDirectory() as temp_dir:
                build_dir = Path(temp_dir) / "build"
                build_dir.mkdir()
                
                context = {}
                try:
                    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                        reference_code, context
                    )
                    
                    if Model is None:
                        result.error = "Failed to load reference model"
                        return result
                    
                    os.environ["TORCH_USE_CUDA_DSA"] = "1"
                    ModelNew = load_custom_model(solution_code, context, str(build_dir))
                    
                    if ModelNew is None:
                        result.error = "Failed to compile custom kernel"
                        return result
                    
                    result.compiled = True
                    print("Custom kernel compiled successfully")
                    
                except Exception as e:
                    result.error = f"Compilation error: {e}"
                    return result
                        
            print("Profiling reference PyTorch model...")
            ref_script = self.nsys_dir / f"{problem_name}_reference_profile.py"
            self.create_profile_script(
                str(ref_script), reference_code, solution_code, is_custom=False,
                problem_name=problem_name
            )
            
            ref_nsys_file = self.nsys_dir / f"{problem_name}_reference.nsys-rep"
            ref_success = self.profiler.run_nsys_command(
                str(ref_script), str(ref_nsys_file)
            )
            
            if ref_success:
                result.nsys_reference_file = str(ref_nsys_file)
                ref_kernel_time, ref_total_time = self.profiler.extract_kernel_times(str(ref_nsys_file))
                result.reference_kernel_time = ref_kernel_time
                result.reference_total_time = ref_total_time
                
                result.reference_nvtx_info = self.profiler.extract_nvtx_info(str(ref_nsys_file))
                print(f"Reference profiling successful - kernel time: {ref_kernel_time}μs")
                
                if result.reference_nvtx_info.get("profile_run_stats"):
                    stats = result.reference_nvtx_info["profile_run_stats"]
                    print(f"  - Profile runs: {stats['count']}, avg: {stats['mean_ms']:.2f}ms, range: {stats['min_ms']:.2f}-{stats['max_ms']:.2f}ms")
                if result.reference_nvtx_info.get("forward_pass_stats"):
                    stats = result.reference_nvtx_info["forward_pass_stats"]
                    print(f"  - Forward passes: {stats['count']}, avg: {stats['mean_ms']:.2f}ms")
            else:
                result.error = "Failed to profile reference model"
                return result
            
            print("Profiling custom CUDA kernel...")
            custom_script = self.nsys_dir / f"{problem_name}_custom_profile.py"
            self.create_profile_script(
                str(custom_script), reference_code, solution_code, is_custom=True,
                problem_name=problem_name
            )
            
            custom_nsys_file = self.nsys_dir / f"{problem_name}_custom.nsys-rep"
            custom_success = self.profiler.run_nsys_command(
                str(custom_script), str(custom_nsys_file)
            )
            
            if custom_success:
                result.nsys_custom_file = str(custom_nsys_file)
                custom_kernel_time, custom_total_time = self.profiler.extract_kernel_times(str(custom_nsys_file))
                result.custom_kernel_time = custom_kernel_time
                result.custom_total_time = custom_total_time
                
                result.custom_nvtx_info = self.profiler.extract_nvtx_info(str(custom_nsys_file))
                print(f"Custom profiling successful - kernel time: {custom_kernel_time}μs")
                
                if result.custom_nvtx_info.get("profile_run_stats"):
                    stats = result.custom_nvtx_info["profile_run_stats"]
                    print(f"  - Profile runs: {stats['count']}, avg: {stats['mean_ms']:.2f}ms, range: {stats['min_ms']:.2f}-{stats['max_ms']:.2f}ms")
                if result.custom_nvtx_info.get("forward_pass_stats"):
                    stats = result.custom_nvtx_info["forward_pass_stats"]
                    print(f"  - Forward passes: {stats['count']}, avg: {stats['mean_ms']:.2f}ms")
                
                if (result.reference_kernel_time is not None and 
                    result.custom_kernel_time is not None and 
                    result.custom_kernel_time > 0):
                    result.speedup = result.reference_kernel_time / result.custom_kernel_time
                    print(f"Speedup: {result.speedup:.2f}x")
                    
                    if (result.reference_nvtx_info.get("profile_run_stats") and 
                        result.custom_nvtx_info.get("profile_run_stats")):
                        ref_nvtx_avg = result.reference_nvtx_info["profile_run_stats"]["mean_ms"]
                        custom_nvtx_avg = result.custom_nvtx_info["profile_run_stats"]["mean_ms"]
                        nvtx_speedup = ref_nvtx_avg / custom_nvtx_avg if custom_nvtx_avg > 0 else None
                        if nvtx_speedup:
                            print(f"NVTX-based speedup: {nvtx_speedup:.2f}x")
                else:
                    result.error = "Could not extract kernel times for speedup calculation"
                
                result.profiled_successfully = True
            else:
                result.error = "Failed to profile custom model"
                return result
                
        except Exception as e:
            result.error = f"Unexpected error: {e}"
            print(f"Error evaluating {problem_name}: {e}")
        
        return result
    
    def evaluate_all_speedups(self) -> Dict:
        """Evaluate speedups for all generated kernels"""
        problems = self.load_problem_files()
        
        if not problems:
            print("No problems found to evaluate!")
            return {}
        
        summary = {
            "total_problems": len(problems),
            "compiled": 0,
            "profiled_successfully": 0,
            "speedups_calculated": 0,
            "average_speedup": 0,
            "evaluation_time": 0,
            "speedup_stats": {
                "min": None,
                "max": None,
                "median": None,
                "geometric_mean": None
            }
        }
        
        results = []
        speedups = []
        start_time = time.time()
        
        for problem_name, (reference_code, solution_code) in problems.items():
            result = self.evaluate_single_speedup(problem_name, reference_code, solution_code)
            results.append(result.to_dict())
            
            if result.compiled:
                summary["compiled"] += 1
            if result.profiled_successfully:
                summary["profiled_successfully"] += 1
            if result.speedup is not None:
                summary["speedups_calculated"] += 1
                speedups.append(result.speedup)
        
        summary["evaluation_time"] = time.time() - start_time
        
        if speedups:
            import statistics
            import math
            
            summary["average_speedup"] = statistics.mean(speedups)
            summary["speedup_stats"]["min"] = min(speedups)
            summary["speedup_stats"]["max"] = max(speedups)
            summary["speedup_stats"]["median"] = statistics.median(speedups)
            
            valid_speedups = [s for s in speedups if s > 0]
            if valid_speedups:
                log_sum = sum(math.log(s) for s in valid_speedups)
                summary["speedup_stats"]["geometric_mean"] = math.exp(log_sum / len(valid_speedups))
        
        if summary["total_problems"] > 0:
            summary["compilation_rate"] = summary["compiled"] / summary["total_problems"]
            summary["profiling_rate"] = summary["profiled_successfully"] / summary["total_problems"]
            summary["speedup_calculation_rate"] = summary["speedups_calculated"] / summary["total_problems"]
        
        detailed_results = {
            "summary": summary,
            "results": results,
            "metadata": {
                "device": str(self.device),
                "hardware": torch.cuda.get_device_name(self.device),
                "timestamp": time.time(),
                "nsys_profiles_dir": str(self.nsys_dir)
            }
        }
        
        detailed_results = check_metadata_serializable_all_types(detailed_results)
        
        results_file = self.results_dir / "speedup_evaluation.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SPEEDUP EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {summary['total_problems']}")
        print(f"Compiled successfully: {summary['compiled']} ({summary.get('compilation_rate', 0):.1%})")
        print(f"Profiled successfully: {summary['profiled_successfully']} ({summary.get('profiling_rate', 0):.1%})")
        print(f"Speedups calculated: {summary['speedups_calculated']} ({summary.get('speedup_calculation_rate', 0):.1%})")
        
        if speedups:
            print(f"\nKERNEL SPEEDUP ANALYSIS:")
            print(f"  Average speedup: {summary['average_speedup']:.2f}x")
            print(f"  Median speedup: {summary['speedup_stats']['median']:.2f}x")
            print(f"  Min speedup: {summary['speedup_stats']['min']:.2f}x")
            print(f"  Max speedup: {summary['speedup_stats']['max']:.2f}x")
            print(f"  Geometric mean: {summary['speedup_stats']['geometric_mean']:.2f}x")
        
        nvtx_successful = sum(1 for result in results if result.get('reference_nvtx_info', {}).get('total_nvtx_ranges', 0) > 0)
        if nvtx_successful > 0:
            print(f"\nNVTX PROFILING ANALYSIS:")
            print(f"  Problems with NVTX data: {nvtx_successful}/{summary['total_problems']}")
            
            # NVTX timing consistency
            nvtx_variances = []
            for result in results:
                ref_nvtx = result.get('reference_nvtx_info', {})
                custom_nvtx = result.get('custom_nvtx_info', {})
                
                if ref_nvtx.get('profile_run_stats') and custom_nvtx.get('profile_run_stats'):
                    ref_stats = ref_nvtx['profile_run_stats']
                    custom_stats = custom_nvtx['profile_run_stats']
                    ref_variance = (ref_stats['max_ms'] - ref_stats['min_ms']) / ref_stats['mean_ms'] if ref_stats['mean_ms'] > 0 else 0
                    custom_variance = (custom_stats['max_ms'] - custom_stats['min_ms']) / custom_stats['mean_ms'] if custom_stats['mean_ms'] > 0 else 0
                    nvtx_variances.append((ref_variance, custom_variance))
            
            if nvtx_variances:
                avg_ref_variance = sum(v[0] for v in nvtx_variances) / len(nvtx_variances)
                avg_custom_variance = sum(v[1] for v in nvtx_variances) / len(nvtx_variances)
                print(f"  Average timing variance (ref): {avg_ref_variance:.1%}")
                print(f"  Average timing variance (custom): {avg_custom_variance:.1%}")
        
        print(f"\nFILES AND OUTPUTS:")
        print(f"  Evaluation time: {summary['evaluation_time']:.2f} seconds")
        print(f"  Results saved to: {results_file}")
        print(f"  NSys profiles saved to: {self.nsys_dir}")
        print(f"  Use 'nsys-ui' to view detailed profiling results")
        
        return detailed_results

def main():
    level1_dir = "level1"
    solutions_dir = "kernel_agent_solutions"
    results_dir = "speedup_results"
    
    evaluator = SpeedupEvaluator(
        level1_dir=level1_dir,
        solutions_dir=solutions_dir,
        results_dir=results_dir
    )
    
    evaluator.evaluate_all_speedups()

if __name__ == "__main__":
    main()