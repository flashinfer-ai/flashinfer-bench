import os
import json
import shutil
import torch
import tempfile
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import time
import numpy as np
from dataclasses import dataclass

# Evaluation Functions (extracted from KernelBench)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_original_model_and_inputs(model_src: str, context: dict) -> Tuple[Optional[type], Optional[Callable], Optional[Callable]]:
    """Load PyTorch model and input functions from source code"""
    try:
        compile(model_src, "<string>", "exec")
        exec(model_src, context)
        
        get_init_inputs_fn = context.get("get_init_inputs")
        get_inputs_fn = context.get("get_inputs")
        Model = context.get("Model")
        
        return (Model, get_init_inputs_fn, get_inputs_fn)
    except Exception as e:
        print(f"Error loading original model: {e}")
        return None, None, None

def load_custom_model(model_src: str, context: dict, build_directory: Optional[str] = None) -> Optional[type]:
    """Load custom CUDA kernel model from source code"""
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Set TORCH_EXTENSIONS_DIR to control where CUDA extensions are built
        model_src = "import os\n" + f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n" + model_src
    
    try:
        compile(model_src, "<string>", "exec")
        exec(model_src, context)
        ModelNew = context.get("ModelNew")
        return ModelNew
    except Exception as e:
        print(f"Error loading custom model: {e}")
        return None

@dataclass
class KernelExecResult:
    """Result of kernel execution and evaluation"""
    correctness: bool = False
    runtime: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

def time_execution_with_cuda_event(model, *inputs, num_trials: int = 100, 
                                 verbose: bool = False, device: torch.device = None) -> List[float]:
    """Time model execution using CUDA events"""
    if device is None:
        device = torch.device("cuda:0")
    
    elapsed_times = []
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize(device)
    
    # timing
    for i in range(num_trials):
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            _ = model(*inputs)
        end_event.record()
        
        torch.cuda.synchronize(device)
        elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
        elapsed_times.append(elapsed_time)
        
        if verbose and i % 10 == 0:
            print(f"Trial {i+1}/{num_trials}: {elapsed_time:.3f}ms")
    
    return elapsed_times

def get_timing_stats(elapsed_times: List[float], device: torch.device = None) -> Dict[str, float]:
    """Calculate timing statistics from elapsed times"""
    if not elapsed_times:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    return {
        "mean": float(np.mean(elapsed_times)),
        "std": float(np.std(elapsed_times)),
        "min": float(np.min(elapsed_times)),
        "max": float(np.max(elapsed_times)),
        "median": float(np.median(elapsed_times))
    }

def compare_outputs(output1, output2, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    """Compare two outputs for correctness"""
    try:
        if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return torch.allclose(output1, output2, rtol=rtol, atol=atol)
        elif isinstance(output1, (list, tuple)) and isinstance(output2, (list, tuple)):
            if len(output1) != len(output2):
                return False
            return all(compare_outputs(o1, o2, rtol, atol) for o1, o2 in zip(output1, output2))
        else:
            return torch.allclose(torch.tensor(output1), torch.tensor(output2), rtol=rtol, atol=atol)
    except Exception as e:
        print(f"Error comparing outputs: {e}")
        return False

def run_and_check_correctness(original_model, custom_model, get_inputs_fn,
                            metadata: Dict = None, num_correct_trials: int = 5,
                            verbose: bool = False, seed: int = 42,
                            device: torch.device = None) -> KernelExecResult:
    """Run correctness tests comparing original and custom models"""
    if device is None:
        device = torch.device("cuda:0")
    
    if metadata is None:
        metadata = {}
    
    result = KernelExecResult(metadata=metadata)
    
    passed_trials = 0
    
    try:
        for trial in range(num_correct_trials):
            set_seed(seed + trial)
            inputs = get_inputs_fn()
            inputs = [x.cuda(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Get reference output
            with torch.no_grad():
                set_seed(seed + trial)
                ref_output = original_model(*inputs)
            
            # Get custom output
            with torch.no_grad():
                set_seed(seed + trial)
                custom_output = custom_model(*inputs)
            
            # Compare outputs
            if compare_outputs(ref_output, custom_output):
                passed_trials += 1
                if verbose:
                    print(f"Trial {trial + 1}: PASSED")
            else:
                if verbose:
                    print(f"Trial {trial + 1}: FAILED")
        
        result.correctness = (passed_trials == num_correct_trials)
        result.metadata["correctness_trials"] = f"({passed_trials} / {num_correct_trials})"
        result.metadata["passed_trials"] = passed_trials
        result.metadata["total_trials"] = num_correct_trials
        
        if verbose:
            print(f"Correctness: {passed_trials}/{num_correct_trials} trials passed")
        
    except Exception as e:
        result.correctness = False
        result.metadata["error"] = str(e)
        if verbose:
            print(f"Correctness check failed: {e}")
    
    return result

def graceful_eval_cleanup(context: dict, device: torch.device):
    """Clean up after evaluation"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    except:
        pass

def check_metadata_serializable_all_types(obj) -> Any:
    """Ensure all metadata is JSON serializable"""
    if isinstance(obj, dict):
        return {k: check_metadata_serializable_all_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [check_metadata_serializable_all_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(check_metadata_serializable_all_types(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return str(obj)


class BenchmarkConfig:
    """Configuration class for benchmark settings"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Default configuration
        self.config = {
            "level1_dir": "level1",
            "solutions_dir": "kernel_agent_solutions", 
            "correctness_results_dir": "correctness_results",
            "speedup_results_dir": "speedup_results",
            "mcp_servers": ["../src/nsys.py", "../src/benchmark.py"],
            "max_iterations": 5,
            "num_correctness_test_cases": 5,
            "nsys_warmup_runs": 3,
            "nsys_profile_runs": 10,
            "cleanup_nsys_files": False,
            "verbose": True
        }
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        self.config.update(user_config)
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def get(self, key, default=None):
        return self.config.get(key, default)

class BenchmarkLogger:
    """Simple logging utility for benchmark operations"""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        self.start_time = time.time()
        
        if self.log_file:
            # Create log directory if needed
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize log file
            with open(self.log_file, 'w') as f:
                f.write(f"Benchmark started at {time.ctime()}\n")
                f.write("="*60 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.time() - self.start_time
        formatted_msg = f"[{timestamp:8.2f}s] {level}: {message}"
        
        if self.verbose:
            print(formatted_msg)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARN")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")


def validate_benchmark_setup(config: BenchmarkConfig, logger: BenchmarkLogger) -> bool:
    """Validate that the benchmark environment is properly set up"""
    
    logger.info("Validating benchmark setup...")
    
    level1_path = Path(config["level1_dir"])
    if not level1_path.exists():
        logger.error(f"Level 1 test cases directory not found: {level1_path}")
        return False
    
    py_files = list(level1_path.glob("*.py"))
    if not py_files:
        logger.error(f"No Python files found in {level1_path}")
        return False
    
    logger.info(f"Found {len(py_files)} test cases in {level1_path}")
    
    for server_path in config["mcp_servers"]:
        if not Path(server_path).exists():
            logger.error(f"MCP server not found: {server_path}")
            return False
    
    logger.info(f"Found {len(config['mcp_servers'])} MCP servers")
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available - required for kernel evaluation")
            return False
        
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available - using device: {device_name}")
        
    except ImportError:
        logger.error("PyTorch not installed - required for evaluation")
        return False
    
    try:
        import subprocess
        result = subprocess.run(["nsys", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            nsys_version = result.stdout.strip().split('\n')[0]
            logger.info(f"NSys available - {nsys_version}")
        else:
            logger.warning("NSys not found - speedup evaluation will not work")
            
    except FileNotFoundError:
        logger.warning("NSys not found in PATH - speedup evaluation will not work")
    
    logger.success("Benchmark setup validation completed")
    return True

def create_benchmark_directories(config: BenchmarkConfig, logger: BenchmarkLogger):
    """Create necessary directories for benchmark execution"""
    
    directories = [
        config["solutions_dir"],
        config["correctness_results_dir"], 
        config["speedup_results_dir"]
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {dir_path}")

def load_benchmark_results(results_dir: str, result_type: str = "both") -> Dict[str, Any]:
    """
    Args:
        results_dir: Directory containing result files
        result_type: "correctness", "speedup", or "both"
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    
    if result_type in ["correctness", "both"]:
        correctness_file = Path(results_dir) / "correctness_results" / "correctness_evaluation.json"
        if correctness_file.exists():
            with open(correctness_file, 'r') as f:
                results["correctness"] = json.load(f)
    
    if result_type in ["speedup", "both"]:
        speedup_file = Path(results_dir) / "speedup_results" / "speedup_evaluation.json"
        if speedup_file.exists():
            with open(speedup_file, 'r') as f:
                results["speedup"] = json.load(f)
    
    return results

def generate_benchmark_report(results: Dict[str, Any], output_file: str, 
                            logger: BenchmarkLogger) -> None:    
    report_lines = []
    report_lines.append("CUDA KERNEL BENCHMARK REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated at: {time.ctime()}")
    report_lines.append("")
    
    if "correctness" in results:
        correctness = results["correctness"]
        summary = correctness.get("summary", {})
        
        report_lines.append("CORRECTNESS EVALUATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Total problems: {summary.get('total_problems', 0)}")
        report_lines.append(f"Compiled successfully: {summary.get('compiled', 0)} ({summary.get('compilation_rate', 0):.1%})")
        report_lines.append(f"Passed correctness: {summary.get('correct', 0)} ({summary.get('correctness_rate', 0):.1%})")
        report_lines.append(f"Compilation errors: {summary.get('compilation_errors', 0)}")
        report_lines.append(f"Runtime errors: {summary.get('runtime_errors', 0)}")
        report_lines.append("")
        
        report_lines.append("Per-problem correctness:")
        for result in correctness.get("results", []):
            status = "✓" if result.get("correctness", False) else "✗"
            passed = result.get("passed_test_cases", 0)
            total = result.get("num_test_cases", 0)
            report_lines.append(f"  {status} {result['problem_name']}: {passed}/{total} tests passed")
        report_lines.append("")
    
    if "speedup" in results:
        speedup = results["speedup"]
        summary = speedup.get("summary", {})
        
        report_lines.append("SPEEDUP EVALUATION")
        report_lines.append("-" * 30)
        report_lines.append(f"Total problems: {summary.get('total_problems', 0)}")
        report_lines.append(f"Profiled successfully: {summary.get('profiled_successfully', 0)} ({summary.get('profiling_rate', 0):.1%})")
        report_lines.append(f"Speedups calculated: {summary.get('speedups_calculated', 0)} ({summary.get('speedup_calculation_rate', 0):.1%})")
        
        if summary.get("speedups_calculated", 0) > 0:
            stats = summary.get("speedup_stats", {})
            report_lines.append(f"Average speedup: {summary.get('average_speedup', 0):.2f}x")
            report_lines.append(f"Median speedup: {stats.get('median', 0):.2f}x")
            report_lines.append(f"Min speedup: {stats.get('min', 0):.2f}x")
            report_lines.append(f"Max speedup: {stats.get('max', 0):.2f}x")
            report_lines.append(f"Geometric mean: {stats.get('geometric_mean', 0):.2f}x")
        report_lines.append("")
        
        report_lines.append("Per-problem speedups:")
        for result in speedup.get("results", []):
            speedup_val = result.get("speedup")
            if speedup_val is not None:
                status = "🚀" if speedup_val > 1.0 else "🐌"
                report_lines.append(f"  {status} {result['problem_name']}: {speedup_val:.2f}x")
            else:
                report_lines.append(f"  ❌ {result['problem_name']}: Failed to measure")
        report_lines.append("")
    
    if "correctness" in results and "speedup" in results:
        correctness_summary = results["correctness"].get("summary", {})
        speedup_summary = results["speedup"].get("summary", {})
        
        report_lines.append("COMBINED ANALYSIS")
        report_lines.append("-" * 30)
        
        correct_problems = set()
        for result in results["correctness"].get("results", []):
            if result.get("correctness", False):
                correct_problems.add(result["problem_name"])
        
        fast_problems = set()
        for result in results["speedup"].get("results", []):
            if result.get("speedup", 0) > 1.0:
                fast_problems.add(result["problem_name"])
        
        correct_and_fast = correct_problems.intersection(fast_problems)
        
        report_lines.append(f"Correct kernels: {len(correct_problems)}")
        report_lines.append(f"Fast kernels (>1x speedup): {len(fast_problems)}")
        report_lines.append(f"Correct AND fast kernels: {len(correct_and_fast)}")
        
        if correct_and_fast:
            report_lines.append("Successful kernels (correct + fast):")
            for problem in sorted(correct_and_fast):
                speedup_val = None
                for result in results["speedup"].get("results", []):
                    if result["problem_name"] == problem:
                        speedup_val = result.get("speedup")
                        break
                
                if speedup_val:
                    report_lines.append(f"  ✨ {problem}: {speedup_val:.2f}x speedup")
        report_lines.append("")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Benchmark report saved to: {output_file}")
    
    if logger.verbose:
        print("\n" + "\n".join(report_lines))

def cleanup_intermediate_files(config: BenchmarkConfig, logger: BenchmarkLogger, 
                             keep_solutions: bool = True, keep_logs: bool = True):
    """Clean up intermediate files generated during benchmarking"""
    
    logger.info("Cleaning up intermediate files...")
    
    if not config.get("cleanup_nsys_files", False):
        nsys_dir = Path(config["speedup_results_dir"]) / "nsys_profiles"
        if nsys_dir.exists():
            # Keep .nsys-rep files but remove temporary scripts
            for script_file in nsys_dir.glob("*_profile.py"):
                script_file.unlink()
                logger.info(f"Removed: {script_file}")
    
    if not keep_solutions:
        solutions_dir = Path(config["solutions_dir"])
        if solutions_dir.exists():
            shutil.rmtree(solutions_dir)
            logger.info(f"Removed solutions directory: {solutions_dir}")
    
    if not keep_logs:
        logs_dir = Path(config["solutions_dir"]) / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
            logger.info(f"Removed logs directory: {logs_dir}")

def get_problem_statistics(level1_dir: str) -> Dict[str, Any]:
    """Get statistics about the test problems"""
    
    level1_path = Path(level1_dir)
    py_files = list(level1_path.glob("*.py"))
    
    stats = {
        "total_problems": len(py_files),
        "problem_names": [f.stem for f in py_files],
        "problem_files": [str(f) for f in py_files]
    }
    
    operation_counts = {
        "matmul": 0,
        "conv": 0,
        "norm": 0,
        "activation": 0,
        "pooling": 0,
        "other": 0
    }
    
    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read().lower()
            
            problem_name = py_file.stem.lower()
            
            if any(x in problem_name for x in ["matmul", "mm", "gemm"]):
                operation_counts["matmul"] += 1
            elif any(x in problem_name for x in ["conv", "convolution"]):
                operation_counts["conv"] += 1
            elif any(x in problem_name for x in ["norm", "batch", "layer", "instance"]):
                operation_counts["norm"] += 1
            elif any(x in problem_name for x in ["relu", "gelu", "selu", "tanh", "sigmoid", "softmax"]):
                operation_counts["activation"] += 1
            elif any(x in problem_name for x in ["pool", "pooling"]):
                operation_counts["pooling"] += 1
            else:
                operation_counts["other"] += 1
                
        except Exception:
            operation_counts["other"] += 1
    
    stats["operation_breakdown"] = operation_counts
    
    return stats

def create_default_config_file(config_file: str = "benchmark_config.json") -> str:
    """Create a default configuration file"""
    
    default_config = {
        "level1_dir": "level1",
        "solutions_dir": "kernel_agent_solutions",
        "correctness_results_dir": "correctness_results", 
        "speedup_results_dir": "speedup_results",
        "mcp_servers": [
            "../src/nsys.py",
            "../src/benchmark.py"
        ],
        "max_iterations": 5,
        "num_correctness_test_cases": 5,
        "nsys_warmup_runs": 3,
        "nsys_profile_runs": 10,
        "cleanup_nsys_files": False,
        "verbose": True
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    return config_file