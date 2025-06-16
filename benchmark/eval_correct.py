import os
import sys
import json
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

from utils import (
    load_original_model_and_inputs,
    load_custom_model,
    run_and_check_correctness,
    KernelExecResult,
    set_seed,
    graceful_eval_cleanup,
    check_metadata_serializable_all_types
)

@dataclass
class CorrectnessResult:
    problem_name: str
    compiled: bool = False
    correctness: bool = False
    compilation_error: Optional[str] = None
    runtime_error: Optional[str] = None
    num_test_cases: int = 0
    passed_test_cases: int = 0
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_name": self.problem_name,
            "compiled": self.compiled,
            "correctness": self.correctness,
            "compilation_error": self.compilation_error,
            "runtime_error": self.runtime_error,
            "num_test_cases": self.num_test_cases,
            "passed_test_cases": self.passed_test_cases,
            "metadata": self.metadata or {}
        }

class CorrectnessEvaluator:
    def __init__(self, level1_dir: str = "level1", solutions_dir: str = "kernel_agent_solutions", 
                 results_dir: str = "correctness_results"):
        self.level1_dir = Path(level1_dir)
        self.solutions_dir = Path(solutions_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for kernel evaluation")
        
        print(f"Using device: {self.device} ({torch.cuda.get_device_name(self.device)})")
        
    def load_problem_files(self) -> Dict[str, Tuple[str, str]]:
        """Load PyTorch reference and generated solution files
        
        Returns:
            Dict mapping problem_name to (reference_code, solution_code)
        """
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
    
    def evaluate_single_kernel(self, problem_name: str, reference_code: str, 
                             solution_code: str, num_test_cases: int = 5) -> CorrectnessResult:
        """Evaluate correctness of a single kernel"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {problem_name}")
        print(f"{'='*60}")
        
        result = CorrectnessResult(problem_name=problem_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"
            build_dir.mkdir()
            
            try:
                print("Loading reference model...")
                context = {}
                Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                    reference_code, context
                )
                
                if Model is None:
                    result.compilation_error = "Failed to load reference model"
                    return result
                
                set_seed(42)
                init_inputs = get_init_inputs()
                init_inputs = [
                    x.cuda(self.device) if isinstance(x, torch.Tensor) else x 
                    for x in init_inputs
                ]
                
                with torch.no_grad():
                    set_seed(42)
                    original_model = Model(*init_inputs)
                    original_model = original_model.cuda(self.device)
                
                print("Reference model loaded successfully")
                
                print("Compiling custom CUDA kernel...")
                try:
                    os.environ["TORCH_USE_CUDA_DSA"] = "1"
                    ModelNew = load_custom_model(solution_code, context, str(build_dir))
                    torch.cuda.synchronize(self.device)
                    
                    if ModelNew is None:
                        result.compilation_error = "Failed to load custom model class"
                        return result
                        
                    result.compiled = True
                    print("Kernel compiled successfully")
                    
                except Exception as e:
                    result.compilation_error = str(e)
                    result.compiled = False
                    print(f"Compilation failed: {e}")
                    return result
                
                print("Instantiating custom model...")
                try:
                    with torch.no_grad():
                        set_seed(42)
                        custom_model = ModelNew(*init_inputs)
                        custom_model = custom_model.cuda(self.device)
                    print("Custom model instantiated successfully")
                    
                except Exception as e:
                    result.runtime_error = str(e)
                    print(f"Model instantiation failed: {e}")
                    return result
                
                print(f"Running {num_test_cases} correctness tests...")
                result.num_test_cases = num_test_cases
                
                metadata = {}
                metadata["hardware"] = torch.cuda.get_device_name(self.device)
                metadata["device"] = str(self.device)
                
                try:
                    kernel_result = run_and_check_correctness(
                        original_model,
                        custom_model,
                        get_inputs,
                        metadata=metadata,
                        num_correct_trials=num_test_cases,
                        verbose=True,
                        seed=42,
                        device=self.device
                    )
                    
                    result.correctness = kernel_result.correctness
                    result.metadata = kernel_result.metadata
                    
                    if "correctness_trials" in kernel_result.metadata:
                        trials_str = kernel_result.metadata["correctness_trials"]
                        if "(" in trials_str and "/" in trials_str:
                            passed_str = trials_str.split("(")[1].split("/")[0].strip()
                            result.passed_test_cases = int(passed_str)
                        else:
                            result.passed_test_cases = num_test_cases if result.correctness else 0
                    else:
                        result.passed_test_cases = num_test_cases if result.correctness else 0
                    
                    print(f"Correctness: {result.correctness}")
                    print(f"Passed: {result.passed_test_cases}/{result.num_test_cases} test cases")
                    
                except Exception as e:
                    result.runtime_error = str(e)
                    print(f"Correctness testing failed: {e}")
                
            except Exception as e:
                result.compilation_error = str(e)
                print(f"Unexpected error: {e}")
            
            finally:
                try:
                    graceful_eval_cleanup(context, self.device)
                except:
                    pass
        
        return result
    
    def evaluate_all_kernels(self, num_test_cases: int = 5) -> Dict:
        """Evaluate correctness of all generated kernels"""
        problems = self.load_problem_files()
        
        if not problems:
            print("No problems found to evaluate!")
            return {}
        
        summary = {
            "total_problems": len(problems),
            "compiled": 0,
            "correct": 0,
            "compilation_errors": 0,
            "runtime_errors": 0,
            "evaluation_time": 0
        }
        
        results = []
        start_time = time.time()
        
        for problem_name, (reference_code, solution_code) in problems.items():
            result = self.evaluate_single_kernel(
                problem_name, reference_code, solution_code, num_test_cases
            )
            
            results.append(result.to_dict())
            
            if result.compiled:
                summary["compiled"] += 1
            if result.correctness:
                summary["correct"] += 1
            if result.compilation_error and not result.compiled:
                summary["compilation_errors"] += 1
            if result.runtime_error:
                summary["runtime_errors"] += 1
        
        summary["evaluation_time"] = time.time() - start_time
        
        summary["compilation_rate"] = summary["compiled"] / summary["total_problems"]
        summary["correctness_rate"] = summary["correct"] / summary["total_problems"]
        
        detailed_results = {
            "summary": summary,
            "results": results,
            "metadata": {
                "device": str(self.device),
                "hardware": torch.cuda.get_device_name(self.device),
                "num_test_cases_per_kernel": num_test_cases,
                "timestamp": time.time()
            }
        }
        
        detailed_results = check_metadata_serializable_all_types(detailed_results)
        
        results_file = self.results_dir / "correctness_evaluation.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("CORRECTNESS EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {summary['total_problems']}")
        print(f"Compiled successfully: {summary['compiled']} ({summary['compilation_rate']:.1%})")
        print(f"Passed correctness: {summary['correct']} ({summary['correctness_rate']:.1%})")
        print(f"Compilation errors: {summary['compilation_errors']}")
        print(f"Runtime errors: {summary['runtime_errors']}")
        print(f"Evaluation time: {summary['evaluation_time']:.2f} seconds")
        print(f"Results saved to: {results_file}")
        
        return detailed_results

def main():
    level1_dir = "level1"
    solutions_dir = "kernel_agent_solutions"
    results_dir = "correctness_results"
    num_test_cases = 5
    
    evaluator = CorrectnessEvaluator(
        level1_dir=level1_dir,
        solutions_dir=solutions_dir,
        results_dir=results_dir
    )
    
    evaluator.evaluate_all_kernels(num_test_cases=num_test_cases)

if __name__ == "__main__":
    main()