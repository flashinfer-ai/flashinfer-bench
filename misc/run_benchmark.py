import argparse
import json
import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class AggregatedResults:
    
    def __init__(self):
        self.datapoint_results = {}
        self.total_datapoints = 0
        self.successful_datapoints = 0
        self.failed_datapoints = 0
        
    def add_datapoint_result(self, datapoint_name: str, result_file: str):
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            self.datapoint_results[datapoint_name] = result_data
            self.total_datapoints += 1
            
            if result_data.get('success_rate', 0) > 0:
                self.successful_datapoints += 1
            else:
                self.failed_datapoints += 1
                
        except Exception as e:
            print(f"Warning: Failed to load result for {datapoint_name}: {e}")
            self.failed_datapoints += 1
            self.total_datapoints += 1
    
    def compute_summary(self) -> Dict[str, Any]:
        if not self.datapoint_results:
            return {
                "total_datapoints": 0,
                "successful_datapoints": 0,
                "failed_datapoints": 0,
                "overall_success_rate": 0.0,
                "error": "No valid results found"
            }
        
        compilation_rates = []
        correctness_rates = []
        success_rates = []
        avg_speedups = []
        max_speedups = []
        
        datapoint_summaries = {}
        
        for datapoint_name, result in self.datapoint_results.items():
            compilation_rate = result.get('compilation_success_rate', 0.0)
            correctness_rate = result.get('correctness_success_rate', 0.0) 
            success_rate = result.get('success_rate', 0.0)
            avg_speedup = result.get('avg_speedup', 0.0)
            max_speedup = result.get('max_speedup', 0.0)
            
            compilation_rates.append(compilation_rate)
            correctness_rates.append(correctness_rate)
            success_rates.append(success_rate)
            
            if avg_speedup > 0:
                avg_speedups.append(avg_speedup)
            if max_speedup > 0:
                max_speedups.append(max_speedup)
            
            datapoint_summaries[datapoint_name] = {
                "compilation_success_rate": compilation_rate,
                "correctness_success_rate": correctness_rate, 
                "success_rate": success_rate,
                "avg_speedup": avg_speedup,
                "max_speedup": max_speedup,
                "total_generations": len(result.get('generations', []))
            }
        
        summary = {
            "total_datapoints": self.total_datapoints,
            "successful_datapoints": self.successful_datapoints,
            "failed_datapoints": self.failed_datapoints,
            "overall_success_rate": self.successful_datapoints / self.total_datapoints if self.total_datapoints > 0 else 0.0,
            
            "avg_compilation_success_rate": np.mean(compilation_rates) if compilation_rates else 0.0,
            "avg_correctness_success_rate": np.mean(correctness_rates) if correctness_rates else 0.0,
            "avg_success_rate": np.mean(success_rates) if success_rates else 0.0,
            
            "avg_speedup_across_successful": np.mean(avg_speedups) if avg_speedups else 0.0,
            "max_speedup_achieved": np.max(max_speedups) if max_speedups else 0.0,
            "num_datapoints_with_speedup": len(avg_speedups),
            
            "datapoint_summaries": datapoint_summaries
        }
        
        return summary
    
    def print_summary(self):
        summary = self.compute_summary()
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total datapoints processed: {summary['total_datapoints']}")
        print(f"  Successful datapoints: {summary['successful_datapoints']}")
        print(f"  Failed datapoints: {summary['failed_datapoints']}")
        print(f"  Overall success rate: {summary['overall_success_rate']*100:.1f}%")
        
        print(f"\nAverage Performance Across Datapoints:")
        print(f"  Average compilation success rate: {summary['avg_compilation_success_rate']*100:.1f}%")
        print(f"  Average correctness success rate: {summary['avg_correctness_success_rate']*100:.1f}%")
        print(f"  Average success rate: {summary['avg_success_rate']*100:.1f}%")
        
        if summary['num_datapoints_with_speedup'] > 0:
            print(f"\nSpeedup Statistics:")
            print(f"  Datapoints with successful speedup: {summary['num_datapoints_with_speedup']}")
            print(f"  Average speedup across successful: {summary['avg_speedup_across_successful']:.2f}x")
            print(f"  Maximum speedup achieved: {summary['max_speedup_achieved']:.2f}x")
        
        print(f"\nPer-Datapoint Results:")
        print(f"{'Datapoint':<50} {'Success Rate':<12} {'Avg Speedup':<12} {'Max Speedup':<12}")
        print("-" * 86)
        
        for name, result in summary['datapoint_summaries'].items():
            datapoint_short = name.replace('.py', '')[:49]
            success_rate = f"{result['success_rate']*100:.1f}%"
            avg_speedup = f"{result['avg_speedup']:.2f}x" if result['avg_speedup'] > 0 else "N/A"
            max_speedup = f"{result['max_speedup']:.2f}x" if result['max_speedup'] > 0 else "N/A"
            
            print(f"{datapoint_short:<50} {success_rate:<12} {avg_speedup:<12} {max_speedup:<12}")


def find_datapoint_files(dataset_dir: str) -> List[str]:
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    pattern = os.path.join(dataset_dir, "*.py")
    datapoint_files = glob.glob(pattern)
    
    datapoint_files = [f for f in datapoint_files if not os.path.basename(f).startswith('__')]
    
    if not datapoint_files:
        raise ValueError(f"No datapoint files found in {dataset_dir}")
    
    return sorted(datapoint_files)


def run_single_benchmark(benchmark_script: str, datapoint_file: str, generator_file: str, 
                        output_file: str, args: argparse.Namespace) -> bool:
    cmd = [
        sys.executable, benchmark_script,
        datapoint_file,
        generator_file,
        "--output", output_file,
        "--warmup", str(args.warmup),
        "--iter", str(args.iter),
        "--report-n", str(args.report_n),
        "--max-diff-limit", str(args.max_diff_limit),
        "--correctness-trials", str(args.correctness_trials),
        "--seed", str(args.seed),
        "--loader-type", args.loader_type,
        "--backend", args.backend,
        "--use-ncu", args.use_ncu,
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        print(f"Running: {os.path.basename(datapoint_file)}")
        if args.verbose:
            print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=not args.verbose, text=True, timeout=args.timeout)
        
        if result.returncode == 0:
            print(f"Success")
            return True
        else:
            print(f"Failed with return code {result.returncode}")
            if not args.verbose and result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timed out after {args.timeout} seconds")
        return False
    except Exception as e:
        print(f"Failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("generator", help="Path to kernel generator file (.py)")
    
    parser.add_argument("--dataset-dir", required=True,
                       help="Directory containing datapoint files to benchmark")
    
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iter", type=int, default=10, help="Number of timing iterations per round")
    parser.add_argument("--report-n", type=int, default=16, help="Number of generation rounds")
    parser.add_argument("--max-diff-limit", type=float, default=1e-5, help="Maximum difference for correctness")
    
    parser.add_argument("--correctness-trials", type=int, default=1, 
                       help="Number of correctness trials with different inputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--device", help="CUDA device to use (e.g., 'cuda:0', 0, or 'auto')")
    parser.add_argument("--backend", choices=["cuda", "triton"], default="cuda", help="Backend to use")
    parser.add_argument("--loader-type", choices=["auto", "kernelbench", "triton", "flashinfer"], default="auto", 
                        help="Type of kernel loader to use")
    
    parser.add_argument("--use-ncu", type=str, default="false", help="Enable NCU profiling")
    
    parser.add_argument("--output", default="benchmark_results/aggregated_results.json", 
                       help="Output aggregated results JSON file")
    parser.add_argument("--benchmark-script", default="benchmark/benchmark.py", 
                       help="Path to the benchmark.py script")
    parser.add_argument("--timeout", type=int, default=1800, 
                       help="Timeout per datapoint in seconds (30 minutes)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--keep-temp", action="store_true", 
                       help="Keep temporary directory for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.generator):
        print(f"Error: Generator file '{args.generator}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.benchmark_script):
        print(f"Error: Benchmark script '{args.benchmark_script}' not found")
        sys.exit(1)
    
    try:
        datapoint_files = find_datapoint_files(args.dataset_dir)
        print(f"Found {len(datapoint_files)} datapoint files in {args.dataset_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    benchmark_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(benchmark_results_dir, exist_ok=True)
    
    temp_dir = os.path.join(benchmark_results_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using results directory: {temp_dir}")
    
    if not os.path.isabs(args.output):
        args.output = os.path.join(benchmark_results_dir, os.path.basename(args.output))
    
    try:
        aggregator = AggregatedResults()
        
        for i, datapoint_file in enumerate(datapoint_files, 1):
            datapoint_name = os.path.basename(datapoint_file)
            temp_result_file = os.path.join(temp_dir, f"result_{i:03d}_{datapoint_name.replace('.py', '.json')}")
            
            print(f"\n[{i}/{len(datapoint_files)}] Processing {datapoint_name}")
            
            success = run_single_benchmark(
                args.benchmark_script, datapoint_file, args.generator,
                temp_result_file, args
            )
            
            if success and os.path.exists(temp_result_file):
                aggregator.add_datapoint_result(datapoint_name, temp_result_file)
            else:
                print(f"  Warning: No result file generated for {datapoint_name}")
        
        summary = aggregator.compute_summary()
        
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nAggregated results saved to: {args.output}")
        
        aggregator.print_summary()
        
    finally:
        if not args.keep_temp:
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up results directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up results directory: {e}")
        else:
            print(f"\nResults directory preserved: {temp_dir}")


if __name__ == "__main__":
    main()
