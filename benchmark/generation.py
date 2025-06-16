import os
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict
import time

# add the src directory to the path to import router
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from router import KernelAgentClient

class KernelGenerator:
    def __init__(self, problem_dir: str, solutions_dir: str, mcp_servers: List[str]):
        self.problem_dir = Path(problem_dir)
        self.solutions_dir = Path(solutions_dir)
        self.mcp_servers = mcp_servers
        
        self.solutions_dir.mkdir(exist_ok=True)        
        self.logs_dir = self.solutions_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def load_problems(self) -> Dict[str, str]:
        problems = {}
        
        if not self.problem_dir.exists():
            raise FileNotFoundError(f"Level 1 directory not found: {self.problem_dir}")
        
        for file_path in self.problem_dir.glob("*.py"):
            problem_name = file_path.stem
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            problems[problem_name] = content
            
        print(f"Loaded {len(problems)} level 1 problems:")
        for name in sorted(problems.keys()):
            print(f"  - {name}")
            
        return problems
    
    async def generate_kernel_for_problem(self, problem_name: str, problem_content: str, 
                                        max_iterations: int = 5) -> Dict:
        """Generate CUDA kernel for a single problem using the router client"""
        print(f"\n{'='*60}")
        print(f"Generating kernel for: {problem_name}")
        print(f"{'='*60}")
        
        client = KernelAgentClient()
        
        try:
            # connect to MCP servers
            for i, server_script in enumerate(self.mcp_servers):
                server_name = f"server_{i}" if i > 0 else "default"
                await client.connect_to_server(server_script, server_name)
                print(f"Connected to server: {server_name}")
            
            client.logs.add_original_code(problem_content)
            
            start_time = time.time()
            
            # for max iterations
            # kernel_solution = await client.generate_kernel(problem_content, max_iterations=max_iterations)
            
            kernel_solution = await client.generate_kernel(problem_content)
            
            generation_time = time.time() - start_time
            
            solution_file = self.solutions_dir / f"{problem_name}_solution.py"
            with open(solution_file, 'w', encoding='utf-8') as f:
                f.write(kernel_solution)
            
            log_file = self.logs_dir / f"{problem_name}_logs.jsonl"
            client.logs.dump(str(log_file))
            
            print(f"\nKernel generated successfully!")
            print(f"Solution saved to: {solution_file}")
            print(f"Logs saved to: {log_file}")
            print(f"Generation time: {generation_time:.2f} seconds")
            
            return {
                "problem_name": problem_name,
                "success": True,
                "solution_file": str(solution_file),
                "log_file": str(log_file),
                "generation_time": generation_time,
                "kernel_length": len(kernel_solution)
            }
            
        except Exception as e:
            print(f"Error generating kernel for {problem_name}: {str(e)}")
            return {
                "problem_name": problem_name,
                "success": False,
                "error": str(e),
                "generation_time": 0,
                "kernel_length": 0
            }
        finally:
            await client.cleanup()
    
    async def generate_all_kernels(self, max_iterations: int = 5, resume: bool = True) -> Dict:
        """Generate CUDA kernels for all level 1 problems"""
        problems = self.load_problems()
        
        skipped_count = 0
        if resume:
            remaining_problems = {}
            
            for problem_name, problem_content in problems.items():
                solution_file = self.solutions_dir / f"{problem_name}_solution.py"
                if solution_file.exists():
                    print(f"Skipping {problem_name} (solution already exists)")
                    skipped_count += 1
                else:
                    remaining_problems[problem_name] = problem_content
            
            problems = remaining_problems
            print(f"\nResuming generation: {skipped_count} problems skipped, {len(problems)} remaining")
        
        results = {
            "summary": {
                "total_problems": len(problems),
                "successful": 0,
                "failed": 0,
                "total_time": 0,
                "max_iterations_used": max_iterations,
                "skipped": skipped_count
            },
            "results": []
        }
        
        if not problems:
            print("No problems to process!")
            return results
        
        start_time = time.time()
        
        for problem_name, problem_content in problems.items():
            result = await self.generate_kernel_for_problem(
                problem_name, problem_content, max_iterations
            )
            results["results"].append(result)
            
            if result["success"]:
                results["summary"]["successful"] += 1
            else:
                results["summary"]["failed"] += 1
            
            results["summary"]["total_time"] += result["generation_time"]
        
        total_time = time.time() - start_time
        results["summary"]["total_wall_time"] = total_time
        
        summary_file = self.solutions_dir / "generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems processed: {results['summary']['total_problems']}")
        print(f"Problems skipped: {results['summary']['skipped']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Max iterations per problem: {max_iterations}")
        print(f"Total generation time: {results['summary']['total_time']:.2f} seconds")
        print(f"Total wall time: {total_time:.2f} seconds")
        if len(problems) > 0:
            print(f"Average time per problem: {total_time / len(problems):.2f} seconds")
        print(f"Summary saved to: {summary_file}")
        
        failed_problems = [r for r in results["results"] if not r["success"]]
        if failed_problems:
            print(f"\nFailed problems ({len(failed_problems)}):")
            for result in failed_problems:
                print(f"  - {result['problem_name']}: {result.get('error', 'Unknown error')}")
        
        return results

async def main():
    problem_dir = "level1"
    solutions_dir = "kernel_agent_solutions"
    mcp_servers = [
        "../src/nsys.py",
        "../src/benchmark.py"
    ]
    max_iterations = 5
    resume = True  # set to false to regenerate all solutions from scratch
    
    # Validate inputs
    problem_path = Path(problem_dir)
    if not problem_path.exists():
        print(f"Error: Problem directory not found: {problem_path}")
        return
    
    py_files = list(problem_path.glob("*.py"))
    if not py_files:
        print(f"Error: No Python files found in {problem_path}")
        return
    
    for server_path in mcp_servers:
        if not Path(server_path).exists():
            print(f"Error: MCP server not found: {server_path}")
            return
    
    print(f"Starting kernel generation for up to {len(py_files)} problems...")
    print(f"MCP servers: {mcp_servers}")
    print(f"Max iterations per problem: {max_iterations}")
    print(f"Resume mode: {'enabled' if resume else 'disabled'}")
    
    generator = KernelGenerator(
        problem_dir=problem_dir,
        solutions_dir=solutions_dir,
        mcp_servers=mcp_servers
    )
    
    await generator.generate_all_kernels(max_iterations=max_iterations, resume=resume)

if __name__ == "__main__":
    asyncio.run(main())