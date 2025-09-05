from typing import List, Optional
import openai
import os
import re
import random

from flashinfer_bench import (
    Benchmark,
    BenchmarkConfig,
    Definition,
    Solution,
    TraceSet,
    Trace,
    Workload,
    EvaluationStatus,
    BuildSpec,
    SourceFile,
    SupportedLanguages,
)

from kernel_generator_prompts import get_prompt, get_optimization_prompt

class KernelGenerator:
    def __init__(self, model_name: str, language: str = "triton", target_gpu: str = "H100", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5", "claude-4", "llama-4", "gemini-2.5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "A100", "V100", "RTX4090", default: "H100")
            api_key: API key (if None, uses OPENAI_API_KEY environment variable)
            base_url: Base URL for the API (if None, uses OpenAI's default)
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
        
        client_kwargs = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
            
        self.client = openai.OpenAI(**client_kwargs)
    
    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }
        if self.language.lower() in language_map:
            return language_map[self.language.lower()]
        else:
            # Default to Python if unknown language
            return SupportedLanguages.PYTHON
    
    def generate(self, definition: Definition, pass_k: int = 1) -> List[Solution]:
        """        
        Args:
            definition: The workload definition to implement
            pass_k: Number of independent solutions to generate
        """
        solutions = []
        
        prompt = get_prompt(self.language, definition, self.target_gpu)
        
        for i in range(pass_k):
            print(f"Generating solution at round {i+1}")
            try:
                if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=prompt
                    )
                    generated_code = response.output_text.strip()
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    generated_code = response.choices[0].message.content.strip()
                
                #print(f"Generated code: {generated_code}")
                generated_code = self._clean_generated_code(generated_code)
                
                entry_point = "main.py::run"
                
                solution = Solution(
                    name=f"{self.model_name}_optimized_{definition.name}_{self.language}_solution_{i+1}",
                    definition=definition.name,
                    author=self.model_name,
                    spec=BuildSpec(
                        language=self._get_supported_language(),
                        target_hardware=[self.target_gpu],
                        entry_point=entry_point,
                    ),
                    sources=[SourceFile(
                        path="main.py",
                        content=generated_code
                    )],
                    description=f"{self.model_name} generated kernel for {definition.name}"
                )
                
                solutions.append(solution)
                
            except Exception as e:
                print(f"Error while generating solution at round {i+1}: {e}")
                continue
        
        return solutions
    
    def optimized_generate(self, traceset: TraceSet, definition: Definition, rounds: int = 3) -> Solution:
        """
        Generate an optimized solution through iterative improvement using benchmark feedback.
        
        Args:
            traceset: The TraceSet containing workloads for evaluation
            definition: The workload definition to implement
            rounds: Maximum number of optimization rounds (default: 3)
            
        Returns:
            Solution: an optimized solution
        """
        workloads = traceset.workload.get(definition.name, [])
        if not workloads:
            raise ValueError(f"No workloads found for definition '{definition.name}' in the provided TraceSet")
        
        selected_workload = random.choice(workloads)
        
        print(f"Generating optimized solution for {definition.name}")
        print(f"Using workload {selected_workload.workload.uuid} for optimization feedback")
        prompt = get_prompt(self.language, definition, self.target_gpu)
        current_code = self._generate_code_from_prompt(prompt)
        
        for round_num in range(1, rounds + 1):
            print(f"\n=== Optimization Round {round_num}/{rounds} ===")
            
            solution = self._create_solution_from_code(current_code, definition, round_num)
            
            temp_traceset = TraceSet(
                root=traceset.root,
                definitions={definition.name: definition},
                solutions={definition.name: [solution]},
                workload={definition.name: [selected_workload]},
                traces={}
            )
            
            print(f"Evaluating solution...")
            benchmark = Benchmark(temp_traceset, log_level="WARNING")  # Reduce log verbosity
            result_traceset = benchmark.evaluate(BenchmarkConfig())
            
            traces = result_traceset.traces.get(definition.name, [])
            if not traces:
                print("No evaluation traces found, stopping optimization")
                break
                
            trace = traces[0]  # Should be only one trace
            evaluation = trace.evaluation
            
            print(f"Evaluation status: {evaluation.status.value}")
            
            if evaluation.status == EvaluationStatus.PASSED:
                print(f"Solution PASSED! Speedup: {evaluation.performance.speedup_factor:.2f}x")
                return solution
            
            if round_num == rounds:
                print(f"Reached maximum rounds ({rounds}), returning current solution")
                return solution
            
            print(f"Solution failed with {evaluation.status.value}, extracting feedback for next round...")
            if evaluation.error:
                print(f"Error details: {evaluation.error}")
            
            # Generate optimization prompt
            optimization_prompt = get_optimization_prompt(
                definition, trace, current_code, self.target_gpu
            )
            
            # Generate improved code
            print(f"Generating optimized code for round {round_num + 1}...")
            current_code = self._generate_code_from_prompt(optimization_prompt)
        
        # This should not be reached due to the return in the last round
        return solution
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code to ensure valid Python syntax."""
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        
        hex_float_pattern = r'0x[0-9a-fA-F]*\.[0-9a-fA-F]*p[-+]?\d+'
        hex_floats = re.findall(hex_float_pattern, code)
        
        for hex_float in hex_floats:
            try:
                if hex_float == "0x1.62e42fefa39efp-1":
                    decimal_val = "0.6931471805599453"
                elif hex_float == "0x1.71547652b82fep0":
                    decimal_val = "2.718281828459045"
                elif hex_float == "0x1.921fb54442d18p1":
                    decimal_val = "3.141592653589793"
                else:
                    decimal_val = "1.0"
                
                code = code.replace(hex_float, decimal_val)
            except Exception as e:
                print(f"Warning: Could not convert hex float {hex_float}: {e}")
                code = code.replace(hex_float, "1.0")
        
        return code
    
    def _generate_code_from_prompt(self, prompt: str) -> str:
        try:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt
                )
                generated_code = response.output_text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                generated_code = response.choices[0].message.content.strip()
            
            generated_code = self._clean_generated_code(generated_code)
            
            return generated_code
            
        except Exception as e:
            print(f"Error while generating code: {e}")
            raise
    
    def _create_solution_from_code(self, code: str, definition: Definition, round_num: int) -> Solution:
        solution = Solution(
            name=f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}",
            definition=definition.name,
            author=self.model_name,
            spec=BuildSpec(
                language=self._get_supported_language(),
                target_hardware=[self.target_gpu],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(
                path="main.py",
                content=code
            )],
            description=f"{self.model_name} optimized kernel for {definition.name} (round {round_num})"
        )
        return solution
    
