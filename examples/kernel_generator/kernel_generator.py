import os
import random
import re
from typing import Dict, List, Optional

import openai
from kernel_generator_prompts import get_optimization_prompt, get_prompt

from flashinfer_bench import (
    Benchmark,
    BenchmarkConfig,
    BuildSpec,
    Definition,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedLanguages,
    Trace,
    TraceSet,
    Workload,
)


class KernelGenerator:
    def __init__(
        self,
        model_name: str,
        language: str = "triton",
        target_gpu: str = "H100",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "high",  # only used for openai reasoning models
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.reasoning_effort = reasoning_effort

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in LLM_API_KEY environment variable"
                )

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
            # Default Python
            return SupportedLanguages.PYTHON

    def generate(
        self, traceset: TraceSet, definition: Definition, max_opt_rounds: int = 10
    ) -> Solution:
        """
        Generate an optimized solution through iterative improvement using flashinfer-bench feedback.

        Args:
            traceset: The TraceSet containing workloads for evaluation
            definition: The workload definition to implement kernel for
            max_opt_rounds: Maximum number of optimization rounds (default: 10)

        Returns:
            Solution: a solution dataclass containing the optimized kernel code
        """
        workloads = traceset.workload.get(definition.name, [])
        if not workloads:
            raise ValueError(
                f"No workloads found for definition '{definition.name}' in the provided TraceSet"
            )

        selected_workload = random.choice(workloads)

        print(f"Generating optimized solution for {definition.name}")
        print(f"Using workload {selected_workload.workload.uuid} for optimization feedback")
        prompt = get_prompt(self.language, definition, self.target_gpu)
        code_result = self._generate_code_from_prompt(prompt)
        current_code = code_result["cleaned"]
        current_raw_code = code_result["raw"]

        for round_num in range(1, max_opt_rounds + 1):
            print(f"\n=== Optimization Round {round_num}/{max_opt_rounds} ===")

            solution = self._create_solution_from_code(current_code, definition, round_num)

            temp_traceset = TraceSet(
                root=traceset.root,
                definitions={definition.name: definition},
                solutions={definition.name: [solution]},
                workload={definition.name: [selected_workload]},
                traces={},
            )

            print(f"Evaluating solution...")
            benchmark = Benchmark(temp_traceset, log_level="WARNING")
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

            if round_num == max_opt_rounds:
                print(f"Reached maximum rounds ({max_opt_rounds}), returning current solution")
                return solution

            print(
                f"Solution failed with {evaluation.status.value}, extracting feedback for next round..."
            )
            if evaluation.error:
                print(f"Error details: {evaluation.error}")

            optimization_prompt = get_optimization_prompt(
                self.language, definition, trace, current_raw_code, self.target_gpu
            )

            print(f"Generating optimized code for round {round_num + 1}...")
            code_result = self._generate_code_from_prompt(optimization_prompt)
            current_code = code_result["cleaned"]
            current_raw_code = code_result["raw"]

    def _parse_xml_files(self, code: str) -> Dict[str, str]:
        files = {}

        patterns = {
            "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
            "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
            "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
        }

        for filename, pattern in patterns.items():
            match = re.search(pattern, code, re.DOTALL)
            if match:
                content = match.group(1).strip()
                files[filename] = content
            else:
                print(f"Warning: Could not find {filename} in generated code")

        return files

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code. For CUDA, parse XML and return dict. For others, clean Python syntax."""
        if self.language.lower() == "cuda":
            return self._parse_xml_files(code)

        # For non-CUDA languages (triton, python), clean up markdown and hex floats
        if "```" in code:
            if code.startswith("```"):
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                code = "\n".join(lines)
            
            if code.endswith("```"):
                lines = code.split("\n")
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                code = "\n".join(lines)
            
            code = code.replace("```", "")

        hex_float_pattern = r"0x[0-9a-fA-F]*\.[0-9a-fA-F]*p[-+]?\d+"
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

    def _generate_code_from_prompt(self, prompt: str):
        try:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                response = self.client.responses.create(
                    model=self.model_name, input=prompt, reasoning={"effort": self.reasoning_effort}
                )
                generated_code = response.output_text.strip()
            else:  # We use the completions api for OpenAI SDK compatible models
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}]
                )
                generated_code = response.choices[0].message.content.strip()

            cleaned_code = self._clean_generated_code(generated_code)

            return {"raw": generated_code, "cleaned": cleaned_code}

        except Exception as e:
            print(f"Error while generating code: {e}")
            raise

    def _create_solution_from_code(self, code, definition: Definition, round_num: int) -> Solution:
        # Include reasoning effort in name and description for GPT-5 models
        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_{self.reasoning_effort}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, reasoning effort: {self.reasoning_effort})"
        else:
            solution_name = (
                f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}"
            )
            solution_description = (
                f"{self.model_name} optimized kernel for {definition.name} (round {round_num})"
            )

        # Handle different code formats based on language
        if self.language.lower() == "cuda" and isinstance(code, dict):
            # For CUDA, we have multiple files
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_point = "main.cpp::run"
        else:
            # For single-file languages (triton, python)
            if isinstance(code, dict):
                code = next(iter(code.values()))

            sources = [SourceFile(path="main.py", content=code)]
            entry_point = "main.py::run"

        solution = Solution(
            name=solution_name,
            definition=definition.name,
            author=self.model_name,
            spec=BuildSpec(
                language=self._get_supported_language(),
                target_hardware=[self.target_gpu],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution
