import asyncio
import os
import random
import re
from typing import Dict, List, Optional, Tuple

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


def __parse_hex_float(hex_str: str) -> float:

    hex_str = hex_str.strip()

    # hex float pattern: 0x[mantissa]p[exponent]
    pattern = r"^0[xX]([0-9a-fA-F]*\.?[0-9a-fA-F]+)[pP]([+-]?\d+)$"
    match = re.match(pattern, hex_str)

    if not match:
        raise ValueError(f"Invalid hex format: {hex_str}")

    mantissa_str, exponent_str = match.groups()

    if "." in mantissa_str:
        integer_part, fractional_part = mantissa_str.split(".")
    else:
        integer_part = mantissa_str
        fractional_part = ""

    if integer_part:
        mantissa_value = float(int(integer_part, 16))
    else:
        mantissa_value = 0.0

    if fractional_part:
        frac_value = 0.0
        for i, digit in enumerate(fractional_part, 1):
            frac_value += int(digit, 16) / (16**i)
        mantissa_value += frac_value

    exponent = int(exponent_str)

    result = mantissa_value * (2**exponent)

    return result


def _replace_hex_floats_in_code(code: str) -> str:

    hex_float_pattern = r"0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]+[pP][+-]?\d+"

    def replace_match(match):
        hex_str = match.group(0)
        try:
            decimal_value = __parse_hex_float(hex_str)
            return f"{decimal_value:.17g}"
        except Exception as e:
            print(f"Warning: Could not convert hex float {hex_str}: {e}")
            return hex_str

    return re.sub(hex_float_pattern, replace_match, code)


class KernelGenerator:
    def __init__(
        self,
        model_name: str,
        language: str = "triton",
        target_gpu: str = "H100",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "high",  # only used for openai reasoning models
        use_ffi: bool = True,
        destination_passing_style: bool = True,
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
            use_ffi: Use FFI bindings when generating CUDA kernels.
            destination_passing_style: Generate kernels in destination-passing style. Value-returning style if false.
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.reasoning_effort = reasoning_effort
        self.use_ffi = use_ffi
        self.destination_passing_style = destination_passing_style

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in LLM_API_KEY environment variable"
                )

        client_kwargs = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

    def _get_supported_language(self) -> SupportedLanguages:
        language_map = {
            "python": SupportedLanguages.PYTHON,
            "triton": SupportedLanguages.TRITON,
            "cuda": SupportedLanguages.CUDA,
        }
        if self.language.lower() in language_map:
            return language_map[self.language.lower()]
        else:
            return SupportedLanguages.PYTHON

    def generate(
        self,
        trace_set: TraceSet,
        definition: Definition,
        gen_rounds: int = 10,
        beam: bool = False,
        beam_width: int = 3,
    ) -> Solution:
        """
        Generate an optimized solution through iterative improvement using flashinfer-bench feedback.

        Args:
            trace_set: The TraceSet containing workloads for evaluation
            definition: The workload definition to implement kernel for
            gen_rounds: Number of generation rounds to run (or search depth if beam=True)
            beam: beam search flag, default to False as it's more expensive to run
            beam_width: Number of candidates to maintain in beam search (default: 3)

        Returns:
            Solution: a solution dataclass containing the optimized kernel code
        """
        workloads = trace_set.workloads.get(definition.name, [])
        if not workloads:
            raise ValueError(
                f"No workloads found for definition '{definition.name}' in the provided TraceSet"
            )

        selected_workload = random.choice(workloads)

        print(f"Generating optimized solution for {definition.name}")
        print(f"Using workload {selected_workload.workload.uuid} for optimization feedback")

        if beam:
            return self._beam_search_generate(
                trace_set, definition, selected_workload, gen_rounds, beam_width
            )
        else:
            return asyncio.run(
                self._sequential_generate_async(
                    trace_set, definition, selected_workload, gen_rounds
                )
            )

    async def _sequential_generate_async(
        self, trace_set: TraceSet, definition: Definition, selected_workload, gen_rounds: int
    ) -> Solution:
        prompt = get_prompt(
            self.language, definition, self.target_gpu, self.use_ffi, self.destination_passing_style
        )
        code_result = await self._generate_code_from_prompt(prompt)
        current_code = code_result["cleaned"]
        current_raw_code = code_result["raw"]

        passing_solutions: List[Tuple[Solution, Trace]] = []
        last_solution = None
        last_trace = None

        for round_num in range(1, gen_rounds + 1):
            print(f"\nGeneration Round {round_num}/{gen_rounds}")

            solution = self._create_solution_from_code(current_code, definition, round_num)
            last_solution = solution

            traces = self._evaluate_solutions(trace_set, definition, [solution], selected_workload)
            trace = traces[0] if traces else None
            if trace:
                last_trace = trace
                evaluation = trace.evaluation
                print(f"Evaluation status: {evaluation.status.value}")

                if evaluation.status == EvaluationStatus.PASSED:
                    speedup = evaluation.performance.speedup_factor
                    print(f"Solution PASSED! Speedup: {speedup:.2f}x")
                    passing_solutions.append((solution, trace))
                else:
                    print(f"Solution failed with {evaluation.status.value}")
                    if evaluation.log:
                        print("Error details:")
                        print(evaluation.log)

            if round_num < gen_rounds:
                best_trace = self._get_best_trace(passing_solutions)
                opt_trace = best_trace if best_trace else last_trace

                if opt_trace:
                    optimization_prompt = get_optimization_prompt(
                        self.language,
                        definition,
                        opt_trace,
                        current_raw_code,
                        self.target_gpu,
                        self.use_ffi,
                        self.destination_passing_style,
                    )
                else:
                    optimization_prompt = get_prompt(
                        self.language,
                        definition,
                        self.target_gpu,
                        self.use_ffi,
                        self.destination_passing_style,
                    )

                print(f"Generating code for round {round_num + 1}...")
                code_result = await self._generate_code_from_prompt(optimization_prompt)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]

        return self._select_best_solution(passing_solutions, last_solution)

    def _beam_search_generate(
        self,
        trace_set: TraceSet,
        definition: Definition,
        selected_workload,
        depth: int,
        beam_width: int,
    ) -> Solution:
        print(f"Starting beam search with width={beam_width}, depth={depth}")
        return asyncio.run(
            self._beam_search_generate_async(
                trace_set, definition, selected_workload, depth, beam_width
            )
        )

    async def _beam_search_generate_async(
        self,
        trace_set: TraceSet,
        definition: Definition,
        selected_workload,
        depth: int,
        beam_width: int,
    ) -> Solution:
        passing_solutions: List[Tuple[Solution, Trace]] = []

        prompt = get_prompt(
            self.language, definition, self.target_gpu, self.use_ffi, self.destination_passing_style
        )

        print(f"\nBeam Level 0: Generating {beam_width} initial candidates...")
        code_results = await asyncio.gather(
            *[self._generate_code_from_prompt(prompt) for _ in range(beam_width)]
        )

        initial_candidates = [
            {"code": code_result["cleaned"], "raw_code": code_result["raw"], "round_num": 0}
            for code_result in code_results
        ]

        solutions = [
            self._create_solution_from_code(candidate["code"], definition, 0, candidate_idx=i)
            for i, candidate in enumerate(initial_candidates)
        ]

        print(f"Evaluating {len(solutions)} candidates...")
        traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)

        beam = []
        for i, (candidate, solution, trace) in enumerate(
            zip(initial_candidates, solutions, traces)
        ):
            if trace:
                evaluation = trace.evaluation
                speedup = (
                    evaluation.performance.speedup_factor
                    if evaluation.status == EvaluationStatus.PASSED
                    else 0.0
                )
                print(f"Candidate {i+1}: {evaluation.status.value}, speedup={speedup:.2f}x")

                if evaluation.status == EvaluationStatus.PASSED:
                    passing_solutions.append((solution, trace))

                beam.append(
                    {
                        "solution": solution,
                        "trace": trace,
                        "code": candidate["code"],
                        "raw_code": candidate["raw_code"],
                        "speedup": speedup,
                        "round_num": 0,
                    }
                )

        beam.sort(key=lambda x: x["speedup"], reverse=True)
        beam = beam[:beam_width]
        last_solution = beam[0]["solution"] if beam else None

        for level in range(1, depth + 1):
            print(f"\nBeam Level {level}/{depth}: Expanding {len(beam)} candidates...")

            prompts = [
                get_optimization_prompt(
                    self.language,
                    definition,
                    beam_item["trace"],
                    beam_item["raw_code"],
                    self.target_gpu,
                    self.use_ffi,
                    self.destination_passing_style,
                )
                for beam_item in beam
            ]

            code_results = await asyncio.gather(
                *[self._generate_code_from_prompt(prompt) for prompt in prompts]
            )

            solutions = [
                self._create_solution_from_code(
                    code_result["cleaned"], definition, level, candidate_idx=i
                )
                for i, code_result in enumerate(code_results)
            ]

            print(f"Evaluating {len(solutions)} expanded candidates...")
            traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)

            new_candidates = []
            for beam_idx, (code_result, solution, trace) in enumerate(
                zip(code_results, solutions, traces)
            ):
                if trace:
                    evaluation = trace.evaluation
                    speedup = (
                        evaluation.performance.speedup_factor
                        if evaluation.status == EvaluationStatus.PASSED
                        else 0.0
                    )
                    print(
                        f"  Candidate {beam_idx+1}: {evaluation.status.value}, speedup={speedup:.2f}x"
                    )

                    if evaluation.status == EvaluationStatus.PASSED:
                        passing_solutions.append((solution, trace))

                    new_candidates.append(
                        {
                            "solution": solution,
                            "trace": trace,
                            "code": code_result["cleaned"],
                            "raw_code": code_result["raw"],
                            "speedup": speedup,
                            "round_num": level,
                        }
                    )

            if new_candidates:
                new_candidates.sort(key=lambda x: x["speedup"], reverse=True)
                beam = new_candidates[:beam_width]
                last_solution = beam[0]["solution"]
                print(f"Beam level {level} complete. Top speedup: {beam[0]['speedup']:.2f}x")
            else:
                print(f"No valid candidates at level {level}, stopping beam search")
                break

        print(f"\nBeam search complete. Found {len(passing_solutions)} passing solutions.")
        return self._select_best_solution(passing_solutions, last_solution)

    def _evaluate_solutions(
        self,
        trace_set: TraceSet,
        definition: Definition,
        solutions: List[Solution],
        selected_workload,
    ) -> List[Optional[Trace]]:
        if not solutions:
            return []

        temp_trace_set = TraceSet(
            root=trace_set.root,
            definitions={definition.name: definition},
            solutions={definition.name: solutions},
            workloads={definition.name: [selected_workload]},
            traces={definition.name: []},
        )

        benchmark = Benchmark(temp_trace_set, BenchmarkConfig())
        result_trace_set = benchmark.run_all()

        traces = result_trace_set.traces.get(definition.name, [])

        trace_map = {trace.solution: trace for trace in traces}
        return [trace_map.get(sol.name) for sol in solutions]

    def _get_best_trace(self, passing_solutions: List[Tuple[Solution, Trace]]) -> Optional[Trace]:
        if not passing_solutions:
            return None

        best_solution_trace = max(
            passing_solutions, key=lambda st: st[1].evaluation.performance.speedup_factor
        )
        return best_solution_trace[1]

    def _select_best_solution(
        self, passing_solutions: List[Tuple[Solution, Trace]], fallback_solution: Optional[Solution]
    ) -> Solution:
        if passing_solutions:
            best_solution_trace = max(
                passing_solutions, key=lambda st: st[1].evaluation.performance.speedup_factor
            )
            best_solution = best_solution_trace[0]
            best_speedup = best_solution_trace[1].evaluation.performance.speedup_factor
            print(f"\nReturning best solution with speedup: {best_speedup:.2f}x")
            return best_solution
        elif fallback_solution:
            print(f"\nNo passing solutions found, returning last generated solution")
            return fallback_solution
        else:
            raise ValueError("No solutions generated")

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

    def _extract_code_from_markdown(self, text: str) -> str:
        code_block_pattern = r"```(?:python|triton|py|cuda|cpp|c\+\+)?\s*\n(.*?)```"

        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            extracted_code = "\n\n".join(match.strip() for match in matches)
            return extracted_code

        # detect raw code in case no markdown
        stripped = text.strip()
        python_start_patterns = [
            r"^import\s",
            r"^from\s+\w+\s+import",
            r"^#.*\n",
            r'^"""',
            r"^'''",
            r"^@",
            r"^def\s",
            r"^class\s",
        ]

        for pattern in python_start_patterns:
            if re.match(pattern, stripped):
                return stripped

        code_start_indicators = ["\nimport ", "\nfrom ", "\n@triton", "\ndef ", "\nclass "]

        for indicator in code_start_indicators:
            idx = text.find(indicator)
            if idx != -1:
                potential_code = text[idx + 1 :].strip()
                if not potential_code.endswith((".", "!", "?")):
                    return potential_code

        return stripped

    def _clean_generated_code(self, code: str) -> str:
        if self.language.lower() == "cuda":
            return self._parse_xml_files(code)

        code = self._extract_code_from_markdown(code)

        if "```" in code:
            lines = code.split("\n")
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines that are just code block markers
                if stripped == "```" or re.match(r"^```\w*$", stripped):
                    continue
                cleaned_lines.append(line)
            code = "\n".join(cleaned_lines)

        # Convert hexadecimal float literals to decimal (Triton/Python don't support them)
        code = _replace_hex_floats_in_code(code)

        return code

    async def _generate_code_from_prompt(self, prompt: str):
        """Generate code from prompt using async API"""
        try:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                response = await self.client.responses.create(
                    model=self.model_name, input=prompt, reasoning={"effort": self.reasoning_effort}
                )
                generated_code = response.output_text.strip()
            else:
                response = await self.client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}]
                )
                generated_code = response.choices[0].message.content.strip()

            cleaned_code = self._clean_generated_code(generated_code)

            return {"raw": generated_code, "cleaned": cleaned_code}

        except Exception as e:
            print(f"Error while generating code: {e}")
            raise

    def _create_solution_from_code(
        self, code, definition: Definition, round_num: int, candidate_idx: int = 0
    ) -> Solution:
        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}_{self.reasoning_effort}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx}, reasoning effort: {self.reasoning_effort})"
        else:
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx})"

        if self.language.lower() == "cuda" and isinstance(code, dict):
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_point = "main.cpp::run"
        else:
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
                destination_passing_style=self.destination_passing_style,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution
