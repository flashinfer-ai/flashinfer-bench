from __future__ import annotations

import asyncio
import html
import os
import random
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai
from kernel_generator_prompts import get_optimization_prompt, get_prompt

from flashinfer_bench.data import (
    BuildSpec,
    Definition,
    EvaluationStatus,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
    Trace,
    TraceSet,
)

if TYPE_CHECKING:
    from flashinfer_bench.bench.config import BenchmarkConfig


class KernelGenerator:
    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
        if base_url is None:
            return None

        parts = urlsplit(base_url)
        path = parts.path.rstrip("/")
        if not path:
            path = "/v1"

        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))

    def __init__(
        self,
        model_name: str,
        language: str = "triton",
        target_gpu: str = "H100",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning_effort: str = "high",  # only used for openai reasoning models
        api_mode: str = "auto",
        temperature: Optional[float] = None,
        use_ffi: bool = True,
        feedback_benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        """
        Args:
            model_name: Name of the model to use (e.g., "gpt-5")
            language: Programming language for code generation (default: "triton")
            target_gpu: Target GPU architecture (e.g., "H100", "B200", "RTX4090", default: "H100")
            api_key: API key (if None, uses LLM_API_KEY environment variable)
            base_url: Base URL for the API (need to provide for non-openai api models)
            reasoning_effort: Reasoning effort for OpenAI reasoning models ("low", "medium", "high", default: "medium")
            api_mode: LLM API mode ("auto", "chat", "responses")
            temperature: Optional temperature passed to chat.completions requests
            use_ffi: Use FFI bindings when generating CUDA kernels.
            feedback_benchmark_config: BenchmarkConfig used for inner-loop generation feedback.
        """
        self.model_name = model_name
        self.language = language
        self.target_gpu = target_gpu
        self.base_url = self._normalize_base_url(base_url)
        self.reasoning_effort = reasoning_effort
        self.api_mode = api_mode
        self.temperature = temperature
        self.use_ffi = use_ffi
        if feedback_benchmark_config is None:
            from flashinfer_bench.bench.config import BenchmarkConfig

            feedback_benchmark_config = BenchmarkConfig()
        self.feedback_benchmark_config = feedback_benchmark_config

        if self.api_mode not in {"auto", "chat", "responses"}:
            raise ValueError(f"Unsupported api_mode '{self.api_mode}'")

        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided or set in LLM_API_KEY environment variable"
                )

        client_kwargs = {"api_key": api_key}
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.sync_client = openai.OpenAI(**client_kwargs)

    def _is_reasoning_model(self) -> bool:
        return self.model_name.startswith("gpt-5") or self.model_name.startswith("o3")

    def _resolve_api_mode(self) -> str:
        if self.api_mode != "auto":
            return self.api_mode
        if self.base_url is not None:
            return "chat"
        if self._is_reasoning_model():
            return "responses"
        return "chat"

    def list_models(self) -> List[Dict[str, Any]]:
        """Return provider model metadata via the OpenAI-compatible /models endpoint."""
        response = self.sync_client.models.list()
        data = getattr(response, "data", response)
        models: List[Dict[str, Any]] = []
        for item in data:
            if hasattr(item, "model_dump"):
                model = item.model_dump()
            elif isinstance(item, dict):
                model = dict(item)
            else:
                model = {"id": getattr(item, "id", str(item))}
            models.append(model)
        models.sort(key=lambda model: model.get("id", ""))
        return models

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
        selected_workload: Optional[Trace] = None,
        workload_uuid: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Solution:
        """
        Generate an optimized solution through iterative improvement using flashinfer-bench feedback.

        Args:
            trace_set: The TraceSet containing workloads for evaluation
            definition: The workload definition to implement kernel for
            gen_rounds: Number of generation rounds to run (or search depth if beam=True)
            beam: beam search flag, default to False as it's more expensive to run
            beam_width: Number of candidates to maintain in beam search (default: 3)
            selected_workload: Optional explicit workload trace for inner-loop optimization
            workload_uuid: Optional workload UUID to deterministically choose a workload
            random_seed: Optional RNG seed used when selecting a random workload

        Returns:
            Solution: a solution dataclass containing the optimized kernel code
        """
        workloads = trace_set.workloads.get(definition.name, [])
        if not workloads:
            raise ValueError(
                f"No workloads found for definition '{definition.name}' in the provided TraceSet"
            )

        if selected_workload is None and workload_uuid is not None:
            selected_workload = next(
                (workload for workload in workloads if workload.workload.uuid == workload_uuid), None
            )
            if selected_workload is None:
                raise ValueError(
                    f"Workload UUID '{workload_uuid}' not found for definition '{definition.name}'"
                )

        if selected_workload is None:
            chooser = random.Random(random_seed) if random_seed is not None else random
            selected_workload = chooser.choice(workloads)

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
        prompt = get_prompt(self.language, definition, self.target_gpu, self.use_ffi)
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
                    )
                else:
                    optimization_prompt = get_prompt(
                        self.language, definition, self.target_gpu, self.use_ffi
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

        prompt = get_prompt(self.language, definition, self.target_gpu, self.use_ffi)

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

        from flashinfer_bench.bench.benchmark import Benchmark

        benchmark = Benchmark(temp_trace_set, self.feedback_benchmark_config)
        try:
            result_trace_set = benchmark.run_all(dump_traces=False)
        finally:
            benchmark.close()

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

    def _normalize_source_filename(self, filename: str) -> str:
        return filename.strip().strip("`'\"").replace("\\", "/").removeprefix("./")

    def _is_cuda_source_filename(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in {".h", ".hpp", ".cuh", ".cu", ".cpp", ".cc", ".cxx"}

    def _strip_wrapping_code_fence(self, content: str) -> str:
        stripped = content.strip()
        fence_pattern = re.compile(r"```[^\n`]*\n(?P<body>.*?)(?:\n)?```", re.DOTALL)

        while True:
            match = fence_pattern.fullmatch(stripped)
            if not match:
                return stripped
            stripped = match.group("body").strip()

    def _store_cuda_file(self, files: Dict[str, str], filename: str, content: str) -> None:
        normalized_name = self._normalize_source_filename(filename)
        normalized_content = self._strip_wrapping_code_fence(content)
        if not normalized_name or not normalized_content:
            return
        if not self._is_cuda_source_filename(normalized_name):
            return
        files.setdefault(normalized_name, normalized_content)

    def _extract_filename_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None

        patterns = (
            r"""(?:filename|file|path|name|title)\s*[:=]\s*["'`]?([A-Za-z0-9_./-]+\.(?:h|hpp|cuh|cu|cpp|cc|cxx))["'`]?""",
            r"""([A-Za-z0-9_./-]+\.(?:h|hpp|cuh|cu|cpp|cc|cxx))""",
        )

        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                return self._normalize_source_filename(matches[-1])
        return None

    def _extract_filename_from_prefix(self, prefix: str) -> Optional[str]:
        if not prefix:
            return None

        candidate_lines = []
        for line in reversed(prefix.splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith("```"):
                continue
            candidate_lines.append(stripped)
            if len(candidate_lines) == 3:
                break

        candidate_lines.reverse()
        return self._extract_filename_from_text("\n".join(candidate_lines))

    def _guess_cuda_filename(self, info: str, content: str, existing_files: Dict[str, str]) -> Optional[str]:
        lowered = f"{info}\n{content}".lower()
        if "#pragma once" in lowered or ("#ifndef" in lowered and "#define" in lowered):
            return "kernel.h"
        if "__global__" in lowered or "__device__" in lowered or "__host__" in lowered:
            return "kernel.cu"
        if "pybind11_module" in lowered or "torch/extension.h" in lowered:
            return "main.cpp"
        if re.search(r"\brun\s*\(", content):
            return "main.cpp"
        if "```cuda" in lowered or info.strip().lower() == "cuda":
            return "kernel.cu"
        if any(token in info.lower() for token in ("cpp", "c++", "cc", "cxx")):
            return "main.cpp"
        if existing_files:
            return None
        if "cuda" in lowered:
            return "kernel.cu"
        return "main.cpp"

    def _parse_xml_files(self, code: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        xml_like_code = html.unescape(code)
        pattern = re.compile(
            r"<(?P<tag>[A-Za-z_][\w:-]*)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>",
            re.DOTALL | re.IGNORECASE,
        )

        for match in pattern.finditer(xml_like_code):
            attrs = match.group("attrs")
            filename_match = re.search(
                r"""\b(?:name|path|filename)\s*=\s*["'](?P<filename>[^"']+)["']""",
                attrs,
                flags=re.IGNORECASE,
            )
            if not filename_match:
                continue
            self._store_cuda_file(files, filename_match.group("filename"), match.group("body"))

        return files

    def _parse_markdown_files(self, code: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        fence_pattern = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)(?:\n)?```", re.DOTALL)

        for match in fence_pattern.finditer(code):
            info = match.group("info").strip()
            body = match.group("body")
            prefix = code[max(0, match.start() - 200) : match.start()]

            filename = self._extract_filename_from_text(info)
            if filename is None:
                filename = self._extract_filename_from_prefix(prefix)
            if filename is None:
                filename = self._guess_cuda_filename(info, body, files)
            if filename is not None:
                self._store_cuda_file(files, filename, body)

        return files

    def _select_cuda_entry_source(self, sources: List[SourceFile]) -> str:
        preferred_names = (
            "main.cpp",
            "main.cc",
            "main.cxx",
            "binding.cpp",
            "binding.cc",
            "binding.cxx",
            "launcher.cpp",
            "host.cpp",
            "main.cu",
        )
        host_suffixes = {".cpp", ".cc", ".cxx", ".cu"}

        for preferred_name in preferred_names:
            for source in sources:
                if source.path == preferred_name:
                    return source.path

        for source in sources:
            if Path(source.path).suffix.lower() in host_suffixes and re.search(
                r"\brun\s*\(", source.content
            ):
                return source.path

        for source in sources:
            if Path(source.path).suffix.lower() in host_suffixes:
                return source.path

        return sources[0].path

    def _clean_generated_code(self, code: str):
        if self.language.lower() == "cuda":
            files = self._parse_xml_files(code)
            for filename, content in self._parse_markdown_files(code).items():
                files.setdefault(filename, content)

            if files:
                return files

            fallback_filename = self._guess_cuda_filename("", code, files) or "main.cpp"
            fallback_body = self._strip_wrapping_code_fence(code)
            if fallback_body:
                print(
                    f"Warning: Could not identify structured CUDA files in generated code; "
                    f"treating response as {fallback_filename}"
                )
                return {fallback_filename: fallback_body}

            print("Warning: Could not identify structured CUDA files in generated code")
            return {}

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

    async def _generate_code_from_prompt(self, prompt: str):
        """Generate code from prompt using async API"""
        try:
            api_mode = self._resolve_api_mode()
            if api_mode == "responses":
                response = await self.client.responses.create(
                    model=self.model_name, input=prompt, reasoning={"effort": self.reasoning_effort}
                )
                generated_code = response.output_text.strip()
            else:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                response = await self.client.chat.completions.create(**request_kwargs)
                if isinstance(response, str):
                    response_prefix = response.strip().replace("\n", " ")[:200]
                    raise RuntimeError(
                        "Provider returned plain text/HTML instead of an OpenAI-compatible "
                        f"chat completion payload. base_url={self.base_url!r}. "
                        "If you passed a bare domain, use the API root ending in '/v1'. "
                        f"Response prefix: {response_prefix!r}"
                    )
                if not hasattr(response, "choices"):
                    raise RuntimeError(
                        "Provider returned an unexpected chat completion response type: "
                        f"{type(response).__name__}"
                    )
                message_content = response.choices[0].message.content
                if message_content is None:
                    raise RuntimeError("Provider returned an empty assistant message.")
                generated_code = message_content.strip()

            cleaned_code = self._clean_generated_code(generated_code)

            return {"raw": generated_code, "cleaned": cleaned_code}

        except Exception as e:
            print(f"Error while generating code: {e}")
            raise

    def _create_solution_from_code(
        self, code, definition: Definition, round_num: int, candidate_idx: int = 0
    ) -> Solution:
        if self._is_reasoning_model():
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}_{self.reasoning_effort}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx}, reasoning effort: {self.reasoning_effort})"
        else:
            solution_name = f"{self.model_name}_{definition.name}_{self.language}_optimized_r{round_num}_c{candidate_idx}"
            solution_description = f"{self.model_name} optimized kernel for {definition.name} (round {round_num}, candidate {candidate_idx})"

        binding = None
        destination_passing_style = False
        if self.language.lower() == "cuda" and isinstance(code, dict):
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_source = self._select_cuda_entry_source(sources)
            entry_point = f"{entry_source}::run"
            binding = SupportedBindings.TVM_FFI if self.use_ffi else SupportedBindings.TORCH
            destination_passing_style = self.use_ffi
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
                destination_passing_style=destination_passing_style,
                binding=binding,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution
