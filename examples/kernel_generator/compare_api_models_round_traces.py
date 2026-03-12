"""Batch-compare API models and persist per-round generation artifacts."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_api_models import (
    _benchmark_solution,
    _build_feedback_config,
    _build_final_config,
    _discover_models,
    _load_api_key,
    _make_experiment_row,
    _make_trace_row,
    _pick_feedback_workload,
    _print_summary,
    _safe_path_segment,
    _save_solution,
    _select_models,
    _summarize_errors,
    _write_csv,
    build_parser as _base_build_parser,
)
from kernel_generator import KernelGenerator
from kernel_generator_prompts import get_optimization_prompt, get_prompt

from flashinfer_bench.bench.error_taxonomy import classify_trace
from flashinfer_bench.data import EvaluationStatus, Solution, Trace, TraceSet, save_json_file

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local convenience
    load_dotenv = None


class GenerationRoundRecorder:
    def __init__(
        self,
        run_dir: Path,
        model_name: str,
        definition_name: str,
        feedback_workload_uuid: str,
    ) -> None:
        self.run_dir = run_dir
        self.model_name = model_name
        self.definition_name = definition_name
        self.feedback_workload_uuid = feedback_workload_uuid
        self.round_root = (
            run_dir
            / "generation_rounds"
            / _safe_path_segment(model_name)
            / _safe_path_segment(definition_name)
        )
        self.rows: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        generation_mode: str,
        generation_round: int,
        candidate_index: int,
        solution: Solution,
        prompt: str,
        raw_response: str,
        feedback_trace: Optional[Trace],
        parent_solution_name: str = "",
        feedback_error: str = "",
    ) -> None:
        artifact_dir = (
            self.round_root
            / f"{generation_mode}_r{generation_round:02d}_c{candidate_index:02d}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        solution_path = artifact_dir / "solution.json"
        save_json_file(solution, solution_path)

        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        raw_response_path = artifact_dir / "raw_response.txt"
        raw_response_path.write_text(raw_response, encoding="utf-8")

        feedback_trace_path = ""
        if feedback_trace is not None:
            trace_path = artifact_dir / "feedback_trace.json"
            save_json_file(feedback_trace, trace_path)
            feedback_trace_path = str(trace_path)

        self.rows.append(
            _make_generation_round_row(
                model_name=self.model_name,
                definition_name=self.definition_name,
                solution=solution,
                feedback_workload_uuid=self.feedback_workload_uuid,
                generation_mode=generation_mode,
                generation_round=generation_round,
                candidate_index=candidate_index,
                parent_solution_name=parent_solution_name,
                feedback_trace=feedback_trace,
                solution_path=str(solution_path),
                feedback_trace_path=feedback_trace_path,
                prompt_path=str(prompt_path),
                raw_response_path=str(raw_response_path),
                feedback_error=feedback_error,
            )
        )


def _make_generation_round_row(
    *,
    model_name: str,
    definition_name: str,
    solution: Solution,
    feedback_workload_uuid: str,
    generation_mode: str,
    generation_round: int,
    candidate_index: int,
    parent_solution_name: str,
    feedback_trace: Optional[Trace],
    solution_path: str,
    feedback_trace_path: str,
    prompt_path: str,
    raw_response_path: str,
    feedback_error: str,
) -> Dict[str, Any]:
    evaluation = feedback_trace.evaluation if feedback_trace is not None else None
    taxonomy = (
        classify_trace(feedback_trace)
        if feedback_trace is not None and feedback_trace.evaluation is not None
        else None
    )
    correctness = evaluation.correctness if evaluation is not None else None
    performance = evaluation.performance if evaluation is not None else None

    if evaluation is not None:
        status = evaluation.status.value
        status_family = taxonomy.status_family if taxonomy is not None else ""
        secondary_bucket = taxonomy.secondary_bucket if taxonomy is not None else ""
        efficiency_bucket = taxonomy.efficiency_bucket if taxonomy is not None else ""
        log_excerpt = (evaluation.log or "")[:240].replace("\n", "\\n")
    elif feedback_error:
        status = "FEEDBACK_ERROR"
        status_family = "feedback_error"
        secondary_bucket = "feedback.exception"
        efficiency_bucket = ""
        log_excerpt = feedback_error[:240].replace("\n", "\\n")
    else:
        status = "MISSING"
        status_family = ""
        secondary_bucket = ""
        efficiency_bucket = ""
        log_excerpt = ""

    return {
        "model": model_name,
        "definition": definition_name,
        "solution": solution.name,
        "feedback_workload_uuid": feedback_workload_uuid,
        "generation_mode": generation_mode,
        "generation_round": generation_round,
        "candidate_index": candidate_index,
        "parent_solution": parent_solution_name,
        "status": status,
        "status_family": status_family,
        "secondary_bucket": secondary_bucket,
        "efficiency_bucket": efficiency_bucket,
        "max_absolute_error": correctness.max_absolute_error if correctness is not None else "",
        "max_relative_error": correctness.max_relative_error if correctness is not None else "",
        "latency_ms": performance.latency_ms if performance is not None else "",
        "reference_latency_ms": (
            performance.reference_latency_ms if performance is not None else ""
        ),
        "speedup_factor": performance.speedup_factor if performance is not None else "",
        "solution_path": solution_path,
        "feedback_trace_path": feedback_trace_path,
        "prompt_path": prompt_path,
        "raw_response_path": raw_response_path,
        "feedback_error": feedback_error,
        "log_excerpt": log_excerpt,
    }


class TracingKernelGenerator(KernelGenerator):
    def __init__(
        self,
        *args: Any,
        round_recorder: Optional[GenerationRoundRecorder] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.round_recorder = round_recorder

    async def _sequential_generate_async(
        self,
        trace_set: TraceSet,
        definition,
        selected_workload,
        gen_rounds: int,
    ) -> Solution:
        prompt = get_prompt(self.language, definition, self.target_gpu, self.use_ffi)
        code_result = await self._generate_code_from_prompt(prompt)
        current_code = code_result["cleaned"]
        current_raw_code = code_result["raw"]
        current_prompt = prompt

        passing_solutions: List[Tuple[Solution, Trace]] = []
        last_solution = None
        last_trace = None

        for round_num in range(1, gen_rounds + 1):
            print(f"\nGeneration Round {round_num}/{gen_rounds}")

            solution = self._create_solution_from_code(current_code, definition, round_num)
            last_solution = solution
            feedback_trace = None

            try:
                traces = self._evaluate_solutions(
                    trace_set, definition, [solution], selected_workload
                )
                feedback_trace = traces[0] if traces else None
            except Exception as exc:
                if self.round_recorder is not None:
                    self.round_recorder.record(
                        generation_mode="sequential",
                        generation_round=round_num,
                        candidate_index=0,
                        solution=solution,
                        prompt=current_prompt,
                        raw_response=current_raw_code,
                        feedback_trace=None,
                        parent_solution_name=last_trace.solution if last_trace is not None else "",
                        feedback_error=str(exc),
                    )
                raise

            if self.round_recorder is not None:
                self.round_recorder.record(
                    generation_mode="sequential",
                    generation_round=round_num,
                    candidate_index=0,
                    solution=solution,
                    prompt=current_prompt,
                    raw_response=current_raw_code,
                    feedback_trace=feedback_trace,
                    parent_solution_name=last_trace.solution if last_trace is not None else "",
                )

            if feedback_trace is not None and feedback_trace.evaluation is not None:
                last_trace = feedback_trace
                evaluation = feedback_trace.evaluation
                print(f"Evaluation status: {evaluation.status.value}")

                if evaluation.status == EvaluationStatus.PASSED:
                    speedup = evaluation.performance.speedup_factor
                    print(f"Solution PASSED! Speedup: {speedup:.2f}x")
                    passing_solutions.append((solution, feedback_trace))
                else:
                    print(f"Solution failed with {evaluation.status.value}")
                    if evaluation.log:
                        print("Error details:")
                        print(evaluation.log)

            if round_num < gen_rounds:
                best_trace = self._get_best_trace(passing_solutions)
                opt_trace = best_trace if best_trace else last_trace

                if opt_trace:
                    next_prompt = get_optimization_prompt(
                        self.language,
                        definition,
                        opt_trace,
                        current_raw_code,
                        self.target_gpu,
                        self.use_ffi,
                    )
                else:
                    next_prompt = get_prompt(
                        self.language, definition, self.target_gpu, self.use_ffi
                    )

                print(f"Generating code for round {round_num + 1}...")
                code_result = await self._generate_code_from_prompt(next_prompt)
                current_code = code_result["cleaned"]
                current_raw_code = code_result["raw"]
                current_prompt = next_prompt

        return self._select_best_solution(passing_solutions, last_solution)

    async def _beam_search_generate_async(
        self,
        trace_set: TraceSet,
        definition,
        selected_workload,
        depth: int,
        beam_width: int,
    ) -> Solution:
        passing_solutions: List[Tuple[Solution, Trace]] = []

        prompt = get_prompt(self.language, definition, self.target_gpu, self.use_ffi)

        print(f"\nBeam Level 0: Generating {beam_width} initial candidates...")
        code_results = await self._generate_candidates([prompt] * beam_width)

        initial_candidates = [
            {"code": code_result["cleaned"], "raw_code": code_result["raw"], "round_num": 0}
            for code_result in code_results
        ]

        solutions = [
            self._create_solution_from_code(candidate["code"], definition, 0, candidate_idx=i)
            for i, candidate in enumerate(initial_candidates)
        ]

        print(f"Evaluating {len(solutions)} candidates...")
        try:
            traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)
        except Exception as exc:
            self._record_failed_batch(
                generation_mode="beam",
                generation_round=0,
                solutions=solutions,
                prompts=[prompt] * len(solutions),
                raw_responses=[candidate["raw_code"] for candidate in initial_candidates],
                parent_solution_names=[""] * len(solutions),
                error_message=str(exc),
            )
            raise

        beam = []
        for i, (candidate, solution, trace) in enumerate(
            zip(initial_candidates, solutions, traces)
        ):
            self._record_round(
                generation_mode="beam",
                generation_round=0,
                candidate_index=i,
                solution=solution,
                prompt=prompt,
                raw_response=candidate["raw_code"],
                feedback_trace=trace,
                parent_solution_name="",
            )

            if trace is not None and trace.evaluation is not None:
                evaluation = trace.evaluation
                speedup = (
                    evaluation.performance.speedup_factor
                    if evaluation.status == EvaluationStatus.PASSED
                    else 0.0
                )
                print(f"Candidate {i + 1}: {evaluation.status.value}, speedup={speedup:.2f}x")

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

            code_results = await self._generate_candidates(prompts)
            solutions = [
                self._create_solution_from_code(
                    code_result["cleaned"], definition, level, candidate_idx=i
                )
                for i, code_result in enumerate(code_results)
            ]

            print(f"Evaluating {len(solutions)} expanded candidates...")
            try:
                traces = self._evaluate_solutions(trace_set, definition, solutions, selected_workload)
            except Exception as exc:
                self._record_failed_batch(
                    generation_mode="beam",
                    generation_round=level,
                    solutions=solutions,
                    prompts=prompts,
                    raw_responses=[code_result["raw"] for code_result in code_results],
                    parent_solution_names=[beam_item["solution"].name for beam_item in beam],
                    error_message=str(exc),
                )
                raise

            new_candidates = []
            for beam_idx, (code_result, solution, trace, parent) in enumerate(
                zip(code_results, solutions, traces, beam)
            ):
                self._record_round(
                    generation_mode="beam",
                    generation_round=level,
                    candidate_index=beam_idx,
                    solution=solution,
                    prompt=prompts[beam_idx],
                    raw_response=code_result["raw"],
                    feedback_trace=trace,
                    parent_solution_name=parent["solution"].name,
                )

                if trace is not None and trace.evaluation is not None:
                    evaluation = trace.evaluation
                    speedup = (
                        evaluation.performance.speedup_factor
                        if evaluation.status == EvaluationStatus.PASSED
                        else 0.0
                    )
                    print(
                        f"  Candidate {beam_idx + 1}: {evaluation.status.value}, "
                        f"speedup={speedup:.2f}x"
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

    async def _generate_candidates(self, prompts: List[str]) -> List[Dict[str, Any]]:
        import asyncio

        return await asyncio.gather(*[self._generate_code_from_prompt(prompt) for prompt in prompts])

    def _record_failed_batch(
        self,
        *,
        generation_mode: str,
        generation_round: int,
        solutions: List[Solution],
        prompts: List[str],
        raw_responses: List[str],
        parent_solution_names: List[str],
        error_message: str,
    ) -> None:
        for idx, (solution, prompt, raw_response, parent_solution_name) in enumerate(
            zip(solutions, prompts, raw_responses, parent_solution_names)
        ):
            self._record_round(
                generation_mode=generation_mode,
                generation_round=generation_round,
                candidate_index=idx,
                solution=solution,
                prompt=prompt,
                raw_response=raw_response,
                feedback_trace=None,
                parent_solution_name=parent_solution_name,
                feedback_error=error_message,
            )

    def _record_round(
        self,
        *,
        generation_mode: str,
        generation_round: int,
        candidate_index: int,
        solution: Solution,
        prompt: str,
        raw_response: str,
        feedback_trace: Optional[Trace],
        parent_solution_name: str,
        feedback_error: str = "",
    ) -> None:
        if self.round_recorder is None:
            return
        self.round_recorder.record(
            generation_mode=generation_mode,
            generation_round=generation_round,
            candidate_index=candidate_index,
            solution=solution,
            prompt=prompt,
            raw_response=raw_response,
            feedback_trace=feedback_trace,
            parent_solution_name=parent_solution_name,
            feedback_error=feedback_error,
        )


def build_parser():
    parser = _base_build_parser()
    parser.description = (
        "Compare OpenAI-compatible API models on FlashInfer-Bench CUDA code generation "
        "and save per-round generation feedback artifacts."
    )
    return parser


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    args = build_parser().parse_args()
    api_key = _load_api_key(args.api_key_env)
    trace_root = Path(args.trace_set_path).resolve()
    trace_set = TraceSet.from_path(str(trace_root))

    discovered_models: List[Dict[str, Any]] = []
    if args.list_models or args.model_prefixes:
        discovered_models = _discover_models(api_key, args.base_url, args)

    if args.list_models:
        for model in discovered_models:
            created = model.get("created", "")
            owned_by = model.get("owned_by", "")
            print(f"{model.get('id', '')}\tcreated={created}\towned_by={owned_by}")
        return

    selected_models = _select_models(args, discovered_models)
    if not selected_models:
        raise RuntimeError("No models selected")

    definition_names = list(args.definitions) if args.definitions else sorted(trace_set.definitions)
    missing_definitions = [name for name in definition_names if name not in trace_set.definitions]
    if missing_definitions:
        raise ValueError(f"Definitions not found: {missing_definitions}")

    feedback_workloads: Dict[str, Trace] = {}
    for idx, definition_name in enumerate(definition_names):
        feedback_workloads[definition_name] = _pick_feedback_workload(
            definition_name,
            trace_set.workloads.get(definition_name, []),
            workload_mode=args.workload_mode,
            random_seed=args.seed + idx,
            workload_uuid=args.workload_uuid,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).resolve() / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "trace_set_path": str(trace_root),
        "base_url": args.base_url,
        "api_mode": args.api_mode,
        "language": args.language,
        "target_gpu": args.target_gpu,
        "models": selected_models,
        "definitions": definition_names,
        "workload_mode": args.workload_mode,
        "workload_uuid": args.workload_uuid,
        "save_traces": not args.no_save_traces,
        "save_generation_round_artifacts": True,
        "generation_round_artifacts_dir": "generation_rounds",
        "feedback_benchmark": _build_feedback_config(args).__dict__,
        "final_benchmark": _build_final_config(args).__dict__,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if discovered_models:
        with open(run_dir / "available_models.json", "w", encoding="utf-8") as f:
            json.dump(discovered_models, f, indent=2, ensure_ascii=False)

    experiment_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []

    for model_name in selected_models:
        for definition_name in definition_names:
            definition = trace_set.definitions[definition_name]
            feedback_workload = feedback_workloads[definition_name]
            recorder = GenerationRoundRecorder(
                run_dir=run_dir,
                model_name=model_name,
                definition_name=definition_name,
                feedback_workload_uuid=feedback_workload.workload.uuid,
            )
            generator = TracingKernelGenerator(
                model_name=model_name,
                language=args.language,
                target_gpu=args.target_gpu,
                api_key=api_key,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                api_mode=args.api_mode,
                temperature=args.temperature,
                use_ffi=args.use_ffi,
                feedback_benchmark_config=_build_feedback_config(args),
                round_recorder=recorder,
            )

            print(
                f"\n[{model_name}] generating {definition_name} "
                f"using feedback workload {feedback_workload.workload.uuid}"
            )

            try:
                solution = generator.generate(
                    trace_set=trace_set,
                    definition=definition,
                    gen_rounds=args.gen_rounds,
                    beam=args.beam,
                    beam_width=args.beam_width,
                    selected_workload=feedback_workload,
                    random_seed=args.seed,
                )
                solution_path = _save_solution(
                    trace_root,
                    definition_name,
                    definition.op_type,
                    solution,
                )
                traces = _benchmark_solution(
                    trace_set=trace_set,
                    definition_name=definition_name,
                    solution=solution,
                    config=_build_final_config(args),
                    save_traces=not args.no_save_traces,
                )
                experiment_rows.append(
                    _make_experiment_row(
                        model_name=model_name,
                        definition_name=definition_name,
                        solution_name=solution.name,
                        feedback_workload_uuid=feedback_workload.workload.uuid,
                        traces=traces,
                        experiment_status="OK",
                    )
                    | {"solution_path": str(solution_path)}
                )
                for trace in traces:
                    trace_rows.append(
                        _make_trace_row(
                            model_name=model_name,
                            definition_name=definition_name,
                            solution_name=solution.name,
                            feedback_workload_uuid=feedback_workload.workload.uuid,
                            trace=trace,
                        )
                    )
            except Exception as exc:
                error_message = str(exc)
                print(f"[error] {model_name} / {definition_name}: {error_message}")
                experiment_rows.append(
                    _make_experiment_row(
                        model_name=model_name,
                        definition_name=definition_name,
                        solution_name="",
                        feedback_workload_uuid=feedback_workload.workload.uuid,
                        traces=[],
                        experiment_status="FAILED",
                        error_message=error_message,
                    )
                )
                if args.fail_fast:
                    raise
            finally:
                generation_rows.extend(recorder.rows)

    error_rows = _summarize_errors(trace_rows)

    _write_csv(
        run_dir / "experiment_summary.csv",
        experiment_rows,
        fieldnames=[
            "model",
            "definition",
            "solution",
            "solution_path",
            "feedback_workload_uuid",
            "experiment_status",
            "total_traces",
            "passed_traces",
            "compile_errors",
            "runtime_errors",
            "correctness_errors",
            "timeouts",
            "pass_rate",
            "best_speedup",
            "avg_speedup",
            "top_error_bucket",
            "error_message",
        ],
    )
    _write_csv(
        run_dir / "trace_records.csv",
        trace_rows,
        fieldnames=[
            "model",
            "definition",
            "solution",
            "feedback_workload_uuid",
            "benchmark_workload_uuid",
            "status",
            "status_family",
            "secondary_bucket",
            "efficiency_bucket",
            "max_absolute_error",
            "max_relative_error",
            "latency_ms",
            "reference_latency_ms",
            "speedup_factor",
            "log_excerpt",
        ],
    )
    _write_csv(
        run_dir / "error_summary.csv",
        error_rows,
        fieldnames=["model", "definition", "secondary_bucket", "count"],
    )
    _write_csv(
        run_dir / "generation_feedback_records.csv",
        generation_rows,
        fieldnames=[
            "model",
            "definition",
            "solution",
            "feedback_workload_uuid",
            "generation_mode",
            "generation_round",
            "candidate_index",
            "parent_solution",
            "status",
            "status_family",
            "secondary_bucket",
            "efficiency_bucket",
            "max_absolute_error",
            "max_relative_error",
            "latency_ms",
            "reference_latency_ms",
            "speedup_factor",
            "solution_path",
            "feedback_trace_path",
            "prompt_path",
            "raw_response_path",
            "feedback_error",
            "log_excerpt",
        ],
    )

    _print_summary(experiment_rows)
    print("\nGeneration feedback records written to:", run_dir / "generation_feedback_records.csv")
    print("Generation round artifacts written to:", run_dir / "generation_rounds")
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
