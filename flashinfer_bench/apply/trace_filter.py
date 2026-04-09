"""Helpers for building filtered TraceSets for apply experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from flashinfer_bench.data import Solution, TraceSet
from flashinfer_bench.data.trace import EvaluationStatus

_BASELINE_AUTHORS = {"pytorch", "flashinfer", "baseline", "__builtin__"}


def is_baseline_solution(solution: Solution) -> bool:
    """Return whether a solution should be treated as baseline/reference."""

    return solution.author.strip().lower() in _BASELINE_AUTHORS


def hardware_matches(hardware: str, hardware_filters: Sequence[str]) -> bool:
    """Return whether a hardware string matches any requested filter token."""

    if not hardware_filters:
        return True
    hardware_norm = hardware.casefold()
    return any(token.casefold() in hardware_norm for token in hardware_filters if token.strip())


def solution_language_matches(language: str, language_filters: Sequence[str]) -> bool:
    """Return whether a solution language matches any requested filter token."""

    if not language_filters:
        return True
    language_norm = language.casefold()
    return any(token.casefold() in language_norm for token in language_filters if token.strip())


def _solution_language(solution: Solution) -> str:
    language = solution.spec.language
    return str(getattr(language, "value", language))


def collect_solution_names_by_pool(
    trace_set: TraceSet,
    *,
    pool: str,
    definition_names: Optional[Sequence[str]] = None,
    solution_language_filters: Optional[Sequence[str]] = None,
) -> List[str]:
    """Collect solution names from the requested pool."""

    if pool not in {"all", "generated_only", "baseline_only"}:
        raise ValueError(f"Unsupported solution pool: {pool}")

    defs_to_scan = definition_names or list(trace_set.definitions.keys())
    normalized_language_filters = [
        value.strip() for value in solution_language_filters or [] if value.strip()
    ]
    names: List[str] = []
    for def_name in defs_to_scan:
        for solution in trace_set.solutions.get(def_name, []):
            if normalized_language_filters and not solution_language_matches(
                _solution_language(solution),
                normalized_language_filters,
            ):
                continue
            baseline = is_baseline_solution(solution)
            if pool == "generated_only" and baseline:
                continue
            if pool == "baseline_only" and not baseline:
                continue
            names.append(solution.name)
    return names


def build_filtered_trace_set(
    trace_set: TraceSet,
    *,
    solution_names: Optional[Sequence[str]] = None,
    definition_names: Optional[Sequence[str]] = None,
    trace_hardware_filters: Optional[Sequence[str]] = None,
    solution_language_filters: Optional[Sequence[str]] = None,
) -> TraceSet:
    """Return an in-memory TraceSet filtered to selected definitions/solutions."""

    selected_solution_names: Optional[Set[str]] = set(solution_names) if solution_names else None
    selected_definition_names: Set[str] = set(definition_names or [])
    normalized_hardware_filters = [value.strip() for value in trace_hardware_filters or [] if value.strip()]
    normalized_language_filters = [value.strip() for value in solution_language_filters or [] if value.strip()]

    if selected_solution_names:
        missing = [name for name in selected_solution_names if trace_set.get_solution(name) is None]
        if missing:
            raise ValueError(f"Unknown solution name(s): {sorted(missing)}")
        for sol_name in selected_solution_names:
            solution = trace_set.get_solution(sol_name)
            assert solution is not None
            selected_definition_names.add(solution.definition)

    if not selected_definition_names:
        if selected_solution_names or normalized_hardware_filters or normalized_language_filters:
            selected_definition_names = set(trace_set.definitions.keys())
        else:
            return trace_set

    missing_defs = [def_name for def_name in selected_definition_names if def_name not in trace_set.definitions]
    if missing_defs:
        raise ValueError(f"Unknown definition name(s): {sorted(missing_defs)}")

    filtered_solutions: Dict[str, List] = {}
    filtered_traces: Dict[str, List] = {}
    filtered_workloads: Dict[str, List] = {}

    for def_name in sorted(selected_definition_names):
        traces = list(trace_set.traces.get(def_name, []))
        if selected_solution_names is not None:
            traces = [trace for trace in traces if trace.solution in selected_solution_names]
        if normalized_hardware_filters:
            traces = [
                trace
                for trace in traces
                if trace.evaluation is not None
                and hardware_matches(trace.evaluation.environment.hardware, normalized_hardware_filters)
            ]

        trace_solution_names = {trace.solution for trace in traces if trace.solution}
        solutions = list(trace_set.solutions.get(def_name, []))
        if selected_solution_names is not None:
            solutions = [sol for sol in solutions if sol.name in selected_solution_names]
        if normalized_language_filters:
            solutions = [
                sol
                for sol in solutions
                if solution_language_matches(_solution_language(sol), normalized_language_filters)
            ]
        if normalized_hardware_filters:
            solutions = [sol for sol in solutions if sol.name in trace_solution_names]
        if solutions:
            filtered_solutions[def_name] = solutions

        allowed_solution_names = {sol.name for sol in solutions}
        if allowed_solution_names:
            traces = [trace for trace in traces if trace.solution in allowed_solution_names]
        else:
            traces = []
        if traces:
            filtered_traces[def_name] = traces

        workloads = list(trace_set.workloads.get(def_name, []))
        if workloads:
            filtered_workloads[def_name] = workloads

    return TraceSet(
        root=None,
        definitions={
            def_name: trace_set.definitions[def_name] for def_name in sorted(selected_definition_names)
        },
        solutions=filtered_solutions,
        workloads=filtered_workloads,
        traces=filtered_traces,
    )


def count_eligible_traces_by_solution(
    trace_set: TraceSet,
    *,
    solution_names: Sequence[str],
    max_atol: float,
    max_rtol: float,
    trace_hardware_filters: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Count PASSED traces meeting error tolerances for each selected solution."""

    selected = set(solution_names)
    counts: Dict[str, int] = {name: 0 for name in selected}
    normalized_hardware_filters = [value.strip() for value in trace_hardware_filters or [] if value.strip()]
    for traces in trace_set.traces.values():
        for trace in traces:
            if trace.solution not in selected:
                continue
            if trace.evaluation is None or trace.evaluation.status != EvaluationStatus.PASSED:
                continue
            if normalized_hardware_filters and not hardware_matches(
                trace.evaluation.environment.hardware,
                normalized_hardware_filters,
            ):
                continue
            correctness = trace.evaluation.correctness
            if correctness is None:
                continue
            if correctness.max_absolute_error > max_atol or correctness.max_relative_error > max_rtol:
                continue
            counts[trace.solution] += 1
    return counts


def collect_definition_names_from_solutions(
    trace_set: TraceSet, solution_names: Iterable[str]
) -> Dict[str, str]:
    """Map solution name to definition name, raising when a solution is missing."""

    mapping: Dict[str, str] = {}
    for sol_name in solution_names:
        solution = trace_set.get_solution(sol_name)
        if solution is None:
            raise ValueError(f"Unknown solution name: {sol_name}")
        mapping[sol_name] = solution.definition
    return mapping
