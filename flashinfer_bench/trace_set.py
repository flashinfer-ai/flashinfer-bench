import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from flashinfer_bench.utils.json_utils import load_jsonl

from .definition import Definition
from .solution import Solution
from .trace import Trace


@dataclass
class TraceSet:
    """A collection of definitions, solutions, workloads, and traces."""

    definitions: Dict[str, Definition]
    solutions: Dict[str, List[Solution]]
    workload: Dict[str, List[Trace]]
    traces: Dict[str, List[Trace]]
    _implementation_callables: Dict[str, Callable] = field(init=False, default_factory=dict)

    @classmethod
    def from_path(cls, path: str) -> "TraceSet":
        base_path = Path(path)

        # Load definitions with uniqueness check
        definitions = {}
        definitions_files = glob.glob(str(base_path / "definitions" / "*.json"), recursive=True)
        for definition_file in definitions_files:
            with open(definition_file, "r") as f:
                definition = Definition.from_json(f.read())
                if definition.name in definitions:
                    raise ValueError(f"Duplicate definition name: {definition.name}")
                definitions[definition.name] = definition

        # Load solutions grouped by definition
        solutions = {}
        solutions_files = glob.glob(str(base_path / "solutions" / "*.json"), recursive=True)
        for solution_file in solutions_files:
            with open(solution_file, "r") as f:
                solution = Solution.from_json(f.read())
                if solution.definition not in solutions:
                    solutions[solution.definition] = []
                solutions[solution.definition].append(solution)

        # Load workload traces grouped by definition
        workload = {}
        trace_files = glob.glob(str(base_path / "traces" / "**" / "*.jsonl"), recursive=True)
        for trace_file in trace_files:
            loaded_traces = load_jsonl(trace_file, Trace)
            for trace in cast(List[Trace], loaded_traces):
                if trace.is_workload():
                    if trace.definition not in workload:
                        workload[trace.definition] = []
                    workload[trace.definition].append(trace)

        # Load regular traces grouped by definition
        traces = {}
        for trace_file in trace_files:
            loaded_traces = load_jsonl(trace_file, Trace)
            for trace in cast(List[Trace], loaded_traces):
                if not trace.is_workload():
                    if trace.definition not in traces:
                        traces[trace.definition] = []
                    traces[trace.definition].append(trace)

        return cls(
            definitions=definitions,
            solutions=solutions,
            workload=workload,
            traces=traces,
        )

    def add_definition(self, definition: Definition) -> None:
        if definition.name in self.definitions:
            raise ValueError(f"Definition with name '{definition.name}' already exists")
        self.definitions[definition.name] = definition

    def add_solution(self, solution: Solution) -> None:
        if solution.definition not in self.solutions:
            self.solutions[solution.definition] = []
        self.solutions[solution.definition].append(solution)

    def add_trace(self, trace: Trace) -> None:
        if trace.is_workload():
            if trace.definition not in self.workload:
                self.workload[trace.definition] = []
            self.workload[trace.definition].append(trace)
        else:
            if trace.definition not in self.traces:
                self.traces[trace.definition] = []
            self.traces[trace.definition].append(trace)

    def get_definition(self, name: str) -> Optional[Definition]:
        return self.definitions.get(name)

    def get_solution(self, name: str) -> Optional[Solution]:
        for solutions_list in self.solutions.values():
            for solution in solutions_list:
                if solution.name == name:
                    return solution
        return None

    def get_traces_for_definition(self, name: str) -> List[Trace]:
        return self.traces.get(name, [])

    def get_best_op(
        self,
        name: str,
        axes: Optional[Dict[str, int]] = None,
        max_abs_diff: float = 1e-5,
        max_relative_diff: float = 1e-5,
    ) -> Optional[Callable]:
        candidates = self.get_traces_for_definition(name)
        if axes:
            candidates = filter_by_axes(candidates, axes)
        candidates = filter_passed_traces(candidates)
        candidates = filter_by_error(candidates, max_abs_diff, max_relative_diff)
        best = get_best_trace(candidates)
        if best:
            solution = self.get_solution(best.solution)
            if solution:
                from .benchmark import build_solution
                return build_solution(solution)
        return None

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all traces."""
        all_traces = []
        for traces_list in self.traces.values():
            all_traces.extend(traces_list)

        total = len(all_traces)
        passed = sum(1 for t in all_traces if t.evaluation["status"] == "PASSED")
        failed = total - passed
        latencies = [t.evaluation["performance"]["latency_ms"] for t in all_traces]
        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency,
        }


T = TypeVar("T")


def build_index(items: List[T], key_fn: Callable[[T], str]) -> Dict[str, T]:
    return {key_fn(item): item for item in items}

def filter_by_axes(traces: List[Trace], axes: Dict[str, int]) -> List[Trace]:
    return [t for t in traces if all(t.workload["axes"].get(ax) == val for ax, val in axes.items())]

def filter_passed_traces(traces: List[Trace]) -> List[Trace]:
    return [t for t in traces if t.evaluation["status"] == "PASSED"]


def filter_by_error(traces: List[Trace], max_abs: float, max_rel: float) -> List[Trace]:
    return [
        t
        for t in traces
        if t.evaluation["correctness"]["max_absolute_error"] <= max_abs
        and t.evaluation["correctness"]["max_relative_error"] <= max_rel
    ]


def get_best_trace(traces: List[Trace]) -> Optional[Trace]:
    return max(traces, key=lambda t: t.evaluation["performance"]["speedup_factor"], default=None)
