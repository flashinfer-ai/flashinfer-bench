import glob
from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Optional, TypeVar, cast
from pathlib import Path

from flashinfer_bench import Definition
from flashinfer_bench import Solution
from flashinfer_bench import Trace
from flashinfer_bench.utils.json_utils import load_jsonl

@dataclass
class TraceSet:

    definitions: Dict[str, Definition]
    solutions: Dict[str, Solution]
    traces: List[Trace]

    @classmethod
    def from_path(cls, path: str) -> "TraceSet":
        base_path = Path(path)

        definitions = []
        definitions_files = glob.glob(str(base_path / "definitions" / "*.json"), recursive=True)
        for definition_file in definitions_files:
            with open(definition_file, 'r') as f:
                definition = Definition.from_json(f.read())
            definitions.append(definition)
        
        solutions_files = glob.glob(str(base_path / "solutions" / "*.json"), recursive=True)
        solutions = []
        for solution_file in solutions_files:
            with open(solution_file, 'r') as f:
                solution = Solution.from_json(f.read())
            solutions.append(solution)
        
        traces = []
        
        trace_files = glob.glob(str(base_path / "traces" / "*.jsonl"), recursive=True)
        for trace_file in trace_files:
            loaded_traces = load_jsonl(trace_file, Trace)
            traces.extend(cast(List[Trace], loaded_traces))

        indexed_definitions = build_index(definitions, lambda d: d.name)
        indexed_solutions = build_index(solutions, lambda s: s.name)

        return cls(definitions=indexed_definitions, solutions=indexed_solutions, traces=traces)

    def get_definition(self, name: str) -> Optional[Definition]:
        return self.definitions.get(name)

    def get_solution(self, name: str) -> Optional[Solution]:
        return self.solutions.get(name)

    def get_traces_for_definition(self, name: str) -> List[Trace]:
        return [t for t in self.traces if t.definition == name]

    def get_best_op(
        self,
        name: str,
        max_abs_diff: float = 1e-5,
        max_relative_diff: float = 1e-5,
    ) -> Optional[Trace]:
        candidates = filter_passed_traces(self.get_traces_for_definition(name))
        candidates = filter_by_error(candidates, max_abs_diff, max_relative_diff)
        return get_best_trace(candidates)

    def summary(self) -> Dict[str, Any]:
        total = len(self.traces)
        passed = sum(1 for t in self.traces if t.evaluation["status"] == "PASSED")
        failed = total - passed
        latencies = [t.evaluation["performance"]["latency_ms"] for t in self.traces]
        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency
        }


T = TypeVar("T")
def build_index(items: List[T], key_fn: Callable[[T], str]) -> Dict[str, T]:
    return {key_fn(item): item for item in items}

def filter_passed_traces(traces: List[Trace]) -> List[Trace]:
    return [t for t in traces if t.evaluation["status"] == "PASSED"]

def filter_by_error(traces: List[Trace], max_abs: float, max_rel: float) -> List[Trace]:
    return [
        t for t in traces
        if t.evaluation["correctness"]["max_absolute_error"] <= max_abs and
           t.evaluation["correctness"]["max_relative_error"] <= max_rel
    ]

def get_best_trace(traces: List[Trace]) -> Optional[Trace]:
    return max(traces, key=lambda t: t.evaluation["performance"]["speedup_factor"], default=None)