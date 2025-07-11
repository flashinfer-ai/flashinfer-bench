import glob
import json
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, TypeVar, cast
from pathlib import Path

from flashinfer_bench import Definition
from flashinfer_bench import Solution
from flashinfer_bench import Trace
from flashinfer_bench.utils.json_utils import load_json, load_jsonl

@dataclass
class TraceSet:

    definitions: Dict[str, Definition]
    solutions: Dict[str, Solution]
    traces: List[Trace]

    @classmethod
    def from_path(cls, path: str) -> "TraceSet":
        base_path = Path(path)
        
        definition_files = glob.glob(str(base_path / "definitions" / "*.json"))
        definitions = [cast(Definition, load_json(f, Definition)) for f in definition_files]

        solution_files = glob.glob(str(base_path / "solutions" / "*.json"))
        solutions = [cast(Solution, load_json(f, Solution)) for f in solution_files]

        trace_files = glob.glob(str(base_path / "traces" / "*.jsonl"))
        traces = []
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


T = TypeVar("T")
def build_index(items: List[T], key_fn: Callable[[T], str]) -> Dict[str, T]:
    return {key_fn(item): item for item in items}

def filter_passed_traces(traces: List[Trace]) -> List[Trace]:
    return [t for t in traces if t.evaluation.status == "PASSED"]

def filter_by_error(traces: List[Trace], max_abs: float, max_rel: float) -> List[Trace]:
    return [
        t for t in traces
        if t.evaluation.correctness.max_absolute_error <= max_abs and
           t.evaluation.correctness.max_relative_error <= max_rel
    ]

def get_best_trace(traces: List[Trace]) -> Optional[Trace]:
    return max(traces, key=lambda t: t.evaluation.performance.speedup_factor, default=None)