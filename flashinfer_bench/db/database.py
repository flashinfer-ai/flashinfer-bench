import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from flashinfer_bench.specs.definition import Definition
from flashinfer_bench.specs.solution import Solution
from flashinfer_bench.specs.trace import Trace

from .index import build_index
from .query import filter_passed_traces, filter_by_error, get_best_trace


@dataclass
class Database:
    definitions: List[Definition]
    solutions: List[Solution]
    traces: List[Trace]

    _definitions_index: Dict[str, Definition] = field(init=False, default_factory=dict)
    _solutions_index: Dict[str, Solution] = field(init=False, default_factory=dict)
    _traces_by_definition: Dict[str, List[Trace]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self._definitions_index = build_index(self.definitions, lambda d: d.name)
        self._solutions_index = build_index(self.solutions, lambda s: s.name)
        for trace in self.traces:
            self._traces_by_definition.setdefault(trace.definition, []).append(trace)

    @classmethod
    def from_uri(cls, uri: str) -> "Database":
        root = Path(uri)

        with open(root / "definitions.json", "r") as f:
            definitions = [Definition(**d) for d in json.load(f)]

        solutions_path = root / "solutions.json"
        if solutions_path.exists():
            with open(solutions_path, "r") as f:
                solutions = [Solution(**s) for s in json.load(f)]
        else:
            solutions = []

        traces_path = root / "traces.jsonl"
        traces = []
        with open(traces_path, "r") as f:
            traces = [Trace.from_dict(json.loads(line)) for line in f]

        return cls(definitions=definitions, solutions=solutions, traces=traces)

    def get_definition(self, name: str) -> Optional[Definition]:
        return self._definitions_index.get(name)

    def get_solution(self, name: str) -> Optional[Solution]:
        return self._solutions_index.get(name)

    def get_traces_for_definition(self, name: str) -> List[Trace]:
        return self._traces_by_definition.get(name, [])

    def get_best_op(
        self,
        name: str,
        max_abs_diff: float = 1e-5,
        max_relative_diff: float = 1e-5,
    ) -> Optional[Trace]:
        candidates = filter_passed_traces(self.get_traces_for_definition(name))
        candidates = filter_by_error(candidates, max_abs_diff, max_relative_diff)
        return get_best_trace(candidates)
