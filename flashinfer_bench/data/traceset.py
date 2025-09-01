"""TraceSet as a pure data warehouse for definitions, solutions, and traces."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .definition import Definition
from .json_codec import load_json_file, load_jsonl_file
from .solution import Solution
from .trace import EvaluationStatus, Trace


@dataclass
class TraceSet:
    """A pure data warehouse for definitions, solutions, workloads, and traces.

    This class only handles data storage, loading, saving, querying, and statistics.
    """

    # def_name -> Definition
    definitions: Dict[str, Definition] = field(default_factory=dict)
    # def_name -> List[Solution]
    solutions: Dict[str, List[Solution]] = field(default_factory=dict)
    # def_name -> List[Trace] (workload traces)
    workload: Dict[str, List[Trace]] = field(default_factory=dict)
    # def_name -> List[Trace]
    traces: Dict[str, List[Trace]] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str) -> "TraceSet":
        """Load a TraceSet from a directory structure."""
        base_path = Path(path)

        # Load json files from all subdirectories
        definitions = {}
        for p in sorted((base_path / "definitions").rglob("*.json")):
            d = load_json_file(p, Definition)
            if d.name in definitions:
                raise ValueError(f"Duplicate definition name: {d.name}")
            definitions[d.name] = d

        seen_solutions = set()
        solutions = defaultdict(list)
        for p in sorted((base_path / "solutions").rglob("*.json")):
            s = load_json_file(p, Solution)
            if s.name in seen_solutions:
                raise ValueError(f"Duplicate solution name: {s.name}")
            seen_solutions.add(s.name)
            solutions[s.definition].append(s)

        workloads = defaultdict(list)
        traces = defaultdict(list)
        for p in sorted((base_path / "traces").rglob("*.jsonl")):
            for t in load_jsonl_file(p, Trace):
                (workloads if t.is_workload() else traces)[t.definition].append(t)

        return cls(
            definitions=definitions,
            solutions=solutions,
            workload=workloads,
            traces=traces,
        )

    def get_definition(self, name: str) -> Optional[Definition]:
        """Get a definition by name."""
        return self.definitions.get(name)

    def get_solution(self, name: str) -> Optional[Solution]:
        """Get a solution by name."""
        for solution_list in self.solutions.values():
            for solution in solution_list:
                if solution.name == name:
                    return solution
        return None

    def get_traces_for_definition(self, name: str) -> List[Trace]:
        """Get all traces for a definition."""
        return self.traces.get(name, [])

    def get_workloads_for_definition(self, name: str) -> List[Trace]:
        """Get all workload-only traces for a definition."""
        return self.workload.get(name, [])

    def get_best_trace(
        self,
        def_name: str,
        axes: Optional[Dict[str, int]] = None,
        max_abs_error: float = 1e-5,
        max_rel_error: float = 1e-5,
    ) -> Optional[Trace]:
        """Get the best trace for a definition based on performance.

        This returns the Trace object itself.
        """
        candidates = self.get_traces_for_definition(def_name)

        # Axes exact match
        # TODO(shanli): advanced input filtering
        if axes:
            candidates = [
                t
                for t in candidates
                if all(t.workload.axes.get(ax) == val for ax, val in axes.items())
            ]

        # Filter by successful status
        candidates = [
            t for t in candidates if t.evaluation and t.evaluation.status == EvaluationStatus.PASSED
        ]

        # Filter by error bounds
        candidates = [
            t
            for t in candidates
            if t.evaluation.correctness.max_absolute_error <= max_abs_error
            and t.evaluation.correctness.max_relative_error <= max_rel_error
        ]

        # Return the one with best speedup
        if candidates:
            return max(
                candidates,
                key=lambda t: t.evaluation.performance.speedup_factor if t.evaluation else 0,
            )

        return None

    def summary(self) -> Dict[str, any]:
        """Get a summary of all traces."""
        all_traces = [t for traces in self.traces.values() for t in traces]

        if not all_traces:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "avg_latency_ms": None,
            }

        passed = [
            t for t in all_traces if t.evaluation and t.evaluation.status == EvaluationStatus.PASSED
        ]

        latencies = [t.evaluation.performance.latency_ms for t in passed]

        min_latency = min(latencies) if latencies else None
        max_latency = max(latencies) if latencies else None
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        return {
            "total": len(all_traces),
            "passed": len(passed),
            "failed": len(all_traces) - len(passed),
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "avg_latency_ms": avg_latency,
        }
