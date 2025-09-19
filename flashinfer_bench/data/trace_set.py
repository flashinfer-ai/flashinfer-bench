"""TraceSet as a pure data warehouse for definitions, solutions, and traces."""

import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .definition import Definition
from .json_utils import append_jsonl_file, load_json_file, load_jsonl_file
from .solution import Solution
from .trace import EvaluationStatus, Trace


# TODO(shanli): TraceSet wide validation
@dataclass
class TraceSet:
    """Stores a FlashInfer Trace dataset containing definitions, solutions, workloads, and traces.

    TraceSet serves as a centralized data warehouse for managing FlashInfer benchmark data.
    It provides efficient lookup and filtering capabilities for definitions, solutions, and
    execution traces organized by definition names.

    The data structure is optimized for fast lookups with dictionary-based storage where
    keys are definition names and values are lists of associated objects.
    """

    root: str
    """The root path of the TraceSet. Must be recognized by Path()."""
    definitions: Dict[str, Definition] = field(default_factory=dict)
    """The definitions in the database. Map from definition name to Definition object."""
    solutions: Dict[str, List[Solution]] = field(default_factory=dict)
    """The solutions in the database. Map from definition name to all the solutions for that
    definition."""
    workloads: Dict[str, List[Trace]] = field(default_factory=dict)
    """The workload traces in the database. Map from definition name to all workload traces for that
    definition."""
    traces: Dict[str, List[Trace]] = field(default_factory=dict)
    """The traces in the database. Map from definition name to all traces for that definition."""

    @classmethod
    def from_path(cls: Type["TraceSet"], path: str) -> "TraceSet":
        """Load a TraceSet from a directory structure.

        Loads a complete TraceSet by scanning the directory structure for:
        - definitions/: JSON files containing Definition objects
        - solutions/: JSON files containing Solution objects
        - traces/: JSONL files containing Trace objects (both workload and execution traces)

        Parameters
        ----------
        path : str
            Root directory path containing the dataset structure.

        Returns
        -------
        TraceSet
            A new TraceSet instance populated with data from the directory.

        Raises
        ------
        ValueError
            If duplicate definition names or solution names are found.
        FileNotFoundError
            If the specified path doesn't exist.
        """
        base_path = Path(path)

        # Load json files from all subdirectories
        definitions = {}
        for p in sorted((base_path / "definitions").rglob("*.json")):
            d = load_json_file(Definition, p)
            if d.name in definitions:
                raise ValueError(f"Duplicate definition name: {d.name}")
            definitions[d.name] = d

        seen_solutions = set()
        solutions = defaultdict(list)
        for p in sorted((base_path / "solutions").rglob("*.json")):
            s = load_json_file(Solution, p)
            if s.name in seen_solutions:
                raise ValueError(f"Duplicate solution name: {s.name}")
            seen_solutions.add(s.name)
            solutions[s.definition].append(s)

        workloads = defaultdict(list)
        for p in sorted((base_path / "workloads").rglob("*.jsonl")):
            for t in load_jsonl_file(Trace, p):
                assert t.is_workload_trace()
                workloads[t.definition].append(t)

        traces = defaultdict(list)
        for p in sorted((base_path / "traces").rglob("*.jsonl")):
            for t in load_jsonl_file(Trace, p):
                assert not t.is_workload_trace()
                traces[t.definition].append(t)

        return cls(
            root=base_path,
            definitions=definitions,
            solutions=solutions,
            workloads=workloads,
            traces=traces,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TraceSet to a Python dict.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the TraceSet.
        """
        return {
            "definitions": {
                name: definition.model_dump(mode="json")
                for name, definition in self.definitions.items()
            },
            "solutions": {
                name: [solution.model_dump(mode="json") for solution in solutions]
                for name, solutions in self.solutions.items()
            },
            "workload": {
                name: [workload.model_dump(mode="json") for workload in workloads]
                for name, workloads in self.workloads.items()
            },
            "traces": {
                name: [trace.model_dump(mode="json") for trace in traces]
                for name, traces in self.traces.items()
            },
        }

    def get_solution(self, name: str) -> Optional[Solution]:
        """Get a solution by name from all loaded solutions.

        Searches across all solutions in the TraceSet to find one with the specified name.
        Since solution names are unique across the entire dataset, this returns at most
        one solution.

        Parameters
        ----------
        name : str
            The name of the solution to retrieve.

        Returns
        -------
        Optional[Solution]
            The solution with the given name, or None if not found.
        """
        for solution_list in self.solutions.values():
            for solution in solution_list:
                if solution.name == name:
                    return solution
        return None

    def filter_traces(self, def_name: str, atol: float = 1e-2, rtol: float = 1e-2) -> List[Trace]:
        """Filter traces for a definition based on error bounds.

        Returns only successful traces that meet the specified absolute and relative
        error tolerance criteria. This is useful for finding high-quality implementations
        that produce numerically accurate results.

        Parameters
        ----------
        def_name : str
            Name of the definition to filter traces for.
        atol : float, optional
            Maximum absolute error tolerance, by default 1e-2.
        rtol : float, optional
            Maximum relative error tolerance, by default 1e-2.

        Returns
        -------
        List[Trace]
            List of traces that passed evaluation and meet error criteria.
            Empty list if no traces match the criteria.
        """
        return [
            t
            for t in self.traces.get(def_name, [])
            if t.evaluation
            and t.evaluation.status == EvaluationStatus.PASSED
            and t.evaluation.correctness
            and t.evaluation.correctness.max_absolute_error <= atol
            and t.evaluation.correctness.max_relative_error <= rtol
        ]

    def get_best_trace(
        self,
        def_name: str,
        axes: Optional[Dict[str, int]] = None,
        max_abs_error: float = 1e-2,
        max_rel_error: float = 1e-2,
    ) -> Optional[Trace]:
        """Get the best performing trace for a definition based on speedup factor.

        Finds the trace with the highest speedup factor among those that meet the
        specified criteria including axis constraints and error tolerances.

        Parameters
        ----------
        def_name : str
            Name of the definition to find the best trace for.
        axes : Optional[Dict[str, int]], optional
            Dictionary of axis name to value pairs for exact matching.
            Only traces with exactly matching axis values will be considered.
            If None, all axis configurations are considered.
        max_abs_error : float, optional
            Maximum absolute error tolerance, by default 1e-2.
        max_rel_error : float, optional
            Maximum relative error tolerance, by default 1e-2.

        Returns
        -------
        Optional[Trace]
            The best performing trace meeting all criteria, or None if no traces
            match the requirements.
        """
        candidates = self.traces.get(def_name, [])

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
        """Get a comprehensive summary of all traces in the TraceSet.

        Computes aggregate statistics across all execution traces including success rates,
        latency statistics, and overall dataset size metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the following keys:
            - total: Total number of traces
            - passed: Number of traces with successful evaluation
            - failed: Number of traces with failed evaluation
            - min_latency_ms: Minimum latency among successful traces (None if no successful traces)
            - max_latency_ms: Maximum latency among successful traces (None if no successful traces)
            - avg_latency_ms: Average latency among successful traces (None if no successful traces)
        """
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

    def backup_traces(self) -> None:
        """Backup the traces directory to a new directory. This is useful when we want to keep the
        old traces for reference.
        """
        traces_dir = self.root / "traces"
        backup = self.root / f"traces_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Move traces directory to backup
        if traces_dir.exists():
            shutil.move(str(traces_dir), str(backup))
        else:
            backup.mkdir(parents=True, exist_ok=True)

        # Create new traces directory
        traces_dir.mkdir(parents=True, exist_ok=True)

    def add_traces(self, traces: List[Trace]) -> None:
        """Add traces to the TraceSet, and store the traces to disk.

        Parameters
        ----------
        traces : List[Trace]
            The traces to add to the TraceSet.
        """
        buckets: Dict[Path, List[Trace]] = defaultdict(list)
        traces_path = Path(self.root) / "traces"
        for trace in traces:
            # Add to in-memory database
            self.traces.setdefault(trace.definition, []).append(trace)
            defn = self.definitions[trace.definition]
            path = traces_path / defn.op_type / f"{defn.name}.jsonl"
            # Add to disk bucket
            buckets[path].append(trace)

        # Write to disk
        for path, traces in buckets.items():
            append_jsonl_file(traces, path)
