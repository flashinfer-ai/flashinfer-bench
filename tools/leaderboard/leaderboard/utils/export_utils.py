from typing import Any, Dict, List
import sys, os

from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from flashinfer_bench.trace_set import TraceSet, filter_passed_traces, filter_by_error

_TRACE_SET = None
_LEADERBOARD_JSON = None

def get_trace_set() -> TraceSet:
    global _TRACE_SET
    if _TRACE_SET is None:
        _TRACE_SET = TraceSet.from_path("../../dataset/")
    return _TRACE_SET

def get_definitions():
    return get_trace_set().definitions

def get_a_definition(definition_name: str) -> Dict[str, Any]:
    """Get the definition details for a specific definition name."""
    definitions = get_trace_set().definitions
    if definition_name not in definitions:
        raise ValueError(f"Definition '{definition_name}' not found.")
    return definitions[definition_name]

def to_leaderboard_json(
    max_abs_error: float = 1e-5,
    max_rel_error: float = 1e-5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate leaderboard JSON from a TraceSet, averaging latency per solution."""
    leaderboard = {}

    for def_name, traces in get_trace_set().traces.items():
        valid = filter_passed_traces(traces)
        valid = filter_by_error(valid, max_abs_error, max_rel_error)

        # Group traces by solution
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for trace in valid:
            grouped[trace.solution].append(trace)

        entries = []
        for solution_name, traces_for_solution in grouped.items():
            try:
                latencies = [t.evaluation["performance"]["latency_ms"] for t in traces_for_solution]
                speedups = [t.evaluation["performance"]["speedup_factor"] for t in traces_for_solution]

                # Use first trace for static fields like device
                ref = traces_for_solution[0]
                perf = ref.evaluation["performance"]
                env = ref.evaluation["environment"]

                entries.append({
                    "solution": solution_name,
                    "latency_ms": mean(latencies),
                    "speedup": mean(speedups),
                    "device": env["device"],
                    "status": ref.evaluation["status"],
                    "timestamp": ref.evaluation["timestamp"],
                })
            except Exception as e:
                print(f"[Warning] Skipping invalid solution group '{solution_name}': {e}")

        entries.sort(key=lambda x: x["latency_ms"])
        leaderboard[def_name] = entries

    return leaderboard

# def to_leaderboard_json(
#     max_abs_error: float = 1e-5,
#     max_rel_error: float = 1e-5,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """Generate leaderboard JSON from a TraceSet without modifying the class."""
#     leaderboard = {}

#     for def_name, traces in get_trace_set().traces.items():
#         valid = filter_passed_traces(traces)
#         valid = filter_by_error(valid, max_abs_error, max_rel_error)

#         entries = []
#         for trace in valid:
#             try:
#                 evaluation = trace.evaluation
#                 perf = evaluation["performance"]
#                 environment = evaluation["environment"]
#                 entries.append({
#                     "solution": trace.solution,
#                     "latency_ms": perf["latency_ms"],
#                     "speedup": perf["speedup_factor"],
#                     "device": environment["device"],
#                     "status": evaluation["status"],
#                     "timestamp": evaluation["timestamp"],
#                 })
#             except Exception as e:
#                 print(f"[Warning] Skipping invalid trace: {e}")

#         entries.sort(key=lambda x: x["latency_ms"])
#         leaderboard[def_name] = entries

#     return leaderboard

def get_leaderboard() -> dict:
    global _LEADERBOARD_JSON
    if _LEADERBOARD_JSON is None:
        _LEADERBOARD_JSON = to_leaderboard_json()
    return _LEADERBOARD_JSON