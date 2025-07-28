from typing import Any, Dict, List
import sys, os

from collections import defaultdict
from statistics import mean

import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from flashinfer_bench.trace_set import TraceSet, filter_passed_traces, filter_by_error, Definition

_TRACE_SET = None
_LEADERBOARD_JSON = None

def get_trace_set() -> TraceSet:
    global _TRACE_SET
    if _TRACE_SET is None:
        path = os.getenv("FLASHINFER_BENCH_DATASET_PATH", "../../dataset/")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path '{path}' does not exist. Please set FLASHINFER_BENCH_DATASET_PATH environment variable or ensure the dataset is available.")
        _TRACE_SET = TraceSet.from_path(path)
    return _TRACE_SET

def get_definitions():
    return get_trace_set().definitions

def grouped_definitions():
    """Group definitions by their category."""
    definitions = get_definitions()
    grouped = defaultdict(list)
    for name, definition in definitions.items():
        category = definition.get_type()
        grouped[category].append(name)
    return dict(grouped)

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
    trace_set = get_trace_set()

    for def_name, traces in trace_set.traces.items():
        valid = filter_passed_traces(traces)
        valid = filter_by_error(valid, max_abs_error, max_rel_error)

        entries_by_device_and_workload = defaultdict(lambda: defaultdict(list))
        
        for trace in valid:
            try:
                device = trace.evaluation["environment"]["device"]
                axes = trace.workload.get("axes", {})
                w_id = json.dumps(axes, sort_keys=True)  # e.g. {"M": 248} → '{"M": 248}'

                entries_by_device_and_workload[device][w_id].append(trace)
            except Exception as e:
                print(f"[Warning] Failed to group trace for solution '{trace.solution}': {e}")

        result = {}
        for device, workloads in entries_by_device_and_workload.items():
            result[device] = []
            for w_id, traces_for_workload in workloads.items():
                for trace in traces_for_workload:
                    perf = trace.evaluation["performance"]
                    solution_name = trace.solution
                    solution = trace_set.get_solution(solution_name)
                    result[device].append({
                        "workload": w_id,
                        "solution": solution_name,
                        "latency_ms": perf["latency_ms"],
                        "speedup": perf["speedup_factor"],
                        "author": solution.get_author() if solution else "Unknown",
                        "solution_file": solution.get_code() if solution else "",
                    })
            result[device].sort(key=lambda x: (x["workload"], x["latency_ms"]))

        leaderboard[def_name] = result

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