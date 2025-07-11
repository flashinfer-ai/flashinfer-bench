from typing import List, Optional
from flashinfer_bench.specs.trace import Trace

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
