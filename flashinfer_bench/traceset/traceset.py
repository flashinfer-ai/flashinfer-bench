import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from flashinfer_bench.specs.trace import Trace

class TraceSet:
    """
    A collection of Trace objects, representing benchmark results.
    Provides methods for loading, summarizing, selecting, and exporting traces.
    Compatible with the Trace JSON schema described in the benchmark documentation.
    """
    def __init__(self, traces: List[Trace]):
        self.traces = traces

    @classmethod
    def from_path(cls, path: str) -> "TraceSet":
        traces = []
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                # Validate required fields for schema compliance
                assert "definition" in obj and "solution" in obj and "workload" in obj and "evaluation" in obj, "Missing required Trace fields."
                traces.append(Trace.from_dict(obj))
        return cls(traces)

    @classmethod
    def from_traces(cls, traces: List[Trace]) -> "TraceSet":
        return cls(traces)

    def get_best_op(self, key: Optional[str] = None) -> Optional[Trace]:
        if not self.traces:
            return None
        if key is None:
            key = "latency_ms"
        def get_metric(trace: Trace):
            return getattr(trace.evaluation.performance, key, float('inf'))
        return min(self.traces, key=get_metric)

    def summary(self) -> Dict[str, Any]:
        total = len(self.traces)
        passed = sum(1 for t in self.traces if t.evaluation.status == "PASSED")
        failed = total - passed
        latencies = [t.evaluation.performance.latency_ms for t in self.traces]
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

    def export(self, path: str):
        from flashinfer_bench.utils.jsonl import write_jsonl
        write_jsonl(path, [t.to_dict() for t in self.traces]) 