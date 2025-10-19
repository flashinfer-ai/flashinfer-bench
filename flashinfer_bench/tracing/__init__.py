from __future__ import annotations

from .builtin.policies import (
    BUILTIN_DEDUP_POLICIES,
    BUILTIN_INPUT_DUMP_POLICIES,
    AttentionDedupPolicy,
    InputDumpPolicyFunction,
    KeepAllPolicy,
    KeepFirstByAxesPolicy,
    KeepFirstKPolicy,
    dump_all,
    dump_int32,
    dump_none,
)
from .config import TracingConfig
from .dedup_policy import DedupPolicy, DedupPolicyFactory
from .runtime import TracingRuntime, get_tracing_runtime
from .tracing import disable_tracing, enable_tracing
from .workload_entry import WorkloadEntry

__all__ = [
    "disable_tracing",
    "enable_tracing",
    "get_tracing_runtime",
    "TracingRuntime",
    "TracingConfig",
    "WorkloadEntry",
    "DedupPolicy",
    "DedupPolicyFactory",
    "InputDumpPolicyFunction",
    "BUILTIN_DEDUP_POLICIES",
    "KeepAllPolicy",
    "KeepFirstKPolicy",
    "KeepFirstByAxesPolicy",
    "AttentionDedupPolicy",
    "BUILTIN_INPUT_DUMP_POLICIES",
    "dump_all",
    "dump_none",
    "dump_int32",
]
