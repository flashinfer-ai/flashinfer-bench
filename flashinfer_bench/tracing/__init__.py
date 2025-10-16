from __future__ import annotations

from .builtin_tracing_config import AttentionDedupPolicy
from .tracing import disable_tracing, enable_tracing
from .tracing_config import TracingConfig
from .tracing_policy import (
    BUILTIN_DEDUP_POLICIES,
    BUILTIN_INPUT_DUMP_POLICIES,
    DedupPolicy,
    DedupPolicyFactory,
    InputDumpPolicyFunction,
    KeepAllPolicy,
    KeepFirstByAxesPolicy,
    KeepFirstKPolicy,
    WorkloadEntry,
    dump_all,
    dump_int32,
    dump_none,
)
from .tracing_runtime import TracingRuntime, get_tracing_runtime

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
