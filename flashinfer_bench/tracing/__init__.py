"""Tracing subsystem for collecting workload traces."""

from __future__ import annotations

from .config import TracingConfig, TracingConfigRegistry
from .policy import FilterPolicy, InputDumpPolicy, PolicyRegistry
from .presets import get_attention_only_configs, get_default_configs, get_full_configs
from .runtime import TracingRuntime
from .tracing import disable_tracing, enable_tracing
from .workload_entry import WorkloadEntry

__all__ = [
    # Core
    "disable_tracing",
    "enable_tracing",
    "TracingRuntime",
    "WorkloadEntry",
    # Config
    "TracingConfig",
    "TracingConfigRegistry",
    "PolicyRegistry",
    "FilterPolicy",
    "InputDumpPolicy",
    # Registry presets
    "get_default_configs",
    "get_full_configs",
    "get_attention_only_configs",
]
