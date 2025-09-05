from __future__ import annotations

"""
Presets for dedup policies, tensor selectors, and rule sets.
"""

# Re-export from legacy module to keep behavior unchanged
from flashinfer_bench.tracing_rules import (  # noqa: F401
    DEDUP_BY_AVG_SEQ_LEN,
    DEDUP_BY_AXES,
    DUMP_ALL,
    DUMP_INT32,
    DUMP_NONE,
    KEEP_ALL,
    KEEP_RANDOM_ONE,
    all_dump_rule,
    axes_only_rule,
    dump_int32,
    fib_attn_tracing,
    fib_full_tracing,
    gemm_rule,
    gqa_paged_decode_rule,
    gqa_paged_prefill_rule,
    gqa_ragged_prefill_rule,
    mla_paged_decode_rule,
    mla_paged_prefill_rule,
    mla_ragged_prefill_rule,
    policy_dedup_by_avg_seq_len,
    policy_dedup_by_axes,
    policy_keep_first_k,
    policy_keep_random_k,
)

__all__ = [
    # policies
    "policy_keep_first_k",
    "policy_keep_random_k",
    "policy_dedup_by_axes",
    "policy_dedup_by_avg_seq_len",
    "KEEP_ALL",
    "KEEP_RANDOM_ONE",
    "DEDUP_BY_AXES",
    "DEDUP_BY_AVG_SEQ_LEN",
    # tensor selectors
    "dump_int32",
    "DUMP_INT32",
    "DUMP_ALL",
    "DUMP_NONE",
    # rule presets
    "gemm_rule",
    "mla_paged_prefill_rule",
    "mla_ragged_prefill_rule",
    "mla_paged_decode_rule",
    "gqa_paged_prefill_rule",
    "gqa_ragged_prefill_rule",
    "gqa_paged_decode_rule",
    "axes_only_rule",
    "all_dump_rule",
    # rulesets
    "fib_full_tracing",
    "fib_attn_tracing",
]
