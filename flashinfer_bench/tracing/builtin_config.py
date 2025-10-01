"""Built-in tracing configurations and presets.

This module provides pre-configured TracingConfig instances for common use cases
such as GEMM, attention kernels, etc.
"""

from .tracing_config import TracingConfig
from .tracing_policies import AttentionDedupPolicy

# ============================================================================
# TracingRule Presets
# ============================================================================

gemm_rule = TracingConfig(tensors_to_dump="dump_none", dedup_policy="keep_first_by_axes")

mla_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

mla_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["seq_indptr", "sm_scale"], dedup_policy=lambda: AttentionDedupPolicy(k=1)
)

mla_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

all_dump_rule = TracingConfig(tensors_to_dump="dump_all", dedup_policy="keep_all")

axes_only_rule = TracingConfig(tensors_to_dump="dump_none", dedup_policy="keep_first_by_axes")

fib_full_tracing = {
    "gemm_n_28672_k_4096": gemm_rule,
    "gemm_n_4096_k_14336": gemm_rule,
    "gemm_n_4096_k_4096": gemm_rule,
    "gemm_n_6144_k_4096": gemm_rule,
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_rule,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_rule,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_rule,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_rule,
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule,
}

fib_attn_tracing = {
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_rule,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_rule,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_rule,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_rule,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_rule,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_rule,
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_rule,
}
