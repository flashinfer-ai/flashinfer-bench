"""Built-in tracing configurations and presets.

This module provides pre-configured TracingConfig instances for common use cases
such as GEMM, attention kernels, etc.
"""

from .tracing_config import TracingConfig
from .tracing_policy import AttentionDedupPolicy

# ============================================================================
# TracingConfig Presets
# ============================================================================

gemm_config = TracingConfig(tensors_to_dump="dump_none", dedup_policy="keep_first_by_axes")

mla_paged_prefill_config = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

mla_ragged_prefill_config = TracingConfig(
    tensors_to_dump=["seq_indptr", "sm_scale"], dedup_policy=lambda: AttentionDedupPolicy(k=1)
)

mla_paged_decode_config = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_paged_prefill_config = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_ragged_prefill_config = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

gqa_paged_decode_config = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=lambda: AttentionDedupPolicy(k=1),
)

all_dump_config = TracingConfig(tensors_to_dump="dump_all", dedup_policy="keep_all")

axes_only_config = TracingConfig(tensors_to_dump="dump_none", dedup_policy="keep_first_by_axes")

FULL_TRACING_CONFIGS = {
    "gemm_n_28672_k_4096": gemm_config,
    "gemm_n_4096_k_14336": gemm_config,
    "gemm_n_4096_k_4096": gemm_config,
    "gemm_n_6144_k_4096": gemm_config,
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_config,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_config,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_config,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_config,
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_config,
}

ATTN_ONLY_TRACING_CONFIGS = {
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_config,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_config,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_config,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_config,
    "mla_ragged_prefill_causal_h16_qk192_vo128": mla_ragged_prefill_config,
}
