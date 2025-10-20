"""Built-in tracing configurations and presets.

This module provides pre-configured TracingConfig instances for common use cases
such as GEMM, attention kernels, etc.
"""

from flashinfer_bench.tracing.config import TracingConfig

from .policies import AttentionFilterPolicy

# ============================================================================
# TracingConfig Presets
# ============================================================================

gemm_config = TracingConfig(input_dump_policy="dump_none", filter_policy="keep_first_by_axes")

mla_paged_prefill_config = TracingConfig(
    input_dump_policy=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    filter_policy=lambda: AttentionFilterPolicy(k=1),
)


mla_paged_decode_config = TracingConfig(
    input_dump_policy=["kv_indptr", "kv_indices", "sm_scale"],
    filter_policy=lambda: AttentionFilterPolicy(k=1),
)

gqa_paged_prefill_config = TracingConfig(
    input_dump_policy=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    filter_policy=lambda: AttentionFilterPolicy(k=1),
)

gqa_ragged_prefill_config = TracingConfig(
    input_dump_policy=["qo_indptr", "kv_indptr", "sm_scale"],
    filter_policy=lambda: AttentionFilterPolicy(k=1),
)

gqa_paged_decode_config = TracingConfig(
    input_dump_policy=["kv_indptr", "kv_indices", "sm_scale"],
    filter_policy=lambda: AttentionFilterPolicy(k=1),
)

all_dump_config = TracingConfig(input_dump_policy="dump_all", filter_policy="keep_all")

axes_only_config = TracingConfig(input_dump_policy="dump_none", filter_policy="keep_first_by_axes")

FULL_TRACING_CONFIGS = {
    "gemm_n128_k2048": gemm_config,
    "gemm_n256_k7168": gemm_config,
    "gemm_n2048_k4096": gemm_config,
    "gemm_n4096_k14336": gemm_config,
    "gemm_n4096_k4096": gemm_config,
    "gemm_n5120_k2048": gemm_config,
    "gemm_n6144_k4096": gemm_config,
    "gemm_n28672_k4096": gemm_config,
    "gqa_paged_decode_h32_kv4_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_decode_h32_kv8_d128_ps1": gqa_paged_decode_config,
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1": gqa_paged_prefill_config,
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": gqa_paged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv4_d128": gqa_ragged_prefill_config,
    "gqa_ragged_prefill_causal_h32_kv8_d128": gqa_ragged_prefill_config,
    "mla_paged_decode_h16_ckv512_kpe64_ps1": mla_paged_decode_config,
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1": mla_paged_prefill_config,
    "fused_add_rmsnorm_h2048": axes_only_config,
    "fused_add_rmsnorm_h4096": axes_only_config,
    "fused_add_rmsnorm_h7168": axes_only_config,
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
}
