from typing import Any, Dict, List

import torch

from .tracing_config import DedupByAvgSeqLenPolicy, TracingConfig


# ============================================================================
# tensors_to_dump Presets
# ============================================================================
def dump_int32():
    """Select only int32 tensors for dumping. These inputs are usually indptrs."""

    def _dump(inputs: Dict[str, Any]) -> List[str]:
        picks: List[str] = []
        for name, val in inputs.items():
            if isinstance(val, torch.Tensor) and val.dtype == torch.int32:
                picks.append(name)
        return picks

    return _dump


# Dump all tensors.
DUMP_ALL = lambda inputs: [name for name, val in inputs.items() if isinstance(val, torch.Tensor)]
# Dump no tensors.
DUMP_NONE = lambda _: []
DUMP_INT32 = dump_int32()

# ============================================================================
# TracingRule Presets
# ============================================================================
gemm_rule = TracingConfig(tensors_to_dump=DUMP_NONE, dedup_policy="dedup_by_axes")

mla_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DedupByAvgSeqLenPolicy(k=1),
)

mla_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["seq_indptr", "sm_scale"], dedup_policy=DedupByAvgSeqLenPolicy(k=1)
)

mla_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DedupByAvgSeqLenPolicy(k=1),
)

gqa_paged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DedupByAvgSeqLenPolicy(k=1),
)

gqa_ragged_prefill_rule = TracingConfig(
    tensors_to_dump=["qo_indptr", "kv_indptr", "sm_scale"], dedup_policy=DedupByAvgSeqLenPolicy(k=1)
)

gqa_paged_decode_rule = TracingConfig(
    tensors_to_dump=["kv_indptr", "kv_indices", "sm_scale"],
    dedup_policy=DedupByAvgSeqLenPolicy(k=1),
)

all_dump_rule = TracingConfig(tensors_to_dump=DUMP_ALL, dedup_policy="keep_all")

axes_only_rule = TracingConfig(tensors_to_dump=DUMP_NONE, dedup_policy="dedup_by_axes")

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
