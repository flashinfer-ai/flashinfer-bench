# Model Kernel Coverage

This document tracks which kernels are supported in FlashInfer-Bench for each model.

- ✅ Definition JSON exists in `flashinfer_trace/definitions/`
- ❌ Definition is referenced in `models.ts` but the file does not exist (missing)
- — Module exists in the architecture but no definition is mapped (unmapped)

## Summary

| Model | Architecture | rmsnorm | gemm | gqa | mla | gdn | moe | mamba_ssu |
|-------|-------------|:-------:|:----:|:---:|:---:|:---:|:---:|:---------:|
| DeepSeek V3/R1 | MLA + Dense/MoE | ✅ | — | — | ✅ (partial) | — | ✅ | — |
| Llama 3.1 8B | GQA + Dense | ✅ | ✅ | ✅ | — | — | — | — |
| Qwen3 30B A3B | GQA + MoE | ✅ | — | ✅ | — | — | — (unmapped) | — |
| Qwen3 Next 80B A3B | GDN + GQA + MoE | ✅ | — | ❌ | — | ✅ (partial) | — (unmapped) | — |

---

## DeepSeek V3 / R1

**Architecture**: 61 decoder layers, MLA attention, hybrid Dense+MoE FFN

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h7168` | rmsnorm | ✅ |
| `fused_add_rmsnorm_h7168` | rmsnorm | ✅ |
| `rmsnorm_h1536` | rmsnorm | ✅ |
| `rmsnorm_h512` | rmsnorm | ✅ |
| `mla_ragged_prefill_causal_h16_qk192_vo128` | mla_ragged | ❌ |
| `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` | mla_paged | ✅ |
| `mla_paged_decode_h16_ckv512_kpe64_ps1` | mla_paged | ✅ |
| `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` | moe | ✅ |

**Coverage**: 7 / 8 definitions present. Missing: MLA ragged prefill definition.

---

## Llama 3.1 8B

**Architecture**: 32 decoder layers, GQA attention, dense MLP

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h4096` | rmsnorm | ✅ |
| `fused_add_rmsnorm_h4096` | rmsnorm | ✅ |
| `gemm_n6144_k4096` | gemm | ✅ |
| `gemm_n4096_k4096` | gemm | ✅ |
| `gemm_n28672_k4096` | gemm | ✅ |
| `gemm_n4096_k14336` | gemm | ✅ |
| `gqa_paged_prefill_causal_h32_kv8_d128_ps1` | gqa_paged | ✅ |
| `gqa_paged_decode_h32_kv8_d128_ps1` | gqa_paged | ✅ |
| `gqa_ragged_prefill_causal_h32_kv8_d128` | gqa_ragged | ✅ |

**Coverage**: 9 / 9 definitions present. Fully covered.

---

## Qwen3 30B A3B

**Architecture**: 32 decoder layers, GQA attention, MoE FFN (30 MoE + 2 dense layers)

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h128` | rmsnorm | ✅ |
| `rmsnorm_h2048` | rmsnorm | ✅ |
| `fused_add_rmsnorm_h2048` | rmsnorm | ✅ |
| `gqa_paged_prefill_causal_h32_kv4_d128_ps1` | gqa_paged | ✅ |
| `gqa_paged_decode_h32_kv4_d128_ps1` | gqa_paged | ✅ |
| `gqa_ragged_prefill_causal_h32_kv4_d128` | gqa_ragged | ✅ |
| MoE gate / topk / experts | moe | — |

**Coverage**: 6 / 6 referenced definitions present. MoE kernels are not yet mapped in `models.ts`.

---

## Qwen3 Next 80B A3B

**Architecture**: 48 layers total — 36 GDN (linear attention) + 12 GQA (standard attention), all layers use MoE FFN

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h2048` | rmsnorm | ✅ |
| `fused_add_rmsnorm_h2048` | rmsnorm | ✅ |
| `gdn_prefill_qk16_v32_d128_k_last` | gdn TP=1 | ❌ |
| `gdn_prefill_qk8_v16_d128_k_last` | gdn TP=2 | ✅ |
| `gdn_prefill_qk4_v8_d128_k_last` | gdn TP=4 | ✅ |
| `gdn_decode_qk16_v32_d128_k_last` | gdn TP=1 | ❌ |
| `gdn_decode_qk8_v16_d128_k_last` | gdn TP=2 | ✅ |
| `gdn_decode_qk4_v8_d128_k_last` | gdn TP=4 | ✅ |
| `gdn_mtp_qk16_v32_d128_k_last` | gdn TP=1 | ✅ |
| `gdn_mtp_qk8_v16_d128_k_last` | gdn TP=2 | ✅ |
| `gdn_mtp_qk4_v8_d128_k_last` | gdn TP=4 | ✅ |
| `gqa_paged_prefill_causal_h16_kv2_d256_ps1` | gqa_paged | ❌ |
| `gqa_paged_decode_h16_kv2_d256_ps1` | gqa_paged | ❌ |
| `gqa_ragged_prefill_causal_h16_kv2_d256` | gqa_ragged | ❌ |
| MoE gate / topk / experts (GDN layers) | moe | — |
| MoE gate / topk / experts (GQA layers) | moe | — |

**Coverage**: 9 / 16 referenced definitions present.


Missing GQA definitions use a different shape (h=16, kv=2, d=256) not covered by any existing definition.

---

## Available Definitions Not Yet Assigned to Any Model

These definitions exist in `flashinfer_trace/definitions/` but are not currently referenced by any model in `models.ts`:

| Definition | Op Type | Notes |
|-----------|---------|-------|
| `gemm_n128_k2048` | gemm | — |
| `gemm_n2048_k4096` | gemm | — |
| `gemm_n256_k7168` | gemm | — |
| `gemm_n5120_k2048` | gemm | — |
| `gqa_paged_decode_h32_kv4_d128_ps64` | gqa_paged | ps64 variant |
| `gqa_paged_decode_h32_kv8_d128_ps64` | gqa_paged | ps64 variant |
| `gqa_paged_prefill_causal_h32_kv4_d128_ps64` | gqa_paged | ps64 variant |
| `gqa_paged_prefill_causal_h32_kv8_d128_ps64` | gqa_paged | ps64 variant |
| `mla_paged_decode_h16_ckv512_kpe64_ps64` | mla_paged | ps64 variant |
| `mla_paged_prefill_causal_h16_ckv512_kpe64_ps64` | mla_paged | ps64 variant |
| `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps1` | dsa_paged | DeepSeek sparse attn |
| `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` | dsa_paged | DeepSeek sparse attn |
| `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` | dsa_paged | DeepSeek sparse indexer |
| `top_k_sampling_from_probs_v128256` | sampling | Llama 3.1 vocab |
| `top_k_sampling_from_probs_v129280` | sampling | Qwen3 vocab |
| `top_k_sampling_from_probs_v151936` | sampling | DeepSeek vocab |
| `top_k_top_p_sampling_from_probs_v128256` | sampling | — |
| `top_k_top_p_sampling_from_probs_v129280` | sampling | — |
| `top_k_top_p_sampling_from_probs_v151936` | sampling | — |
| `top_p_sampling_from_probs_v128256` | sampling | — |
| `top_p_sampling_from_probs_v129280` | sampling | — |
| `top_p_sampling_from_probs_v151936` | sampling | — |
