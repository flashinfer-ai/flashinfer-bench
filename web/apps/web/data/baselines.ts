const baselines = {
  fused_add_rmsnorm_h2048: {
    default: "flashinfer_wrapper_74a870",
  },
  fused_add_rmsnorm_h4096: {
    default: "flashinfer_wrapper_0ff432",
  },
  fused_add_rmsnorm_h7168: {
    default: "flashinfer_wrapper_5bddf1",
  },
  gemm_n128_k2048: {
    default: "torch_matmul_317103",
  },
  gemm_n2048_k4096: {
    default: "torch_matmul_926adc",
  },
  gemm_n256_k7168: {
    default: "torch_matmul_67278e",
  },
  gemm_n28672_k4096: {
    default: "torch_matmul_655587",
  },
  gemm_n4096_k14336: {
    default: "torch_matmul_254647",
  },
  gemm_n4096_k4096: {
    default: "torch_matmul_0d13df",
  },
  gemm_n5120_k2048: {
    default: "torch_matmul_075b0d",
  },
  gemm_n6144_k4096: {
    default: "torch_matmul_3b6488",
  },
  gqa_paged_decode_h32_kv4_d128_ps1: {
    default: "flashinfer_wrapper_78fd04",
  },
  gqa_paged_decode_h32_kv8_d128_ps1: {
    default: "flashinfer_wrapper_a9588f",
  },
  gqa_paged_prefill_causal_h32_kv4_d128_ps1: {
    default: "flashinfer_wrapper_71bd33",
  },
  gqa_paged_prefill_causal_h32_kv8_d128_ps1: {
    default: "flashinfer_wrapper_8cad92",
  },
  gqa_ragged_prefill_causal_h32_kv4_d128: {
    default: "flashinfer_wrapper_acea60",
  },
  gqa_ragged_prefill_causal_h32_kv8_d128: {
    default: "flashinfer_wrapper_f9a07b",
  },
  mla_paged_decode_h16_ckv512_kpe64_ps1: {
    default: "flashinfer_wrapper_03f7b0",
  },
  mla_paged_prefill_causal_h16_ckv512_kpe64_ps1: {
    default: "flashinfer_wrapper_ea3787",
  },
  moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048: {
    default: "flashinfer_moe",
  },
  rmsnorm_h128: {
    default: "flashinfer_wrapper_57c111",
  },
  rmsnorm_h1536: {
    default: "flashinfer_wrapper_a27dc7",
  },
  rmsnorm_h2048: {
    default: "flashinfer_wrapper_0af255",
  },
  rmsnorm_h4096: {
    default: "flashinfer_wrapper_2e27cd",
  },
  rmsnorm_h512: {
    default: "flashinfer_wrapper_846dc8",
  },
  rmsnorm_h7168: {
    default: "flashinfer_wrapper_5d67c6",
  },
  top_k_sampling_from_probs_v128256: {
    default: "flashinfer_wrapper_d86b24bd",
  },
  top_k_sampling_from_probs_v129280: {
    default: "flashinfer_wrapper_4ec4ec35",
  },
  top_k_sampling_from_probs_v151936: {
    default: "flashinfer_wrapper_9c1e50fa",
  },
  top_k_top_p_sampling_from_probs_v128256: {
    default: "flashinfer_wrapper_211bdd6e",
  },
  top_k_top_p_sampling_from_probs_v129280: {
    default: "flashinfer_wrapper_a4e1e7cf",
  },
  top_k_top_p_sampling_from_probs_v151936: {
    default: "flashinfer_wrapper_0bb9995b",
  },
  top_p_sampling_from_probs_v128256: {
    default: "flashinfer_wrapper_5df4fa0b",
  },
  top_p_sampling_from_probs_v129280: {
    default: "flashinfer_wrapper_4b28093b",
  },
  top_p_sampling_from_probs_v151936: {
    default: "flashinfer_wrapper_32ca24af",
  },
} as const satisfies Record<string, Record<string, string>>

export default baselines
