"""Reference test for gqa_ragged_prefill_causal_h8_kv1_d256 (Qwen3 Next 80B A3B TP=2)."""

import math

import flashinfer
import pytest
import torch

DEVICE = "cuda"
NUM_QO_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def reference_gqa_ragged_prefill(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        q_batch = q_f32[q_start:q_end]
        k_batch = k_f32[kv_start:kv_end]
        v_batch = v_f32[kv_start:kv_end]

        num_q_tokens = q_batch.shape[0]
        num_kv_tokens = k_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)
        v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)

        logits = torch.einsum("qhd,khd->qhk", q_batch, k_expanded) * sm_scale

        q_positions = torch.arange(num_q_tokens, device=q.device)
        kv_positions = torch.arange(num_kv_tokens, device=q.device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float("-inf"))

        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_expanded)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse


def run_flashinfer(q, k, v, qo_indptr, kv_indptr, sm_scale):
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        causal=True,
        sm_scale=float(sm_scale),
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    return wrapper.run(q, k, v, return_lse=True)


@pytest.mark.parametrize("batch_size,max_q,max_kv", [(1, 8, 16), (4, 32, 64), (8, 64, 128)])
def test_gqa_ragged_prefill_h8_kv1_d256(batch_size, max_q, max_kv):
    print(
        f"\nTesting GQA Ragged Prefill h8/kv1/d256 (Qwen3 Next 80B A3B TP=2): "
        f"batch={batch_size}, max_q={max_q}, max_kv={max_kv}"
    )
    torch.manual_seed(42)

    q_lens = torch.randint(1, max_q + 1, (batch_size,))
    kv_lens = q_lens + torch.randint(0, max_kv - max_q + 1, (batch_size,))
    total_q = int(q_lens.sum().item())
    total_kv = int(kv_lens.sum().item())

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0).to(DEVICE)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0).to(DEVICE)

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

    ref_output, ref_lse = reference_gqa_ragged_prefill(q, k, v, qo_indptr, kv_indptr, SM_SCALE)
    fi_output, fi_lse = run_flashinfer(q, k, v, qo_indptr, kv_indptr, SM_SCALE)

    out_diff = (ref_output.float() - fi_output.float()).abs()
    lse_diff = (ref_lse - fi_lse).abs()
    print(f"Output max abs diff: {out_diff.max():.6e}")
    print(f"Output mean abs diff: {out_diff.mean():.6e}")
    print(f"LSE max abs diff: {lse_diff.max():.6e}")

    assert out_diff.max() < 0.02, f"Output diff too large: {out_diff.max():.6e}"
    assert lse_diff.max() < 0.02, f"LSE diff too large: {lse_diff.max():.6e}"
    print("✓ PASSED")


if __name__ == "__main__":
    print("Testing GQA Ragged Prefill h8/kv1/d256 (Qwen3 Next 80B A3B TP=2)\n")
    for batch_size, max_q, max_kv in [(1, 8, 16), (4, 32, 64), (8, 64, 128)]:
        test_gqa_ragged_prefill_h8_kv1_d256(batch_size, max_q, max_kv)
    print("\n============================================================")
    print("Summary: 3/3 tests passed")
    print("============================================================")
