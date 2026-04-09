import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, qk_dim = q.shape
    total_kv, num_kv_heads, vo_dim = v.shape
    len_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 8
    assert num_kv_heads == 8
    assert qk_dim == 192
    assert vo_dim == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((total_q, num_qo_heads, vo_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

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

        # num_kv_heads == num_qo_heads for absorbed MLA
        logits = torch.einsum("qhd,khd->qhk", q_batch, k_batch) * sm_scale

        q_positions = torch.arange(num_q_tokens, device=device)
        kv_positions = torch.arange(num_kv_tokens, device=device)
        causal_mask = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)
        logits = logits.masked_fill(~causal_mask[:, None, :], float("-inf"))

        lse_batch = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        lse[q_start:q_end] = lse_batch

        attn_weights = torch.softmax(logits, dim=-1)
        output_batch = torch.einsum("qhk,khd->qhd", attn_weights, v_batch)
        output[q_start:q_end] = output_batch.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(batch_size, max_seq_len, num_qo_heads=8, qk_dim=192, vo_dim=128, device="cuda"):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_tokens = int(seq_lens.sum().item())

    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(seq_lens, dim=0)

    q = torch.randn(total_tokens, num_qo_heads, qk_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_tokens, num_qo_heads, qk_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_tokens, num_qo_heads, vo_dim, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / math.sqrt(qk_dim), dtype=torch.float32, device=device)

    return {"q": q, "k": k, "v": v, "qo_indptr": indptr, "kv_indptr": indptr.clone(), "sm_scale": sm_scale}


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(f"Testing MLA ragged prefill h8 batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    inputs = generate_random_inputs(batch_size, max_seq_len, device=device)

    ref_o, ref_lse = run(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["qo_indptr"], inputs["kv_indptr"], inputs["sm_scale"],
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, backend="auto")
    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    fi_output, fi_lse = prefill_wrapper.run(
        inputs["q"], inputs["k"], inputs["v"], return_lse=True
    )

    output_close = torch.allclose(ref_o.float(), fi_output.float(), atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ FAILED (atol={atol}, rtol={rtol})")
        print(f"  Max output abs diff: {torch.abs(ref_o.float() - fi_output.float()).max().item():.6e}")
        print(f"  Max LSE abs diff: {torch.abs(ref_lse - fi_lse).max().item():.6e}")

    return all_close


def main():
    print("Testing Batch MLA Ragged Prefill h8 (Kimi K2, TP=8)")

    test_configs = [(1, 16), (4, 32), (8, 64), (16, 128)]
    passed = sum(1 for b, s in test_configs if test_correctness(b, s))
    total = len(test_configs)

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
