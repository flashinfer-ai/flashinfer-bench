import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 8
    assert num_kv_heads == 1
    assert head_dim == 256
    assert page_size == 1
    assert total_q == qo_indptr[-1].item()

    device = q.device
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or kv_start >= kv_end:
            continue

        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]
        v_batch = v_cache_flat[page_ids]
        q_batch = q_f32[q_start:q_end]
        num_q_tokens = q_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue

            q_pos = q_batch[q_idx]
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio
                q_head = q_pos[h]
                k_head = k_batch[:max_kv_idx, kv_head]
                v_head = v_batch[:max_kv_idx, kv_head]

                logits = torch.matmul(q_head, k_head.T)
                logits_scaled = logits * sm_scale
                lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits_scaled, dim=-1)
                out_head = torch.matmul(attn, v_head)
                output[global_q_idx, h] = out_head.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    max_pages,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    page_size=1,
    causal=True,
    device="cuda",
):
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        if causal:
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()
        else:
            kv_lens[i] = torch.randint(1, max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = qo_indptr[-1].item()
    num_kv_indices = kv_indptr[-1].item()

    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]
    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        kv_indices[idx : idx + seq_len] = all_page_ids[idx : idx + seq_len]
        idx += seq_len

    k_cache = torch.randn(
        max_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        max_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    q = torch.randn(total_q, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)
    causal_t = torch.tensor(causal, dtype=torch.bool, device=device)
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "num_kv_indices": num_kv_indices,
        "sm_scale": sm_scale,
        "causal": causal_t,
        "page_size": page_size,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, causal=True, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Prefill Causal h8_kv1_d256: "
        f"batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 256
    page_size = 1
    max_pages = max_kv_len * batch_size * 2

    inputs = generate_random_inputs(
        batch_size, max_q_len, max_kv_len, max_pages,
        num_attention_heads, num_key_value_heads, head_dim, page_size, causal, device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Total query tokens: {inputs['total_q']}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"], inputs["k_cache"], inputs["v_cache"],
        inputs["qo_indptr"], inputs["kv_indptr"], inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        causal=inputs["causal"].item(),
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    # Compare outputs
    print("\nComparing outputs...")
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    rel_diff = abs_diff / (torch.abs(fi_output_f32) + 1e-8)

    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"Max relative difference: {rel_diff.max().item():.6e}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6e}")
    print(f"Mean relative difference: {rel_diff.mean().item():.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_o_f32.flatten(), fi_output_f32.flatten(), dim=0
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    # LSE comparison
    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    print(f"\nLSE max absolute difference: {lse_abs_diff.max().item():.6e}")
    print(f"LSE mean absolute difference: {lse_abs_diff.mean().item():.6e}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")

    return all_close


def main():
    print("Testing GQA Paged Prefill Causal h8_kv1_d256_ps1 Reference Implementation")

    test_configs = [
        (1, 8, 16, True),
        (4, 16, 32, True),
        (8, 32, 64, True),
        (16, 64, 128, True),
    ]

    passed = 0
    total = len(test_configs)

    for batch_size, max_q_len, max_kv_len, causal in test_configs:
        try:
            if test_correctness(batch_size, max_q_len, max_kv_len, causal):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
