import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    assert num_qo_heads == 8
    assert num_kv_heads == 1
    assert head_dim == 256
    assert page_size == 1

    device = q.device
    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_start >= page_end:
            output[b].zero_()
            continue

        token_indices = kv_indices[page_start:page_end].to(torch.long)
        num_tokens = token_indices.shape[0]
        if num_tokens == 0:
            output[b].zero_()
            continue

        k_batch = k_cache_flat[token_indices]
        v_batch = v_cache_flat[token_indices]
        q_batch = q[b].to(torch.float32)

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio
            q_head = q_batch[h]
            k_head = k_batch[:, kv_head]
            v_head = v_batch[:, kv_head]

            logits = torch.matmul(q_head, k_head.T)
            logits_scaled = logits * sm_scale
            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=-1)
            out_head = torch.matmul(attn, v_head)
            output[b, h] = out_head.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
    page_size=1,
    device="cuda",
):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_pages_needed = seq_lens.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    q = torch.randn(batch_size, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device)

    num_pages = total_pages_needed + 100
    k_cache = torch.randn(
        num_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_pages, page_size, num_key_value_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    sm_scale = 1.0 / np.sqrt(head_dim)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale,
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(f"Testing GQA Paged Decode h8_kv1_d256: batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    num_attention_heads = 8
    num_key_value_heads = 1
    head_dim = 256
    page_size = 1

    inputs = generate_random_inputs(
        batch_size, max_seq_len, num_attention_heads, num_key_value_heads,
        head_dim, page_size, device,
    )

    print(f"Generated sequences with lengths: {inputs['seq_lens'].cpu().numpy()}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"], inputs["k_cache"], inputs["v_cache"],
        inputs["kv_indptr"], inputs["kv_indices"], inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    decode_wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    fi_output, fi_lse = decode_wrapper.run(
        inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True
    )

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
    print("Testing GQA Paged Decode h8_kv1_d256_ps1 Reference Implementation")

    test_configs = [
        (1, 16),
        (4, 32),
        (8, 64),
        (16, 128),
    ]

    passed = 0
    total = len(test_configs)

    for batch_size, max_seq_len in test_configs:
        try:
            if test_correctness(batch_size, max_seq_len):
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
