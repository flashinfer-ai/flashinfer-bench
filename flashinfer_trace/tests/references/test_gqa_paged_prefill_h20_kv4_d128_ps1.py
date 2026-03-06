"""Reference test for gqa_paged_prefill_causal_h20_kv4_d128_ps1 (Qwen3 14B TP=2)."""
import math

import flashinfer
import torch


NUM_QO_HEADS = 20
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 1


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert num_kv_heads == NUM_KV_HEADS
    assert head_dim == HEAD_DIM
    assert page_size == PAGE_SIZE

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
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
        k_batch = k_cache_flat[page_ids]  # [num_kv_tokens, num_kv_heads, head_dim]
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

                logits = torch.matmul(q_head, k_head.T) * sm_scale
                lse[global_q_idx, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits, dim=-1)
                output[global_q_idx, h] = torch.matmul(attn, v_head).to(torch.bfloat16)

    return output, lse


def generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device="cuda"):
    """Generate random inputs for paged prefill testing."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    num_kv_indices = int(kv_indptr[-1].item())

    # For page_size=1, each KV token occupies one page
    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]
    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = int(kv_lens[i].item())
        kv_indices[idx:idx + seq_len] = all_page_ids[idx:idx + seq_len]
        idx += seq_len

    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    k_cache = torch.randn(max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

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
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing GQA Paged Prefill h20/kv4 ps1 (Qwen3 14B TP=2): batch={batch_size}, max_q={max_q_len}, max_kv={max_kv_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    max_pages = max_kv_len * batch_size * 2
    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device)

    print(f"Query lengths: {inputs['q_lens'].numpy()}")
    print(f"KV lengths: {inputs['kv_lens'].numpy()}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
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
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=True,
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    # Compare
    print("\nComparing outputs...")
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    print(f"Output max abs diff: {abs_diff.max().item():.6e}")
    print(f"Output mean abs diff: {abs_diff.mean().item():.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    print(f"LSE max abs diff: {lse_abs_diff.max().item():.6e}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: output_close={output_close}, lse_close={lse_close}")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing GQA Paged Prefill h20/kv4/ps1 (Qwen3 14B TP=2)")

    test_configs = [(1, 16, 32), (4, 32, 64), (8, 64, 128)]
    passed = 0
    for batch_size, max_q_len, max_kv_len in test_configs:
        try:
            if test_correctness(batch_size, max_q_len, max_kv_len):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(test_configs)} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
