"""Reference test for gqa_paged_decode_h32_kv16_d128_ps64 (Gemma 3 27B)."""

import math

import flashinfer
import torch

NUM_QO_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
PAGE_SIZE = 64


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == NUM_QO_HEADS
    assert num_kv_heads == NUM_KV_HEADS
    assert head_dim == HEAD_DIM
    assert page_size == PAGE_SIZE

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        last_page_len = int(kv_last_page_len[b].item())

        if page_start >= page_end:
            output[b].zero_()
            continue

        page_ids = kv_indices[page_start:page_end].to(torch.long)
        num_pages_for_seq = page_ids.shape[0]

        if num_pages_for_seq == 0:
            output[b].zero_()
            continue

        num_full_pages = num_pages_for_seq - 1
        total_tokens = num_full_pages * page_size + last_page_len

        if total_tokens == 0:
            output[b].zero_()
            continue

        k_batch = torch.zeros(
            (total_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device
        )
        v_batch = torch.zeros(
            (total_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device
        )

        token_idx = 0
        for p_idx, page_id in enumerate(page_ids):
            if p_idx < num_full_pages:
                k_batch[token_idx : token_idx + page_size] = k_cache_f32[page_id]
                v_batch[token_idx : token_idx + page_size] = v_cache_f32[page_id]
                token_idx += page_size
            else:
                k_batch[token_idx : token_idx + last_page_len] = k_cache_f32[
                    page_id, :last_page_len
                ]
                v_batch[token_idx : token_idx + last_page_len] = v_cache_f32[
                    page_id, :last_page_len
                ]
                token_idx += last_page_len

        q_batch = q[b].to(torch.float32)

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio

            q_head = q_batch[h]
            k_head = k_batch[:, kv_head]
            v_head = v_batch[:, kv_head]

            logits = torch.matmul(q_head, k_head.T) * sm_scale
            lse[b, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1)
            output[b, h] = torch.matmul(attn, v_head).to(torch.bfloat16)

    return output, lse


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for testing."""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    num_pages_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_pages_per_seq.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = (seq_lens - 1) % PAGE_SIZE + 1

    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    num_cache_pages = total_pages + 100
    k_cache = torch.randn(
        num_cache_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_cache_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

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


def test_correctness(batch_size=4, max_seq_len=128, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Decode h32/kv16 ps64 (Gemma 3 27B): batch_size={batch_size}, max_seq_len={max_seq_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    inputs = generate_random_inputs(batch_size, max_seq_len, device)
    print(f"Sequence lengths: {inputs['seq_lens'].cpu().numpy()}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        inputs["sm_scale"],
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
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = decode_wrapper.run(
        inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True
    )

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
    print("Testing GQA Paged Decode h32/kv16/ps64 (Gemma 3 27B)")

    test_configs = [(1, 64), (4, 128), (8, 256), (16, 512)]
    passed = 0
    for batch_size, max_seq_len in test_configs:
        try:
            if test_correctness(batch_size, max_seq_len):
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
