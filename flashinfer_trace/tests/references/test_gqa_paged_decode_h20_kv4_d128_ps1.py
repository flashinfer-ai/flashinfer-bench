"""Reference test for gqa_paged_decode_h20_kv4_d128_ps1 (Qwen3 14B TP=2)."""

import math

import flashinfer
import numpy as np
import torch

NUM_QO_HEADS = 20
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 1


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
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

    # For page_size=1, each page stores exactly one token
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

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

        k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio

            q_head = q_batch[h]  # [head_dim]
            k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
            v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]

            logits = torch.matmul(q_head, k_head.T) * sm_scale
            lse[b, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1)
            output[b, h] = torch.matmul(attn, v_head).to(torch.bfloat16)

    return output, lse


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for testing."""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_pages = seq_lens.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    num_pages = total_pages + 100
    k_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
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


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Decode h20/kv4 ps1 (Qwen3 14B TP=2): batch_size={batch_size}, max_seq_len={max_seq_len}"
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
    print("Testing GQA Paged Decode h20/kv4/ps1 (Qwen3 14B TP=2)")

    test_configs = [(1, 16), (4, 32), (8, 64), (16, 128)]
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
