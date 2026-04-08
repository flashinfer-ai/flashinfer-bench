import math

import flashinfer
import pytest
import torch


@torch.no_grad()
def run_reference(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    """Pure-PyTorch reference implementation for gqa_paged_decode_h16_kv2_d64_ps64."""
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    # Check constants
    assert num_qo_heads == 16
    assert num_kv_heads == 2
    assert head_dim == 64
    assert page_size == 64

    # Check constraints
    assert kv_indptr.shape[0] == batch_size + 1
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads  # 8
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())

        if page_start >= page_end:
            output[b].zero_()
            continue

        page_ids = kv_indices[page_start:page_end].to(torch.long)
        num_full_pages = len(page_ids) - 1
        k_tokens, v_tokens = [], []
        for pi, pid in enumerate(page_ids):
            valid = page_size if pi < num_full_pages else last_len
            k_tokens.append(k_cache_f32[pid, :valid])
            v_tokens.append(v_cache_f32[pid, :valid])

        k_batch = torch.cat(k_tokens, dim=0)   # [total_tokens, num_kv_heads, head_dim]
        v_batch = torch.cat(v_tokens, dim=0)
        q_batch = q[b].to(torch.float32)       # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio
            q_head = q_batch[h]                # [head_dim]
            k_head = k_batch[:, kv_head]       # [total_tokens, head_dim]
            v_head = v_batch[:, kv_head]

            logits = torch.matmul(q_head, k_head.T) * sm_scale
            lse[b, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1)
            output[b, h] = torch.matmul(attn, v_head).to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_qo_heads=16,
    num_kv_heads=2,
    head_dim=64,
    page_size=64,
    device="cuda",
):
    """Generate random paged-KV decode inputs."""
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    pages_per_seq = (seq_lens + page_size - 1) // page_size
    total_pages = int(pages_per_seq.sum().item())

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)

    kv_last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

    num_pages = max(total_pages, 1)
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim,
                          dtype=torch.bfloat16, device=device)
    q = torch.randn(batch_size, num_qo_heads, head_dim,
                    dtype=torch.bfloat16, device=device)

    sm_scale = torch.tensor(1.0 / math.sqrt(head_dim), dtype=torch.float32)

    return q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale


def run_flashinfer(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    """FlashInfer BatchDecodeWithPagedKVCacheWrapper — GQA ratio=8 (power of 2)."""
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    device = q.device

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    wrapper.plan(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=float(sm_scale),
    )
    output, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
    return output, lse


@pytest.mark.parametrize("batch_size,max_seq_len", [
    (1, 64),
    (4, 128),
    (8, 256),
    (16, 512),
])
def test_gqa_paged_decode_h16_kv2_d64_ps64(batch_size, max_seq_len):
    """Test that FlashInfer decode output matches the pure-PyTorch reference."""
    device = "cuda"
    atol, rtol = 1e-2, 1e-2

    q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale = (
        generate_random_inputs(batch_size, max_seq_len, device=device)
    )

    ref_output, ref_lse = run_reference(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )
    fi_output, fi_lse = run_flashinfer(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale
    )

    ref_f32 = ref_output.to(torch.float32)
    fi_f32 = fi_output.to(torch.float32)

    max_abs_diff = (ref_f32 - fi_f32).abs().max().item()
    print(f"  batch={batch_size} max_seq={max_seq_len} "
          f"max_abs_diff={max_abs_diff:.4e}")

    assert torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol), (
        f"Output mismatch: max_abs_diff={max_abs_diff:.4e}"
    )
    assert torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol), (
        f"LSE mismatch: max_abs_diff={(ref_lse - fi_lse).abs().max().item():.4e}"
    )


if __name__ == "__main__":
    print("Testing gqa_paged_decode_h16_kv2_d64_ps64")
    configs = [(1, 64), (4, 128), (8, 256), (16, 512)]
    passed = 0
    for bs, ms in configs:
        try:
            test_gqa_paged_decode_h16_kv2_d64_ps64(bs, ms)
            print(f"  ✓ batch={bs} max_seq={ms}")
            passed += 1
        except Exception as e:
            print(f"  ✗ batch={bs} max_seq={ms}: {e}")
    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{len(configs)} tests passed")
    if passed == len(configs):
        print("✓ All tests passed!")
