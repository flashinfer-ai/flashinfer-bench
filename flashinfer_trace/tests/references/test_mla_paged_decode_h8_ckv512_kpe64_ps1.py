import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 8
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            output[b].zero_()
            continue

        pages = kv_indices[page_beg:page_end]
        L_tokens = page_end - page_beg

        if L_tokens <= 0 or pages.numel() == 0:
            output[b].zero_()
            continue

        tok_idx = pages[:L_tokens].to(torch.long)

        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse


def generate_random_inputs(
    batch_size,
    max_seq_len,
    num_qo_heads=8,
    head_dim_ckv=512,
    head_dim_kpe=64,
    page_size=1,
    device="cuda",
):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_pages_needed = seq_lens.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)
    kv_len_arr = seq_lens.clone()

    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    sm_scale = torch.tensor(1.0 / np.sqrt(128 + head_dim_kpe), dtype=torch.float32, device=device)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr,
        "sm_scale": sm_scale,
        "qo_indptr": qo_indptr,
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(f"Testing MLA paged decode h8 batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True

    num_qo_heads = 8
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1

    inputs = generate_random_inputs(
        batch_size, max_seq_len, num_qo_heads, head_dim_ckv, head_dim_kpe, page_size, device
    )

    ref_o, ref_lse = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")
    mla_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"],
        kv_len_arr=inputs["kv_len_arr"],
        num_heads=num_qo_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    fi_output, fi_lse = mla_wrapper.run(
        inputs["q_nope"], inputs["q_pe"], inputs["ckv_cache"], inputs["kpe_cache"], return_lse=True
    )

    output_close = torch.allclose(ref_o.float(), fi_output.float(), atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ FAILED (atol={atol}, rtol={rtol})")
        abs_diff = torch.abs(ref_o.float() - fi_output.float())
        print(f"  Max output abs diff: {abs_diff.max().item():.6e}")
        print(f"  Max LSE abs diff: {torch.abs(ref_lse - fi_lse).max().item():.6e}")

    return all_close


def main():
    print("Testing Batch MLA Paged Decode h8 (Kimi K2, TP=8)")

    test_configs = [(1, 16), (4, 32), (8, 64), (16, 128), (32, 256)]
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
