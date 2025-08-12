import math

import flashinfer
import numpy as np
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, kv_len_arr, sm_scale):
    total_q, num_attention_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    page_size = ckv_cache.shape[1]
    batch_size = kv_len_arr.shape[0]

    # Check constants
    assert num_attention_heads == 128
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    # Check constraints
    assert qo_indptr.shape[0] == batch_size + 1
    assert kv_indptr.shape[0] == batch_size + 1

    device = q_nope.device

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_q, num_attention_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_attention_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for b in range(batch_size):
        # Get query range for this batch element
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())

        # Get KV page range for this batch element
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if q_start >= q_end or page_beg >= page_end:
            # No queries or KV for this batch element
            continue

        # Get actual KV length for this batch
        L_tokens = int(kv_len_arr[b].item())

        if L_tokens <= 0:
            continue

        # Get page indices for this batch
        pages = kv_indices[page_beg:page_end]

        # Since page_size=1, pages are token indices
        # We only need the first L_tokens
        tok_idx = pages[:L_tokens].to(torch.long)

        # Extract KV for this batch
        Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]

        # Extract queries for this batch
        q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_ckv]
        q_pe_batch = q_pe[q_start:q_end].to(torch.float32)  # [q_len, num_heads, head_dim_kpe]

        q_len = q_end - q_start

        # Compute attention scores for each query
        for i in range(q_len):
            qn = q_nope_batch[i]  # [num_heads, head_dim_ckv]
            qp = q_pe_batch[i]  # [num_heads, head_dim_kpe]

            # Compute logits with MLA decomposition
            logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_heads, L_tokens]
            logits_scaled = logits * sm_scale

            # Apply causal mask for prefill
            # Query position in the full sequence
            query_pos = q_start + i
            for j in range(L_tokens):
                # Key position in the sequence
                key_pos = j
                if query_pos < key_pos:
                    logits_scaled[:, j] = -float("inf")

            # Compute 2-base LSE
            lse[q_start + i] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            # Apply softmax and compute output
            attn = torch.softmax(logits_scaled, dim=-1)  # [num_heads, L_tokens]
            out = attn @ Kc  # [num_heads, head_dim_ckv]
            output[q_start + i] = out.to(torch.bfloat16)

    return {"output": output, "lse": lse}


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    num_attention_heads=128,
    head_dim_ckv=512,
    head_dim_kpe=64,
    page_size=1,
    device="cuda",
):
    """Generate random inputs for MLA paged prefill testing."""

    # Generate random sequence lengths for each batch
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32, device=device)
    kv_lens = torch.randint(1, max_kv_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # For prefill, ensure kv_len >= q_len for causal attention
    for i in range(batch_size):
        kv_lens[i] = max(kv_lens[i], q_lens[i])

    total_q = q_lens.sum().item()

    # Calculate total pages needed (since page_size=1, num_pages = total_kv_tokens)
    total_pages_needed = kv_lens.sum().item()

    # Generate qo_indptr based on query lengths
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0)

    # Generate kv_indptr based on KV lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

    # Generate kv_indices (page indices for each sequence)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # kv_len_arr stores the actual KV sequence lengths
    kv_len_arr = kv_lens.clone()

    # Generate query tensors with Matrix Absorption dimensions
    q_nope = torch.randn(
        total_q, num_attention_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        total_q, num_attention_heads, head_dim_kpe, dtype=torch.bfloat16, device=device
    )

    # Generate compressed KV and positional caches
    # Add some extra pages to simulate a real scenario
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate attention parameters
    # MLA uses head dimension before matrix absorption (128 + 64 = 192)
    sm_scale = 1.0 / np.sqrt(192)
    sm_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_len_arr": kv_len_arr,
        "sm_scale": sm_scale,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of MLA paged prefill reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing MLA Paged Prefill with Matrix Absorption")
    print(f"batch_size={batch_size}, max_q_len={max_q_len}, max_kv_len={max_kv_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return

    # Constants from kernel definition
    num_attention_heads = 128
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_q_len,
        max_kv_len,
        num_attention_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        device,
    )

    print(f"Generated query lengths: {inputs['q_lens'].cpu().numpy()}")
    print(f"Generated KV lengths: {inputs['kv_lens'].cpu().numpy()}")
    print(f"Total Q tokens: {inputs['total_q']}")
    print(f"Total pages used: {inputs['kv_indices'].shape[0]}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_output = run(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["ckv_cache"],
        inputs["kpe_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_len_arr"],
        inputs["sm_scale"],
    )
    ref_o = ref_output["output"]
    ref_lse = ref_output["lse"]

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # For paged prefill with Matrix Absorption, use BatchMLAPagedAttentionWrapper
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")

    # Plan the attention computation
    mla_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        kv_indices=inputs["kv_indices"],
        kv_len_arr=inputs["kv_len_arr"],
        num_heads=num_attention_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=True,  # Causal for prefill
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Run FlashInfer
    print("Running FlashInfer...")
    flashinfer_o, flashinfer_lse = mla_wrapper.run(
        q_nope=inputs["q_nope"],
        q_pe=inputs["q_pe"],
        ckv_cache=inputs["ckv_cache"],
        kpe_cache=inputs["kpe_cache"],
        return_lse=True,
    )

    # Compare outputs
    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_o.shape}")
    print(f"FlashInfer output shape: {flashinfer_o.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")
    print(f"FlashInfer LSE shape: {flashinfer_lse.shape}")

    # Check numerical accuracy
    o_diff = torch.abs(ref_o - flashinfer_o)
    lse_diff = torch.abs(ref_lse - flashinfer_lse)

    print(f"\nOutput max diff: {o_diff.max().item():.6f}")
    print(f"Output mean diff: {o_diff.mean().item():.6f}")
    print(f"LSE max diff: {lse_diff.max().item():.6f}")
    print(f"LSE mean diff: {lse_diff.mean().item():.6f}")

    # Assert correctness
    torch.testing.assert_close(ref_o, flashinfer_o, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_lse, flashinfer_lse, atol=atol, rtol=rtol)

    print(f"\n✓ Test passed! Reference matches FlashInfer (atol={atol}, rtol={rtol})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test with various configurations
    test_configs = [
        {"batch_size": 1, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 4, "max_q_len": 32, "max_kv_len": 64},
        {"batch_size": 8, "max_q_len": 64, "max_kv_len": 128},
    ]

    for config in test_configs:
        test_correctness(**config)
