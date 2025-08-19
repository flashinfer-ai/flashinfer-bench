import math

import flashinfer
import torch


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim_qk = q.shape
    total_kv = k.shape[0]
    head_dim_vo = v.shape[-1]
    num_indptr = qo_indptr.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_qk == 192
    assert head_dim_vo == 128

    # Check constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()
    assert total_q == total_kv

    device = q.device

    out = torch.zeros((total_q, num_qo_heads, head_dim_vo), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    for b in range(num_indptr - 1):
        q0, q1 = int(qo_indptr[b].item()), int(qo_indptr[b + 1].item())
        k0, k1 = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
        if q0 >= q1 or k0 >= k1:
            continue

        qb = q[q0:q1]  # [q_len, num_qo_heads, head_dim_qk]
        kb = k[k0:k1]  # [kv_len, num_kv_heads, head_dim_qk]
        vb = v[k0:k1]  # [kv_len, num_kv_heads, head_dim_vo]
        q_len = qb.shape[0]
        kv_len = kb.shape[0]

        logits = torch.einsum("qhd,khd->qhk", qb, kb)  # [q_len, num_qo_heads, kv_len]
        logits_scaled = logits * sm_scale

        # Causal mask, no need to allow offset because indptrs should be identical
        i = torch.arange(q_len, device=device).unsqueeze(-1)  # [q_len, 1]
        j = torch.arange(kv_len, device=device).unsqueeze(0)  # [1, kv_len]
        logits_scaled.masked_fill_((j > i).unsqueeze(1), float("-inf"))

        # Compute 2-base LSE
        lse[q0:q1] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [q_len, num_qo_heads, kv_len]
        out_b = torch.einsum("qhk,khd->qhd", attn, vb)  # [q_len, num_qo_heads, head_dim_vo]
        out[q0:q1] = out_b.to(torch.bfloat16)

    return {"output": out, "lse": lse}


def generate_random_inputs(
    batch_size,
    max_q_len,
    max_kv_len,
    num_heads=16,
    head_dim_qk=192,
    head_dim_vo=128,
    device="cuda",
):
    max_len = int(min(max_q_len, max_kv_len))
    q_lens = torch.randint(1, max_len + 1, (batch_size,), dtype=torch.int32)

    kv_lens = q_lens.clone()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0)
    kv_indptr = qo_indptr.clone()

    qo_indptr = qo_indptr.to(device)
    kv_indptr = kv_indptr.to(device)

    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())

    q = torch.randn(total_q, num_heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, num_heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, num_heads, head_dim_vo, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim_qk)

    return {
        "q": q,
        "k": k,
        "v": v,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "total_kv": total_kv,
        "sm_scale": sm_scale,
        "num_heads": num_heads,
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "device": device,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(
        f"Testing MLA Ragged Prefill (no absorption) "
        f"bs={batch_size}, max_q={max_q_len}, max_kv={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return True  # skip as pass to not break CI

    H = 16
    Dqk = 192
    Dvo = 128

    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, H, Dqk, Dvo, device)
    print(f"q_lens: {inputs['q_lens'].cpu().numpy()}")
    print(f"kv_lens:{inputs['kv_lens'].cpu().numpy()}")
    print(f"total_q={inputs['total_q']}, total_kv={inputs['total_kv']}")

    print("\nRunning reference...")
    ref = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["sm_scale"],
    )
    ref_o = ref["output"]
    ref_lse = ref["lse"]

    print("\nSetting up FlashInfer (ragged prefill wrapper)...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )
    prefill.plan(
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=Dqk,
        head_dim_vo=Dvo,
        causal=True,
        sm_scale=inputs["sm_scale"],
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_o, fi_lse = prefill.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

    print("\nComparing outputs...")
    print(f"Reference output shape: {ref_o.shape}")
    print(f"FlashInfer output shape: {fi_o.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")
    print(f"FlashInfer LSE shape: {fi_lse.shape}")

    abs_diff = (ref_o - fi_o).abs()
    rel_diff = abs_diff / (fi_o.abs() + 1e-8)

    print(
        f"Output: max_abs={abs_diff.max().item():.6e}, "
        f"max_rel={rel_diff.max().item():.6e}, "
        f"mean_abs={abs_diff.mean().item():.6e}, "
        f"mean_rel={rel_diff.mean().item():.6e}"
    )

    cos_sim = torch.nn.functional.cosine_similarity(ref_o.flatten(), fi_o.flatten(), dim=0).item()
    mse = torch.mean((ref_o - fi_o) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}, MSE: {mse:.6e}")

    print("\nComparing LSE (log2)...")
    lse_abs = (ref_lse - fi_lse).abs()
    lse_rel = lse_abs / (fi_lse.abs() + 1e-8)
    print(
        f"LSE:    max_abs={lse_abs.max().item():.6e}, "
        f"max_rel={lse_rel.max().item():.6e}, "
        f"mean_abs={lse_abs.mean().item():.6e}, "
        f"mean_rel={lse_rel.mean().item():.6e}"
    )

    ok_out = torch.allclose(ref_o, fi_o, atol=atol, rtol=rtol)
    ok_lse = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_ok = ok_out and ok_lse

    if all_ok:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
        if not ok_out:
            flat = abs_diff.flatten()
            top_k = min(5, flat.numel())
            vals, idxs = torch.topk(flat, top_k)
            print("\nTop output diffs:")
            for i in range(top_k):
                idx = idxs[i].item()
                q_idx = idx // (H * Dvo)
                h_idx = (idx % (H * Dvo)) // Dvo
                d_idx = idx % Dvo
                print(
                    f"  [q={q_idx}, h={h_idx}, d={d_idx}] "
                    f"ref={ref_o.flatten()[idx].item():.6f} "
                    f"fi={fi_o.flatten()[idx].item():.6f} "
                    f"diff={vals[i].item():.6e}"
                )
        if not ok_lse:
            flat = lse_abs.flatten()
            top_k = min(5, flat.numel())
            vals, idxs = torch.topk(flat, top_k)
            print("\nTop LSE diffs:")
            for i in range(top_k):
                idx = idxs[i].item()
                q_idx = idx // H
                h_idx = idx % H
                print(
                    f"  [q={q_idx}, h={h_idx}] "
                    f"ref={ref_lse.flatten()[idx].item():.6f} "
                    f"fi={fi_lse.flatten()[idx].item():.6f} "
                    f"diff={vals[i].item():.6e}"
                )
    return all_ok


def main():
    print("Testing MLA Ragged Prefill (no absorption) with FlashInfer")
    test_cfgs = [
        (1, 8, 16),
        (4, 16, 32),
        (8, 32, 64),
        (16, 64, 128),
        (32, 128, 256),
    ]
    passed = 0
    for bs, mq, mkv in test_cfgs:
        try:
            if test_correctness(bs, mq, mkv):
                passed += 1
        except Exception as e:
            print(f"✗ Exception: {e}")
            import traceback

            traceback.print_exc()
    total = len(test_cfgs)
    print(f"\n{'='*60}\nSummary: {passed}/{total} tests passed\n{'='*60}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
