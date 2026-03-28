#!/usr/bin/env python3
"""Rewrite gqa_paged reference implementations to be vectorized (GPU, batched ops)."""
import json
import os

DEFS_DIR = os.path.join(os.path.dirname(__file__), "../tmp/flashinfer-trace/definitions/gqa_paged")

# ─── Reference templates ─────────────────────────────────────────────────────

DECODE_PS1_TEMPLATE = """\
import torch
import math


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    # Check constants
    assert num_qo_heads == {H}
    assert num_kv_heads == {KV}
    assert head_dim == 128
    assert page_size == 1

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

    gqa_ratio = num_qo_heads // num_kv_heads
    # page_size=1: squeeze page dim -> [num_pages, num_kv_heads, head_dim]
    k_flat = k_cache.squeeze(1).to(torch.float32)
    v_flat = v_cache.squeeze(1).to(torch.float32)
    q_f32 = q.to(torch.float32)

    for b in range(batch_size):
        ps = int(kv_indptr[b].item())
        pe = int(kv_indptr[b + 1].item())
        if ps >= pe:
            output[b].zero_()
            continue

        idx = kv_indices[ps:pe].to(torch.long)
        # k/v: [num_qo_heads, T, head_dim] (kv heads expanded to match qo heads)
        k = k_flat[idx].permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v = v_flat[idx].permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        q_b = q_f32[b].unsqueeze(1)  # [num_qo_heads, 1, head_dim]

        logits = torch.bmm(q_b, k.transpose(1, 2)).squeeze(1) * sm_scale  # [H, T]
        lse[b] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)  # [H, T]
        output[b] = torch.bmm(attn.unsqueeze(1), v).squeeze(1).to(torch.bfloat16)

    return output, lse"""

DECODE_PS64_TEMPLATE = """\
import torch
import math


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    # Check constants
    assert num_qo_heads == {H}
    assert num_kv_heads == {KV}
    assert head_dim == 128
    assert page_size == {PS}

    device = q.device
    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)
    q_f32 = q.to(torch.float32)

    for b in range(batch_size):
        ps = int(kv_indptr[b].item())
        pe = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())
        if ps >= pe:
            output[b].zero_()
            continue

        page_ids = kv_indices[ps:pe].to(torch.long)
        num_full_pages = len(page_ids) - 1

        # Gather tokens: full pages flat + last partial page
        if num_full_pages > 0:
            k_full = k_cache_f32[page_ids[:num_full_pages]].reshape(-1, num_kv_heads, head_dim)
            v_full = v_cache_f32[page_ids[:num_full_pages]].reshape(-1, num_kv_heads, head_dim)
        else:
            k_full = torch.empty(0, num_kv_heads, head_dim, device=device)
            v_full = torch.empty(0, num_kv_heads, head_dim, device=device)
        k_tokens = torch.cat([k_full, k_cache_f32[page_ids[-1], :last_len]], dim=0)
        v_tokens = torch.cat([v_full, v_cache_f32[page_ids[-1], :last_len]], dim=0)

        # [num_kv_heads, T, D] -> expand to [num_qo_heads, T, D]
        k = k_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v = v_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        q_b = q_f32[b].unsqueeze(1)  # [num_qo_heads, 1, head_dim]

        logits = torch.bmm(q_b, k.transpose(1, 2)).squeeze(1) * sm_scale  # [H, T]
        lse[b] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)
        output[b] = torch.bmm(attn.unsqueeze(1), v).squeeze(1).to(torch.bfloat16)

    return output, lse"""

PREFILL_PS1_TEMPLATE = """\
import torch
import math

CHUNK_Q = 512  # chunk query tokens to bound peak memory for large prefills


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = int(qo_indptr.shape[0]) - 1

    # Check constants
    assert num_qo_heads == {H}
    assert num_kv_heads == {KV}
    assert head_dim == 128
    assert page_size == 1

    device = q.device
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    # page_size=1: squeeze page dim -> [num_pages, num_kv_heads, head_dim]
    k_flat = k_cache.squeeze(1).to(torch.float32)
    v_flat = v_cache.squeeze(1).to(torch.float32)

    for b in range(batch_size):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        kvs = int(kv_indptr[b].item())
        kve = int(kv_indptr[b + 1].item())
        if qs >= qe or kvs >= kve:
            continue

        page_ids = kv_indices[kvs:kve].to(torch.long)
        k = k_flat[page_ids]  # [num_kv, num_kv_heads, head_dim]
        v = v_flat[page_ids]
        num_kv = k.shape[0]
        num_q = qe - qs
        delta = num_kv - num_q  # causal offset: q_i can attend to kv_j if j <= i + delta

        # Expand KV heads: [num_qo_heads, num_kv, head_dim]
        k_exp = k.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v_exp = v.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        kv_pos = torch.arange(num_kv, device=device)

        for chunk_start in range(0, num_q, CHUNK_Q):
            chunk_end = min(chunk_start + CHUNK_Q, num_q)
            q_chunk = q_f32[qs + chunk_start:qs + chunk_end]  # [cq, num_qo_heads, head_dim]

            # logits: [num_qo_heads, cq, num_kv]
            logits = torch.einsum("qhd,hkd->hqk", q_chunk, k_exp) * sm_scale

            # Causal mask: kv_pos > q_idx + delta  =>  mask out future tokens
            q_pos = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(1)  # [cq, 1]
            mask = kv_pos.unsqueeze(0) > q_pos + delta  # [cq, num_kv]
            logits.masked_fill_(mask.unsqueeze(0), float("-inf"))

            lse[qs + chunk_start:qs + chunk_end] = (
                torch.logsumexp(logits, dim=-1) / math.log(2.0)
            ).permute(1, 0)  # [cq, num_qo_heads]

            attn = torch.softmax(logits, dim=-1)  # [num_qo_heads, cq, num_kv]
            output[qs + chunk_start:qs + chunk_end] = torch.einsum(
                "hqk,hkd->qhd", attn, v_exp
            ).to(torch.bfloat16)

    return output, lse"""

PREFILL_PS64_TEMPLATE = """\
import torch
import math

CHUNK_Q = 512  # chunk query tokens to bound peak memory for large prefills


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    batch_size = int(qo_indptr.shape[0]) - 1

    # Check constants
    assert num_qo_heads == {H}
    assert num_kv_heads == {KV}
    assert head_dim == 128
    assert page_size == {PS}

    device = q.device
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(batch_size):
        qs = int(qo_indptr[b].item())
        qe = int(qo_indptr[b + 1].item())
        kvs = int(kv_indptr[b].item())
        kve = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())
        if qs >= qe or kvs >= kve:
            continue

        page_ids = kv_indices[kvs:kve].to(torch.long)
        num_full_pages = len(page_ids) - 1

        # Gather tokens from full pages and last partial page
        if num_full_pages > 0:
            k_full = k_cache_f32[page_ids[:num_full_pages]].reshape(-1, num_kv_heads, head_dim)
            v_full = v_cache_f32[page_ids[:num_full_pages]].reshape(-1, num_kv_heads, head_dim)
        else:
            k_full = torch.empty(0, num_kv_heads, head_dim, device=device)
            v_full = torch.empty(0, num_kv_heads, head_dim, device=device)
        k_tokens = torch.cat([k_full, k_cache_f32[page_ids[-1], :last_len]], dim=0)
        v_tokens = torch.cat([v_full, v_cache_f32[page_ids[-1], :last_len]], dim=0)

        num_kv = k_tokens.shape[0]
        num_q = qe - qs
        delta = num_kv - num_q  # causal offset

        # Expand KV heads: [num_qo_heads, num_kv, head_dim]
        k_exp = k_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        v_exp = v_tokens.permute(1, 0, 2).repeat_interleave(gqa_ratio, dim=0)
        kv_pos = torch.arange(num_kv, device=device)

        for chunk_start in range(0, num_q, CHUNK_Q):
            chunk_end = min(chunk_start + CHUNK_Q, num_q)
            q_chunk = q_f32[qs + chunk_start:qs + chunk_end]  # [cq, num_qo_heads, head_dim]

            # logits: [num_qo_heads, cq, num_kv]
            logits = torch.einsum("qhd,hkd->hqk", q_chunk, k_exp) * sm_scale

            # Causal mask
            q_pos = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(1)
            mask = kv_pos.unsqueeze(0) > q_pos + delta
            logits.masked_fill_(mask.unsqueeze(0), float("-inf"))

            lse[qs + chunk_start:qs + chunk_end] = (
                torch.logsumexp(logits, dim=-1) / math.log(2.0)
            ).permute(1, 0)

            attn = torch.softmax(logits, dim=-1)
            output[qs + chunk_start:qs + chunk_end] = torch.einsum(
                "hqk,hkd->qhd", attn, v_exp
            ).to(torch.bfloat16)

    return output, lse"""

# ─── File → (template, constants) mapping ────────────────────────────────────

FILES = {
    # decode ps1
    "gqa_paged_decode_h20_kv4_d128_ps1.json":   (DECODE_PS1_TEMPLATE,  dict(H=20, KV=4,  PS=1)),
    "gqa_paged_decode_h32_kv4_d128_ps1.json":   (DECODE_PS1_TEMPLATE,  dict(H=32, KV=4,  PS=1)),
    "gqa_paged_decode_h32_kv8_d128_ps1.json":   (DECODE_PS1_TEMPLATE,  dict(H=32, KV=8,  PS=1)),
    "gqa_paged_decode_h32_kv16_d128_ps1.json":  (DECODE_PS1_TEMPLATE,  dict(H=32, KV=16, PS=1)),
    # decode ps64
    "gqa_paged_decode_h20_kv4_d128_ps64.json":  (DECODE_PS64_TEMPLATE, dict(H=20, KV=4,  PS=64)),
    "gqa_paged_decode_h32_kv4_d128_ps64.json":  (DECODE_PS64_TEMPLATE, dict(H=32, KV=4,  PS=64)),
    "gqa_paged_decode_h32_kv8_d128_ps64.json":  (DECODE_PS64_TEMPLATE, dict(H=32, KV=8,  PS=64)),
    "gqa_paged_decode_h32_kv16_d128_ps64.json": (DECODE_PS64_TEMPLATE, dict(H=32, KV=16, PS=64)),
    # prefill ps1
    "gqa_paged_prefill_causal_h20_kv4_d128_ps1.json":  (PREFILL_PS1_TEMPLATE,  dict(H=20, KV=4,  PS=1)),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps1.json":  (PREFILL_PS1_TEMPLATE,  dict(H=32, KV=4,  PS=1)),
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1.json":  (PREFILL_PS1_TEMPLATE,  dict(H=32, KV=8,  PS=1)),
    "gqa_paged_prefill_causal_h32_kv16_d128_ps1.json": (PREFILL_PS1_TEMPLATE,  dict(H=32, KV=16, PS=1)),
    # prefill ps64
    "gqa_paged_prefill_causal_h20_kv4_d128_ps64.json":  (PREFILL_PS64_TEMPLATE, dict(H=20, KV=4,  PS=64)),
    "gqa_paged_prefill_causal_h32_kv4_d128_ps64.json":  (PREFILL_PS64_TEMPLATE, dict(H=32, KV=4,  PS=64)),
    "gqa_paged_prefill_causal_h32_kv8_d128_ps64.json":  (PREFILL_PS64_TEMPLATE, dict(H=32, KV=8,  PS=64)),
    "gqa_paged_prefill_causal_h32_kv16_d128_ps64.json": (PREFILL_PS64_TEMPLATE, dict(H=32, KV=16, PS=64)),
}


def main():
    for fname, (template, consts) in FILES.items():
        path = os.path.join(DEFS_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        new_ref = template.format(**consts)
        data["reference"] = new_ref
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Updated {fname}")


if __name__ == "__main__":
    main()
