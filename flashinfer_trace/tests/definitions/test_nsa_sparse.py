"""Tests for NSA (Native Sparse Attention) sparse definitions.

Note: These tests require SGLang sgl_kernel for ground truth comparison.
If not available, tests will be skipped.
"""

import sys

import numpy as np
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest
from flashinfer_bench.testing.comparators import HitRatioComparator, MultiOutputComparator

# Check SGLang availability
try:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

# Constants
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1
TOPK = 256


def generate_nsa_decode_inputs(
    batch_size: int,
    max_seq_len: int = 512,
    num_qo_heads: int = NUM_QO_HEADS,
    head_dim_ckv: int = HEAD_DIM_CKV,
    head_dim_kpe: int = HEAD_DIM_KPE,
    topk: int = TOPK,
    device: str = "cuda",
):
    """Generate random inputs for NSA sparse decode testing."""
    min_seq_len = max(topk, 256)
    seq_lens = torch.randint(
        min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    total_pages_needed = seq_lens.sum().item()

    # Generate page table
    page_table = torch.zeros(batch_size, max_seq_len, dtype=torch.int32, device=device)
    page_offset = 0
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        page_table[b, :seq_len] = torch.arange(
            page_offset, page_offset + seq_len, dtype=torch.int32, device=device
        )
        page_offset += seq_len

    # Generate sparse indices
    sparse_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        actual_topk = min(topk, seq_len)
        perm = torch.randperm(seq_len, device=device)[:actual_topk]
        selected_pages = page_table[b, perm]
        sparse_indices[b, :actual_topk] = selected_pages.to(torch.int32)

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate KV caches
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
    }


def generate_nsa_prefill_inputs(
    total_num_tokens: int,
    num_qo_heads: int = NUM_QO_HEADS,
    head_dim_ckv: int = HEAD_DIM_CKV,
    head_dim_kpe: int = HEAD_DIM_KPE,
    topk: int = TOPK,
    device: str = "cuda",
):
    """Generate random inputs for NSA sparse prefill testing."""
    num_pages = max(total_num_tokens * 2, 1024)

    sparse_indices = torch.randint(
        0, num_pages, (total_num_tokens, topk), dtype=torch.int32, device=device
    )

    q_nope = torch.randn(
        total_num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        total_num_tokens, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device
    )

    ckv_cache = torch.randn(num_pages, 1, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, 1, head_dim_kpe, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
    }


@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang sgl_kernel not available")
class TestNSASparseDecode(DefinitionTest):
    """Test NSA sparse decode with SGLang as baseline."""

    definition_path = "definitions/nsa_paged/nsa_sparse_decode_h16_ckv512_kpe64_topk256_ps1.json"
    configs = [
        {"batch_size": 1, "max_seq_len": 512},
        {"batch_size": 4, "max_seq_len": 512},
        {"batch_size": 8, "max_seq_len": 1024},
    ]
    atol = 1e-2
    rtol = 5e-2
    comparator = MultiOutputComparator(
        output_names=["output", "lse"],
        comparators={
            "output": HitRatioComparator(atol=1e-1, rtol=2e-1, min_hit_ratio=0.85),
            "lse": HitRatioComparator(atol=1e-1, rtol=2e-1, min_hit_ratio=0.85),
        },
    )

    @staticmethod
    def input_generator(**config):
        return generate_nsa_decode_inputs(
            batch_size=config["batch_size"], max_seq_len=config["max_seq_len"]
        )

    def baseline_fn(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """SGLang FlashMLA baseline implementation."""
        batch_size = q_nope.shape[0]
        device = q_nope.device
        head_dim = HEAD_DIM_CKV + HEAD_DIM_KPE

        # Combine q for FlashMLA
        q_all = torch.cat([q_nope, q_pe], dim=-1)

        # KV cache (combined)
        kv_cache = torch.cat([ckv_cache.squeeze(1), kpe_cache.squeeze(1)], dim=-1)
        kv_for_mla = kv_cache.unsqueeze(1)

        # Indices for MLA
        indices_for_mla = sparse_indices.unsqueeze(1)

        # Handle head padding for FlashMLA
        device_sm_major = torch.cuda.get_device_properties(device).major
        required_padding = 128 if device_sm_major >= 10 else 64

        need_padding = NUM_QO_HEADS % required_padding != 0
        if need_padding:
            q_padded = q_all.new_zeros((batch_size, required_padding, head_dim))
            q_padded[:, :NUM_QO_HEADS, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        fi_output_full, fi_max_logits, fi_lse_full = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_for_mla,
            indices=indices_for_mla,
            sm_scale=float(sm_scale),
            d_v=HEAD_DIM_CKV,
        )

        if need_padding:
            fi_output = fi_output_full[:, :NUM_QO_HEADS, :]
            fi_lse = fi_lse_full[:, :NUM_QO_HEADS]
        else:
            fi_output = fi_output_full
            fi_lse = fi_lse_full

        return {"output": fi_output, "lse": fi_lse}


@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang sgl_kernel not available")
class TestNSASparsePrefill(DefinitionTest):
    """Test NSA sparse prefill with SGLang as baseline."""

    definition_path = (
        "definitions/nsa_paged/nsa_sparse_prefill_causal_h16_ckv512_kpe64_topk256_ps1.json"
    )
    configs = [{"total_num_tokens": 32}, {"total_num_tokens": 64}, {"total_num_tokens": 128}]
    atol = 1e-2
    rtol = 5e-2
    comparator = MultiOutputComparator(
        output_names=["output", "lse"],
        comparators={
            "output": HitRatioComparator(atol=1e-1, rtol=2e-1, min_hit_ratio=0.85),
            "lse": HitRatioComparator(atol=1e-1, rtol=2e-1, min_hit_ratio=0.85),
        },
    )

    @staticmethod
    def input_generator(**config):
        return generate_nsa_prefill_inputs(total_num_tokens=config["total_num_tokens"])

    def baseline_fn(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """SGLang FlashMLA baseline implementation."""
        total_num_tokens = q_nope.shape[0]
        device = q_nope.device
        head_dim = HEAD_DIM_CKV + HEAD_DIM_KPE

        # Combine q for FlashMLA
        q_all = torch.cat([q_nope, q_pe], dim=-1)

        # KV cache (combined)
        kv_cache = torch.cat([ckv_cache.squeeze(1), kpe_cache.squeeze(1)], dim=-1)
        kv_for_mla = kv_cache.unsqueeze(1)

        # Indices for MLA
        indices_for_mla = sparse_indices.unsqueeze(1)

        # Handle head padding for FlashMLA
        device_sm_major = torch.cuda.get_device_properties(device).major
        required_padding = 128 if device_sm_major >= 10 else 64

        need_padding = NUM_QO_HEADS % required_padding != 0
        if need_padding:
            q_padded = q_all.new_zeros((total_num_tokens, required_padding, head_dim))
            q_padded[:, :NUM_QO_HEADS, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        fi_output_full, fi_max_logits, fi_lse_full = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_for_mla,
            indices=indices_for_mla,
            sm_scale=float(sm_scale),
            d_v=HEAD_DIM_CKV,
        )

        if need_padding:
            fi_output = fi_output_full[:, :NUM_QO_HEADS, :]
            fi_lse = fi_lse_full[:, :NUM_QO_HEADS]
        else:
            fi_output = fi_output_full
            fi_lse = fi_lse_full

        return {"output": fi_output, "lse": fi_lse}


if __name__ == "__main__":
    pytest.main(sys.argv)
