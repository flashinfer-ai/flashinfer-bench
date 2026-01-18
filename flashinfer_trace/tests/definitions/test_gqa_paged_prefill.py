"""Tests for GQA paged prefill definitions."""

import math
import sys

import flashinfer
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest


def generate_gqa_paged_prefill_inputs(
    batch_size: int,
    max_q_len: int = 32,
    max_kv_len: int = 64,
    num_qo_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 1,
    device: str = "cuda",
):
    """Generate random inputs for GQA paged prefill testing."""
    # Generate random query and KV lengths
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    # Create indptr arrays
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    # Get total tokens
    total_q = qo_indptr[-1].item()
    num_kv_indices = kv_indptr[-1].item()

    # Generate page indices
    max_pages = max_kv_len * batch_size * 2
    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]

    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        kv_indices[idx : idx + seq_len] = all_page_ids[idx : idx + seq_len]
        idx += seq_len

    # Generate KV cache
    k_cache = torch.randn(
        max_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        max_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Generate query tensor
    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Compute sm_scale
    sm_scale = 1.0 / math.sqrt(head_dim)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


class TestGQAPagedPrefillH32KV4(DefinitionTest):
    """Test GQA paged prefill with 32 QO heads and 4 KV heads."""

    definition_path = "definitions/gqa_paged/gqa_paged_prefill_causal_h32_kv4_d128_ps1.json"
    configs = [
        {"batch_size": 1, "max_q_len": 8, "max_kv_len": 16},
        {"batch_size": 4, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 8, "max_q_len": 32, "max_kv_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_paged_prefill_inputs(
            batch_size=config["batch_size"],
            max_q_len=config["max_q_len"],
            max_kv_len=config["max_kv_len"],
            num_qo_heads=32,
            num_kv_heads=4,
        )

    def baseline_fn(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device
        batch_size = qo_indptr.shape[0] - 1

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

        paged_kv_cache = torch.stack([k_cache, v_cache], dim=1)

        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD"
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=128,
            head_dim_vo=128,
            page_size=1,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q, paged_kv_cache, return_lse=True)


class TestGQAPagedPrefillH32KV8(DefinitionTest):
    """Test GQA paged prefill with 32 QO heads and 8 KV heads."""

    definition_path = "definitions/gqa_paged/gqa_paged_prefill_causal_h32_kv8_d128_ps1.json"
    configs = [
        {"batch_size": 1, "max_q_len": 8, "max_kv_len": 16},
        {"batch_size": 4, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 8, "max_q_len": 32, "max_kv_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_paged_prefill_inputs(
            batch_size=config["batch_size"],
            max_q_len=config["max_q_len"],
            max_kv_len=config["max_kv_len"],
            num_qo_heads=32,
            num_kv_heads=8,
        )

    def baseline_fn(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device
        batch_size = qo_indptr.shape[0] - 1

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

        paged_kv_cache = torch.stack([k_cache, v_cache], dim=1)

        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD"
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            head_dim_vo=128,
            page_size=1,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q, paged_kv_cache, return_lse=True)


if __name__ == "__main__":
    pytest.main(sys.argv)
