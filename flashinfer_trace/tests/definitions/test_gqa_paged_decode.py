"""Tests for GQA paged decode definitions."""

import math
import sys

import flashinfer
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest


def generate_gqa_decode_inputs(
    batch_size: int,
    max_seq_len: int = 64,
    num_qo_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 1,
    device: str = "cuda",
):
    """Generate random inputs for GQA paged decode testing.

    Handles the complex dependencies between kv_indptr and kv_indices.
    """
    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Calculate total pages needed (page_size=1 means num_pages = total_tokens)
    total_pages_needed = seq_lens.sum().item()

    # Generate kv_indptr based on sequence lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

    # Generate kv_indices (consecutive page indices)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Generate query tensor
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Generate K and V caches with extra pages
    num_pages = total_pages_needed + 100
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Compute sm_scale
    sm_scale = 1.0 / math.sqrt(head_dim)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


class TestGQAPagedDecodeH32KV4(DefinitionTest):
    """Test GQA paged decode with 32 QO heads and 4 KV heads."""

    definition_path = "definitions/gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1.json"
    configs = [
        {"batch_size": 1, "max_seq_len": 16},
        {"batch_size": 4, "max_seq_len": 32},
        {"batch_size": 8, "max_seq_len": 64},
        {"batch_size": 16, "max_seq_len": 128},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_decode_inputs(
            batch_size=config["batch_size"],
            max_seq_len=config["max_seq_len"],
            num_qo_heads=32,
            num_kv_heads=4,
        )

    def baseline_fn(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device
        batch_size = q.shape[0]

        # Create workspace buffer
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # For page_size=1, last_page_len is always 1
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim=128,
            page_size=1,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=float(sm_scale),
        )

        return wrapper.run(q, (k_cache, v_cache), return_lse=True)


class TestGQAPagedDecodeH32KV8(DefinitionTest):
    """Test GQA paged decode with 32 QO heads and 8 KV heads."""

    # Relative path to FIB_DATASET_PATH
    definition_path = "definitions/gqa_paged/gqa_paged_decode_h32_kv8_d128_ps1.json"
    configs = [
        {"batch_size": 1, "max_seq_len": 16},
        {"batch_size": 4, "max_seq_len": 32},
        {"batch_size": 8, "max_seq_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_decode_inputs(
            batch_size=config["batch_size"],
            max_seq_len=config["max_seq_len"],
            num_qo_heads=32,
            num_kv_heads=8,
        )

    def baseline_fn(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device
        batch_size = q.shape[0]

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

        wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=kv_last_page_len,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=1,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=float(sm_scale),
        )

        return wrapper.run(q, (k_cache, v_cache), return_lse=True)


if __name__ == "__main__":
    pytest.main(sys.argv)
