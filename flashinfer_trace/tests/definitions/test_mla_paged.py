"""Tests for MLA paged attention definitions."""

import sys

import flashinfer
import numpy as np
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest


def generate_mla_decode_inputs(
    batch_size: int,
    max_seq_len: int = 64,
    num_qo_heads: int = 16,
    head_dim_ckv: int = 512,
    head_dim_kpe: int = 64,
    page_size: int = 1,
    device: str = "cuda",
):
    """Generate random inputs for MLA paged decode testing."""
    # Generate random sequence lengths
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)

    total_pages_needed = seq_lens.sum().item()

    # Generate kv_indptr and kv_indices
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Generate query tensors
    q_nope = torch.randn(
        batch_size, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(batch_size, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate KV caches
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # MLA scale
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


def generate_mla_prefill_inputs(
    batch_size: int,
    max_q_len: int = 32,
    max_kv_len: int = 64,
    num_qo_heads: int = 16,
    head_dim_ckv: int = 512,
    head_dim_kpe: int = 64,
    page_size: int = 1,
    device: str = "cuda",
):
    """Generate random inputs for MLA paged prefill testing."""
    # Generate random sequence lengths
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32, device=device)
    kv_lens = torch.randint(1, max_kv_len + 1, (batch_size,), dtype=torch.int32, device=device)

    # Ensure kv_len >= q_len for causal attention
    for i in range(batch_size):
        kv_lens[i] = max(kv_lens[i], q_lens[i])

    total_q = q_lens.sum().item()
    total_pages_needed = kv_lens.sum().item()

    # Generate indptrs
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens, dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens, dim=0)

    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # Generate query tensors
    q_nope = torch.randn(total_q, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(total_q, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # Generate KV caches
    num_pages = total_pages_needed + 100
    ckv_cache = torch.randn(num_pages, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(num_pages, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)

    # MLA scale
    sm_scale = 1.0 / np.sqrt(128 + head_dim_kpe)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sm_scale": sm_scale,
    }


class TestMLAPagedDecode(DefinitionTest):
    """Test MLA paged decode with 16 heads, ckv=512, kpe=64."""

    definition_path = "definitions/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1.json"
    configs = [
        {"batch_size": 1, "max_seq_len": 16},
        {"batch_size": 4, "max_seq_len": 32},
        {"batch_size": 8, "max_seq_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_mla_decode_inputs(
            batch_size=config["batch_size"], max_seq_len=config["max_seq_len"]
        )

    def baseline_fn(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """FlashInfer baseline implementation."""
        device = q_nope.device
        batch_size = q_nope.shape[0]

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]

        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=False,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=True)


class TestMLAPagedPrefill(DefinitionTest):
    """Test MLA paged prefill with 16 heads, ckv=512, kpe=64."""

    definition_path = "definitions/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.json"
    configs = [
        {"batch_size": 1, "max_q_len": 8, "max_kv_len": 16},
        {"batch_size": 4, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 8, "max_q_len": 32, "max_kv_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_mla_prefill_inputs(
            batch_size=config["batch_size"],
            max_q_len=config["max_q_len"],
            max_kv_len=config["max_kv_len"],
        )

    def baseline_fn(
        self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
    ):
        """FlashInfer baseline implementation."""
        device = q_nope.device
        kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="auto")
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=True)


if __name__ == "__main__":
    pytest.main(sys.argv)
