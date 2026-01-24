"""Tests for GQA ragged prefill definitions."""

import math
import sys

import flashinfer
import pytest
import torch

from flashinfer_bench.testing import DefinitionTest


def generate_gqa_ragged_prefill_inputs(
    batch_size: int,
    max_q_len: int = 32,
    max_kv_len: int = 64,
    num_qo_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    device: str = "cuda",
):
    """Generate random inputs for GQA ragged prefill testing."""
    # Generate random query lengths
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)

    # Generate KV lengths >= query lengths for causal attention
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
    total_kv = kv_indptr[-1].item()

    # Generate tensors
    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Compute sm_scale
    sm_scale = 1.0 / math.sqrt(head_dim)

    return {
        "q": q,
        "k": k,
        "v": v,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "sm_scale": sm_scale,
    }


class TestGQARaggedPrefillH32KV4(DefinitionTest):
    """Test GQA ragged prefill with 32 QO heads and 4 KV heads."""

    definition_path = "definitions/gqa_ragged/gqa_ragged_prefill_causal_h32_kv4_d128.json"
    configs = [
        {"batch_size": 1, "max_q_len": 8, "max_kv_len": 16},
        {"batch_size": 4, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 8, "max_q_len": 32, "max_kv_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_ragged_prefill_inputs(
            batch_size=config["batch_size"],
            max_q_len=config["max_q_len"],
            max_kv_len=config["max_kv_len"],
            num_qo_heads=32,
            num_kv_heads=4,
        )

    def baseline_fn(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD"
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=128,
            head_dim_vo=128,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q, k, v, return_lse=True)


class TestGQARaggedPrefillH32KV8(DefinitionTest):
    """Test GQA ragged prefill with 32 QO heads and 8 KV heads."""

    definition_path = "definitions/gqa_ragged/gqa_ragged_prefill_causal_h32_kv8_d128.json"
    configs = [
        {"batch_size": 1, "max_q_len": 8, "max_kv_len": 16},
        {"batch_size": 4, "max_q_len": 16, "max_kv_len": 32},
        {"batch_size": 8, "max_q_len": 32, "max_kv_len": 64},
    ]
    atol = 1e-2
    rtol = 5e-2

    @staticmethod
    def input_generator(**config):
        return generate_gqa_ragged_prefill_inputs(
            batch_size=config["batch_size"],
            max_q_len=config["max_q_len"],
            max_kv_len=config["max_kv_len"],
            num_qo_heads=32,
            num_kv_heads=8,
        )

    def baseline_fn(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """FlashInfer baseline implementation."""
        device = q.device

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD"
        )
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            head_dim_vo=128,
            causal=True,
            sm_scale=float(sm_scale),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        return wrapper.run(q, k, v, return_lse=True)


if __name__ == "__main__":
    pytest.main(sys.argv)
