"""Reference test for gqa_paged_decode_h24_kv4_d128_ps1.

Validates the reference implementation against FlashInfer BatchDecodeWithPagedKVCacheWrapper.
Captured from Mixtral 8x22B at TP=2: 24 q-heads, 4 kv-heads, head_dim=128, page_size=1.
Group size = 24/4 = 6 (not a power of 2); KV heads are expanded to match q-heads.
"""

import math
from pathlib import Path

import pytest
import torch

NUM_QO_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 1
DEF_NAME = "gqa_paged_decode_h24_kv4_d128_ps1"


def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def load_definition():
    from flashinfer_bench.data import Definition, load_json_file

    definitions_dir = Path(__file__).parent.parent / "flashinfer_trace" / "definitions"
    for op_dir in definitions_dir.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{DEF_NAME}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {DEF_NAME} not found in {definitions_dir}")


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_pages = int(seq_lens.sum().item())

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    num_cache_pages = total_pages + 100
    k_cache = torch.randn(
        num_cache_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        num_cache_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size,max_seq_len", [(1, 16), (4, 64), (8, 128)])
def test_reference_correctness(batch_size, max_seq_len):
    """Validate reference implementation against FlashInfer."""
    import flashinfer

    device = "cuda"
    atol, rtol = 1e-2, 5e-2

    definition = load_definition()
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # group_size=6 is not a power of 2; expand KV heads to Q heads so
    # FlashInfer can use group_size=1 internally.
    gqa_ratio = NUM_QO_HEADS // NUM_KV_HEADS  # = 6
    k_cache_exp = inputs["k_cache"].repeat_interleave(gqa_ratio, dim=2)
    v_cache_exp = inputs["v_cache"].repeat_interleave(gqa_ratio, dim=2)
    fi_kv_heads = NUM_QO_HEADS  # expanded to 24

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=fi_kv_heads,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )
    fi_o, fi_lse = wrapper.run(inputs["q"], (k_cache_exp, v_cache_exp), return_lse=True)

    assert torch.allclose(
        ref_o.float(), fi_o.float(), atol=atol, rtol=rtol
    ), f"Output mismatch: max_abs={torch.abs(ref_o.float() - fi_o.float()).max():.4e}"
    assert torch.allclose(
        ref_lse, fi_lse, atol=atol, rtol=rtol
    ), f"LSE mismatch: max_abs={torch.abs(ref_lse - fi_lse).max():.4e}"


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v"] + sys.argv[1:])
