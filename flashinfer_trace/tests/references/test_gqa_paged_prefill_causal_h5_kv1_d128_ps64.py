"""Reference test for gqa_paged_prefill_causal_h5_kv1_d128_ps64."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 5
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 64


def load_definition(name: str) -> Definition:
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found")


def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    total_q_per_seq = torch.randint(
        1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    total_q = total_q_per_seq.sum().item()
    num_pages_per_seq = (total_q_per_seq + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_pages_per_seq.sum().item()
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = (total_q_per_seq - 1) % PAGE_SIZE + 1

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(total_q_per_seq, dim=0)

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
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
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=2, max_seq_len=256, atol=1e-2, rtol=5e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return False

    definition = load_definition("gqa_paged_prefill_causal_h5_kv1_d128_ps64")
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        inputs["sm_scale"],
    )

    # group_size=5 is not power of 2; use num_kv_heads=1 directly (FlashInfer supports GQA natively for prefill)
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )
    fi_o, fi_lse = wrapper.run(inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True)

    out_ok = torch.allclose(ref_o.float(), fi_o.float(), atol=atol, rtol=rtol)
    lse_ok = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    return out_ok and lse_ok


def main():
    configs = [(1, 16), (2, 256)]
    passed = sum(1 for b, s in configs if test_correctness(b, s))
    print(f"\nSummary: {passed}/{len(configs)} tests passed")


if __name__ == "__main__":
    main()
