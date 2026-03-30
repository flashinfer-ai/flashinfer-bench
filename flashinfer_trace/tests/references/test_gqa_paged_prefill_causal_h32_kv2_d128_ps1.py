"""Reference test for gqa_paged_prefill_causal_h32_kv2_d128_ps1."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 1


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


def generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device="cuda"):
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    num_kv_indices = int(kv_indptr[-1].item())

    all_page_ids = torch.randperm(max_pages, device=device)[:num_kv_indices]
    kv_indices = torch.zeros(num_kv_indices, dtype=torch.int32, device=device)
    idx = 0
    for i in range(batch_size):
        seq_len = int(kv_lens[i].item())
        kv_indices[idx : idx + seq_len] = all_page_ids[idx : idx + seq_len]
        idx += seq_len

    last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    k_cache = torch.randn(
        max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn(
        max_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "last_page_len": last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=2, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return False

    definition = load_definition("gqa_paged_prefill_causal_h32_kv2_d128_ps1")
    run = compile_reference(definition.reference)

    max_pages = max_kv_len * batch_size * 2
    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device)

    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    paged_kv_cache = torch.stack([inputs["k_cache"], inputs["v_cache"]], dim=1)

    wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["last_page_len"],
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
    fi_o, fi_lse = wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    out_ok = torch.allclose(ref_o.float(), fi_o.float(), atol=atol, rtol=rtol)
    lse_ok = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    return out_ok and lse_ok


def main():
    configs = [(1, 16, 32), (2, 32, 64)]
    passed = sum(1 for b, q, kv in configs if test_correctness(b, q, kv))
    print(f"{passed}/{len(configs)} passed")


if __name__ == "__main__":
    main()
