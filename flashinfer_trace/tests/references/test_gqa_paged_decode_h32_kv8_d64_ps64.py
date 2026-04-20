"""Reference test for gqa_paged_decode_h32_kv8_d64_ps64 (Llama 3.2 1B)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
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
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    num_pages_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_pages_per_seq.sum().item()

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = (seq_lens - 1) % PAGE_SIZE + 1

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


def test_correctness(batch_size=4, max_seq_len=256, atol=1e-2, rtol=5e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return False

    definition = load_definition("gqa_paged_decode_h32_kv8_d64_ps64")
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        inputs["sm_scale"],
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )
    fi_o, fi_lse = wrapper.run(inputs["q"], (inputs["k_cache"], inputs["v_cache"]), return_lse=True)

    out_ok = torch.allclose(ref_o.float(), fi_o.float(), atol=atol, rtol=rtol)
    lse_ok = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    return out_ok and lse_ok


def main():
    configs = [(1, 16), (4, 256), (8, 512)]
    passed = sum(1 for b, s in configs if test_correctness(b, s))
    print(f"{passed}/{len(configs)} passed")


if __name__ == "__main__":
    main()
