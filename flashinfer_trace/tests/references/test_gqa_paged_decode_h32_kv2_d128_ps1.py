"""Reference test for gqa_paged_decode_h32_kv2_d128_ps1."""

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


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device)
    total_pages = seq_lens.sum().item()

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
        "seq_lens": seq_lens,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Decode h32/kv2 ps1 (Llama 3.1 405B TP=4): batch_size={batch_size}, max_seq_len={max_seq_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("gqa_paged_decode_h32_kv2_d128_ps1")
    run = compile_reference(definition.reference)
    inputs = generate_random_inputs(batch_size, max_seq_len, device)
    print(f"Sequence lengths: {inputs['seq_lens'].cpu().numpy()}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["sm_scale"],
    )

    # FlashInfer only supports power-of-2 group sizes up to 8. Since group_size =
    # 32/2 = 16 is unsupported, expand KV heads from 2 to 32 (group_size=1, MHA),
    # which gives mathematically equivalent results.
    group_size = NUM_QO_HEADS // NUM_KV_HEADS  # 16
    k_cache_expanded = inputs["k_cache"].repeat_interleave(group_size, dim=2)
    v_cache_expanded = inputs["v_cache"].repeat_interleave(group_size, dim=2)

    print("\nSetting up FlashInfer...")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_QO_HEADS,  # expanded to match q heads (group_size=1)
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"].item(),
    )

    print("Running FlashInfer...")
    fi_o, fi_lse = wrapper.run(
        inputs["q"], (k_cache_expanded, v_cache_expanded), return_lse=True
    )

    # Compare
    print("\nComparing outputs...")
    abs_diff = torch.abs(ref_o.float() - fi_o.float())
    print(f"Output max abs diff: {abs_diff.max().item():.6e}")
    print(f"Output mean abs diff: {abs_diff.mean().item():.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    print(f"LSE max abs diff: {lse_abs_diff.max().item():.6e}")

    out_ok = torch.allclose(ref_o.float(), fi_o.float(), atol=atol, rtol=rtol)
    lse_ok = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = out_ok and lse_ok

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: out_ok={out_ok}, lse_ok={lse_ok}")

    return all_close


def main():
    configs = [(1, 16), (4, 64), (8, 128)]
    passed = sum(1 for b, s in configs if test_correctness(b, s))
    print(f"{passed}/{len(configs)} passed")


if __name__ == "__main__":
    main()
