"""Reference test for gqa_paged_prefill_causal_h24_kv4_d128_ps64 (Mixtral 8x22B TP=2)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 64


def load_definition(name: str) -> Definition:
    """Load a definition by name from definitions directory."""
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    """Compile reference implementation to callable function."""
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device="cuda"):
    """Generate random inputs for paged prefill testing."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_pages_per_seq = (kv_lens + PAGE_SIZE - 1) // PAGE_SIZE
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_pages_per_seq.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    num_kv_pages = int(kv_indptr[-1].item())

    kv_indices = torch.arange(num_kv_pages, dtype=torch.int32, device=device)
    kv_last_page_len = ((kv_lens - 1) % PAGE_SIZE + 1).to(device)

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
        "kv_last_page_len": kv_last_page_len,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=128, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Prefill h24/kv4 ps64 (Mixtral 8x22B TP=2): batch={batch_size}, max_q={max_q_len}, max_kv={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("gqa_paged_prefill_causal_h24_kv4_d128_ps64")
    run = compile_reference(definition.reference)

    max_pages = (max_kv_len * batch_size * 2 + PAGE_SIZE - 1) // PAGE_SIZE + 10
    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, max_pages, device)

    print(f"Query lengths: {inputs['q_lens'].numpy()}")
    print(f"KV lengths: {inputs['kv_lens'].numpy()}")

    # Run reference
    print("\nRunning reference implementation...")
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

    # Setup FlashInfer
    # FlashInfer only supports power-of-2 group sizes. Since group_size = 24/4 = 6
    # is not a power of 2, expand KV heads from 4 to 24 (repeating each KV head
    # 6 times) so group_size=1 (MHA), which gives mathematically equivalent results.
    group_size = NUM_QO_HEADS // NUM_KV_HEADS  # 6
    k_cache_expanded = inputs["k_cache"].repeat_interleave(group_size, dim=2)
    v_cache_expanded = inputs["v_cache"].repeat_interleave(group_size, dim=2)

    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )
    paged_kv_cache = torch.stack([k_cache_expanded, v_cache_expanded], dim=1)

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        paged_kv_indptr=inputs["kv_indptr"],
        paged_kv_indices=inputs["kv_indices"],
        paged_kv_last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_QO_HEADS,  # expanded to match q heads (group_size=1)
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=True,
        sm_scale=inputs["sm_scale"].item(),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], paged_kv_cache, return_lse=True)

    # Compare
    print("\nComparing outputs...")
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    print(f"Output max abs diff: {abs_diff.max().item():.6e}")
    print(f"Output mean abs diff: {abs_diff.mean().item():.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    print(f"LSE max abs diff: {lse_abs_diff.max().item():.6e}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: output_close={output_close}, lse_close={lse_close}")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing GQA Paged Prefill h24/kv4/ps64 (Mixtral 8x22B TP=2)")

    test_configs = [(1, 16, 64), (4, 32, 128), (8, 64, 256)]
    passed = 0
    for batch_size, max_q_len, max_kv_len in test_configs:
        try:
            if test_correctness(batch_size, max_q_len, max_kv_len):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(test_configs)} tests passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
