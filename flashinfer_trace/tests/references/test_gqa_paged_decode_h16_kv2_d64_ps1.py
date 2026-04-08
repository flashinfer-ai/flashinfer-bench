"""Reference test for gqa_paged_decode_h16_kv2_d64_ps1 (Qwen3 32B TP=4)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
PAGE_SIZE = 1


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


def generate_random_inputs(batch_size, max_seq_len, device="cuda"):
    """Generate random inputs for paged decode testing (page_size=1)."""
    # Each decode step has exactly 1 query token per sequence
    # Sequence lengths in KV cache (how many tokens each sequence has)
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)

    # For page_size=1: num_pages_per_seq = seq_len
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens.to(device), dim=0)

    total_pages = int(kv_indptr[-1].item())

    # Use consecutive page indices
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)

    # For page_size=1, last_page_len is always 1
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Query: one token per sequence in decode
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # KV cache: total_pages pages
    num_pages = total_pages + 10  # a few extra
    k_cache = torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "seq_lens": seq_lens,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_seq_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Paged Decode h16/kv2/d64/ps1 (Qwen3 32B TP=4): "
        f"batch={batch_size}, max_seq_len={max_seq_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("gqa_paged_decode_h16_kv2_d64_ps1")
    run = compile_reference(definition.reference)

    inputs = generate_random_inputs(batch_size, max_seq_len, device)

    print(f"Sequence lengths: {inputs['seq_lens'].numpy()}")
    print(f"Total KV pages: {inputs['kv_indices'].shape[0]}")

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

    # Setup FlashInfer
    # GQA group_size = 16 / 2 = 8, which is a power of 2 — natively supported by BatchDecode.
    print("\nSetting up FlashInfer BatchDecodeWithPagedKVCacheWrapper...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    decode_wrapper.plan(
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

    print("Running FlashInfer...")
    fi_output, fi_lse = decode_wrapper.run(
        inputs["q"],
        (inputs["k_cache"], inputs["v_cache"]),
        return_lse=True,
    )

    # Compare outputs
    print("\nComparing outputs...")
    ref_o_f32 = ref_o.float()
    fi_output_f32 = fi_output.float()

    abs_diff = torch.abs(ref_o_f32 - fi_output_f32)
    print(f"Output max abs diff: {abs_diff.max().item():.6e}")
    print(f"Output mean abs diff: {abs_diff.mean().item():.6e}")

    lse_abs_diff = torch.abs(ref_lse - fi_lse)
    print(f"LSE max abs diff: {lse_abs_diff.max().item():.6e}")
    print(f"LSE mean abs diff: {lse_abs_diff.mean().item():.6e}")

    output_close = torch.allclose(ref_o_f32, fi_output_f32, atol=atol, rtol=rtol)
    lse_close = torch.allclose(ref_lse, fi_lse, atol=atol, rtol=rtol)
    all_close = output_close and lse_close

    if all_close:
        print(f"\n✓ PASSED: Outputs and LSE match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED: output_close={output_close}, lse_close={lse_close}")
        if not output_close:
            flat_abs_diff = abs_diff.flatten()
            top_k = min(5, flat_abs_diff.numel())
            top_errors, top_indices = torch.topk(flat_abs_diff, top_k)
            print(f"\nTop {top_k} output error locations:")
            for i in range(top_k):
                idx = top_indices[i].item()
                b_idx = idx // (NUM_QO_HEADS * HEAD_DIM)
                h_idx = (idx % (NUM_QO_HEADS * HEAD_DIM)) // HEAD_DIM
                d_idx = idx % HEAD_DIM
                ref_val = ref_o_f32.flatten()[idx].item()
                fi_val = fi_output_f32.flatten()[idx].item()
                print(f"  [{b_idx}, {h_idx}, {d_idx}]: ref={ref_val:.6f}, fi={fi_val:.6f}, diff={top_errors[i].item():.6e}")

    return all_close


def main():
    """Run comprehensive tests."""
    print("Testing GQA Paged Decode h16/kv2/d64/ps1 (Qwen3 32B TP=4)")

    test_configs = [
        (1, 16),
        (4, 32),
        (8, 64),
        (16, 128),
    ]

    passed = 0
    total = len(test_configs)

    for batch_size, max_seq_len in test_configs:
        try:
            if test_correctness(batch_size, max_seq_len):
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
