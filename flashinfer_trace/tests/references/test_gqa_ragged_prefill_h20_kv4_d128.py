"""Reference test for gqa_ragged_prefill_causal_h20_kv4_d128 (Qwen3 14B TP=2)."""

import math
from pathlib import Path

import flashinfer
import torch
from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

NUM_QO_HEADS = 20
NUM_KV_HEADS = 4
HEAD_DIM = 128


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


def generate_random_inputs(batch_size, max_q_len, max_kv_len, device="cuda"):
    """Generate random inputs for ragged prefill testing."""
    q_lens = torch.randint(1, max_q_len + 1, (batch_size,), dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32)
    for i in range(batch_size):
        kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_len + 1, (1,)).item()

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(q_lens.to(device), dim=0)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)

    total_q = int(qo_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    sm_scale = torch.tensor(1.0 / math.sqrt(HEAD_DIM), dtype=torch.float32, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "total_q": total_q,
        "total_kv": total_kv,
        "sm_scale": sm_scale,
    }


def test_correctness(batch_size=4, max_q_len=32, max_kv_len=64, atol=1e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing GQA Ragged Prefill h20/kv4 (Qwen3 14B TP=2): batch={batch_size}, max_q={max_q_len}, max_kv={max_kv_len}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("gqa_ragged_prefill_causal_h20_kv4_d128")
    run = compile_reference(definition.reference)

    inputs = generate_random_inputs(batch_size, max_q_len, max_kv_len, device)

    print(f"Query lengths: {inputs['q_lens'].numpy()}")
    print(f"KV lengths: {inputs['kv_lens'].numpy()}")
    print(f"Total query tokens: {inputs['total_q']}")
    print(f"Total KV tokens: {inputs['total_kv']}")

    # Run reference
    print("\nRunning reference implementation...")
    ref_o, ref_lse = run(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["sm_scale"],
    )

    # Setup FlashInfer
    print("\nSetting up FlashInfer...")
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    prefill_wrapper.plan(
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=True,
        sm_scale=inputs["sm_scale"],
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    print("Running FlashInfer...")
    fi_output, fi_lse = prefill_wrapper.run(inputs["q"], inputs["k"], inputs["v"], return_lse=True)

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
    print("Testing GQA Ragged Prefill h20/kv4 (Qwen3 14B TP=2)")

    test_configs = [(1, 16, 32), (4, 32, 64), (8, 64, 128)]
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
