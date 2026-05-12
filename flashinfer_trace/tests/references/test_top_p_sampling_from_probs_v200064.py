"""Reference test for top_p_sampling_from_probs_v200064 (MiniMax M2)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

# Paths
DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

VOCAB_SIZE = 200064


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


def generate_random_inputs(batch_size, distribution="peaked", device="cuda"):
    """Generate random test inputs."""
    if distribution == "peaked":
        logits = torch.randn(batch_size, VOCAB_SIZE, device=device) * 0.1
        peak_indices = torch.randint(0, VOCAB_SIZE, (batch_size,), device=device)
        for i in range(batch_size):
            logits[i, peak_indices[i]] += 5.0
    else:
        logits = torch.randn(batch_size, VOCAB_SIZE, device=device)

    probs = torch.softmax(logits, dim=-1).to(torch.float32)
    top_p = torch.rand(batch_size, device=device) * 0.8 + 0.1  # Range [0.1, 0.9]

    return probs, top_p


def test_correctness(batch_size=4, num_trials=5000):
    """Test correctness by comparing sampling distributions with FlashInfer."""
    print(f"\n{'='*60}")
    print(
        f"Testing Top-P Sampling v200064 (MiniMax M2): batch_size={batch_size}, num_trials={num_trials}"
    )
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("top_p_sampling_from_probs_v200064")
    run = compile_reference(definition.reference)

    torch.manual_seed(42)
    probs, top_p = generate_random_inputs(batch_size, "peaked", device)

    # Count frequencies for both implementations
    ref_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)

    print(f"Running {num_trials} trials to compare distributions...")
    for trial in range(num_trials):
        ref_samples = run(probs.clone(), top_p)
        fi_samples = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)

        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1
            fi_counter[i, fi_samples[i]] += 1

    # Compare frequency distributions
    ref_freq = ref_counter.float() / num_trials
    fi_freq = fi_counter.float() / num_trials

    nonzero_mask = probs > 1e-6
    ref_nonzero = ref_freq[nonzero_mask]
    fi_nonzero = fi_freq[nonzero_mask]

    freq_diff = torch.abs(ref_nonzero - fi_nonzero).max().item()
    print(f"Max frequency difference on non-zero tokens: {freq_diff:.4f}")

    passed = freq_diff < 0.05
    if passed:
        print(f"\n✓ PASSED: Sampling distributions match (max_freq_diff={freq_diff:.4f} < 0.05)")
    else:
        print(f"\n✗ FAILED: Sampling distributions differ (max_freq_diff={freq_diff:.4f} >= 0.05)")

    return passed


def main():
    """Run comprehensive tests."""
    print("Testing Top-P Sampling v200064 (MiniMax M2)")

    test_configs = [(1, 5000), (4, 5000), (8, 3000)]
    passed = 0
    for batch_size, num_trials in test_configs:
        try:
            if test_correctness(batch_size, num_trials):
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
