"""
Test top_p_sampling_from_probs reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation in terms of distribution.
"""

import flashinfer
import torch
from test_utils import get_reference_run

# Load reference implementation from definition (use v128256 as default)
run = get_reference_run("top_p_sampling_from_probs_v128256")


def generate_random_inputs(batch_size, vocab_size=128256, distribution="normal", device="cuda"):
    """Generate random test inputs."""
    if distribution == "normal":
        logits = torch.randn(batch_size, vocab_size, device=device)
    elif distribution == "peaked":
        logits = torch.randn(batch_size, vocab_size, device=device) * 0.1
        peak_indices = torch.randint(0, vocab_size, (batch_size,), device=device)
        for i in range(batch_size):
            logits[i, peak_indices[i]] += 5.0
    elif distribution == "uniform":
        logits = torch.zeros(batch_size, vocab_size, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    probs = torch.softmax(logits, dim=-1).to(torch.float32)
    top_p = torch.rand(batch_size, dtype=torch.float32, device=device) * 0.5 + 0.5  # 0.5-1.0

    return probs, top_p


def test_correctness(batch_size=8, vocab_size=128256, num_trials=10000):
    """Test correctness by comparing with FlashInfer implementation."""
    print(f"\n{'=' * 60}")
    print("Testing correctness against FlashInfer")
    print(f"batch_size={batch_size}, num_trials={num_trials}")
    print(f"{'=' * 60}")

    device = "cuda"
    torch.manual_seed(42)

    probs, top_p = generate_random_inputs(batch_size, vocab_size, "peaked", device)

    ref_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)

    for trial in range(num_trials):
        progress_interval = max(1000, num_trials // 5)
        if trial % progress_interval == 0:
            print(f"  Trial {trial}/{num_trials}...")

        torch.manual_seed(42 + trial)
        ref_samples = run(probs, top_p)
        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1

        torch.manual_seed(42 + trial)
        fi_samples = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)
        for i in range(batch_size):
            fi_counter[i, fi_samples[i]] += 1

    ref_freq = ref_counter.float() / num_trials
    fi_freq = fi_counter.float() / num_trials

    similarities = []
    for i in range(batch_size):
        mask = (ref_freq[i] > 0) | (fi_freq[i] > 0)
        if mask.sum() > 0:
            ref = ref_freq[i][mask]
            fi = fi_freq[i][mask]
            similarity = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), fi.unsqueeze(0))
            similarities.append(similarity.item())
            print(f"  Sequence {i}: Cosine similarity = {similarity.item():.4f}")

    avg_similarity = sum(similarities) / len(similarities)
    print(f"\n  Average cosine similarity: {avg_similarity:.4f}")

    assert avg_similarity > 0.95, f"Implementations diverge too much: {avg_similarity:.4f} < 0.95"
    print("  Correctness test passed!")

    return True


def main():
    """Run comprehensive tests for top_p_sampling_from_probs."""
    print("Testing Top-P Sampling from Probabilities (from definition)")

    all_passed = True

    try:
        test_configs = [(2, 128256, 10000), (4, 129280, 10000), (8, 151936, 10000)]

        for batch_size, vocab_size, num_trials in test_configs:
            if not test_correctness(batch_size, vocab_size, num_trials):
                all_passed = False

    except Exception as e:
        print(f"Correctness test failed: {e}")
        all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    main()
