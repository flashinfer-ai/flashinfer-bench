import flashinfer
import torch


@torch.no_grad()
def run(probs, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    # assert vocab_size == 129280

    probs = probs.to(torch.float32)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)

    for i in range(batch_size):
        row = probs[i]
        p = float(top_p[i].item())

        if p <= 0.0:
            # Degenerate to argmax
            out[i] = torch.argmax(row).to(torch.int64)
            continue

        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)

            # Shift mask to keep the first token that crosses p
            to_remove = cdf > p
            to_remove[1:] = to_remove[:-1].clone()
            to_remove[0] = False
            keep = ~to_remove
            keep_idx = idx[keep]

            # Build filtered distribution in original index space
            filtered = torch.zeros_like(row)
            filtered[keep_idx] = row[keep_idx]
            row = filtered / filtered.sum()

        out[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return out


def generate_random_inputs(batch_size, vocab_size=128256, distribution="normal", device="cuda"):
    """Generate random test inputs."""
    # Generate probabilities
    if distribution == "normal":
        logits = torch.randn(batch_size, vocab_size, device=device)
    elif distribution == "peaked":
        # Create peaked distribution
        logits = torch.randn(batch_size, vocab_size, device=device) * 0.1
        peak_indices = torch.randint(0, vocab_size, (batch_size,), device=device)
        for i in range(batch_size):
            logits[i, peak_indices[i]] += 5.0
    elif distribution == "uniform":
        logits = torch.zeros(batch_size, vocab_size, device=device)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1).to(torch.float32)

    # Generate varying top_p values
    top_p = torch.rand(batch_size, device=device) * 0.8 + 0.1  # Range [0.1, 0.9]

    return probs, top_p


def test_correctness(batch_size=1, vocab_size=128256, num_trials=10000):
    """Test correctness by comparing sampling frequency with expected renormalized probabilities.
    Uses the same approach as FlashInfer's test_top_p_sampling_freq."""
    print(f"\n{'=' * 60}")
    print("Testing correctness against expected probabilities")
    print(f"batch_size={batch_size}, vocab_size={vocab_size}, num_trials={num_trials}")
    print(f"{'=' * 60}")

    device = "cuda"
    torch.manual_seed(42)

    # Generate inputs
    probs, top_p = generate_random_inputs(batch_size, vocab_size, "peaked", device)

    # Count frequencies for both implementations
    ref_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)

    for trial in range(num_trials):
        progress_interval = max(1000, num_trials // 5)
        if trial % progress_interval == 0:
            print(f"  Trial {trial}/{num_trials}...")

        # Reference implementation
        torch.manual_seed(42 + trial)
        ref_samples = run(probs, top_p)
        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1

        # FlashInfer implementation
        torch.manual_seed(42 + trial)
        fi_samples = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)
        for i in range(batch_size):
            fi_counter[i, fi_samples[i]] += 1

    # Calculate frequencies
    ref_freq = ref_counter.float() / num_trials
    fi_freq = fi_counter.float() / num_trials

    # Calculate cosine similarity
    similarities = []
    for i in range(batch_size):
        # Only compare tokens that were sampled at least once
        mask = (ref_freq[i] > 0) | (fi_freq[i] > 0)
        if mask.sum() > 0:
            ref = ref_freq[i][mask]
            fi = fi_freq[i][mask]
            similarity = torch.nn.functional.cosine_similarity(ref.unsqueeze(0), fi.unsqueeze(0))
            similarities.append(similarity.item())
            print(f"  Sequence {i}: Cosine similarity = {similarity.item():.4f}")

    avg_similarity = sum(similarities) / len(similarities)
    print(f"\n  Average cosine similarity: {avg_similarity:.4f}")

    # Check similarity
    assert avg_similarity > 0.95, f"Implementations diverge too much: {avg_similarity:.4f} < 0.95"
    print("  Correctness test passed!")

    return True


def main():
    """Run comprehensive tests for top_p_sampling_from_probs."""
    print("Testing Top-P (Nucleus) Sampling from Probabilities")

    all_passed = True

    # Test correctness by comparing with FlashInfer
    try:
        # Test with different configurations (matching FlashInfer's approach)
        # Test different p values with batch_size=1 for efficiency
        test_configs = [
            # (batch_size, vocab_size, num_trials)
            (2, 128256, 10000),
            (4, 129280, 10000),
            (8, 151936, 10000),
        ]

        for batch_size, vocab_size, num_trials in test_configs:
            if not test_correctness(batch_size, vocab_size, num_trials):
                all_passed = False

    except Exception as e:
        print(f"Correctness test failed: {e}")
        all_passed = False

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    main()
