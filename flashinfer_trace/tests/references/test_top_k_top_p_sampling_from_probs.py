import flashinfer
import torch


@torch.no_grad()
def run(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    # assert vocab_size == 128256

    probs = probs.to(torch.float32)
    samples = torch.empty(batch_size, dtype=torch.int64, device=device)

    for i in range(batch_size):
        row = probs[i]
        k = int(top_k[i].item())
        p = float(top_p[i].item())

        # Apply top-k filtering
        if 0 < k < vocab_size:
            idx_sorted = torch.argsort(row, descending=True)
            keep_idx_k = idx_sorted[:k]
            filtered_k = torch.zeros_like(row)
            filtered_k[keep_idx_k] = row[keep_idx_k]
            row = filtered_k / filtered_k.sum()

        # Then apply top-p filtering
        if p <= 0.0:
            samples[i] = torch.argmax(row).to(torch.int64)
            continue

        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)

            to_remove = cdf > p
            if vocab_size > 1:
                to_remove[1:] = to_remove[:-1].clone()
                to_remove[0] = False

            keep_idx_p = idx[~to_remove]
            filtered_p = torch.zeros_like(row)
            filtered_p[keep_idx_p] = row[keep_idx_p]
            row = filtered_p / filtered_p.sum()

        # sample
        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return samples


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

    # Generate varying top_k and top_p values
    top_k = torch.randint(
        10, min(500, vocab_size // 2), (batch_size,), dtype=torch.int32, device=device
    )
    top_p = torch.rand(batch_size, device=device) * 0.8 + 0.1  # Range [0.1, 0.9]

    return probs, top_k, top_p


def test_correctness(batch_size=8, vocab_size=128256, num_trials=10000):
    """Test correctness by comparing with FlashInfer implementation."""
    print(f"\n{'=' * 60}")
    print("Testing correctness against FlashInfer")
    print(f"batch_size={batch_size}, num_trials={num_trials}")
    print(f"{'=' * 60}")

    device = "cuda"
    torch.manual_seed(42)

    # Generate inputs
    probs, top_k, top_p = generate_random_inputs(batch_size, vocab_size, "peaked", device)

    # Count frequencies for both implementations
    ref_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)

    for trial in range(num_trials):
        progress_interval = max(1000, num_trials // 5)
        if trial % progress_interval == 0:
            print(f"  Trial {trial}/{num_trials}...")

        # Reference implementation
        torch.manual_seed(42 + trial)
        ref_samples = run(probs, top_k, top_p)
        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1

        # FlashInfer implementation
        torch.manual_seed(42 + trial)
        fi_samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, filter_apply_order="top_k_first"
        )
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
    """Run comprehensive tests for top_k_top_p_sampling_from_probs."""
    print("Testing Combined Top-K Top-P Sampling from Probabilities")

    all_passed = True

    # Test correctness by comparing with FlashInfer
    try:
        # Test with different configurations
        test_configs = [(2, 128256, 10000), (4, 129280, 10000), (8, 151936, 10000)]

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
