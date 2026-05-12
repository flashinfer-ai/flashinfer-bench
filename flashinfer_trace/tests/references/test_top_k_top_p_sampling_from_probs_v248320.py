import flashinfer
import torch


@torch.no_grad()
def run(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    assert vocab_size == 248320

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
            # FlashInfer official: ascending sort, keep tokens where cdf >= (1-p)
            sorted_prob, indices = torch.sort(row, descending=False)
            cdf = torch.cumsum(sorted_prob, dim=0)
            mask = (cdf >= (1.0 - p))
            filtered_p = torch.zeros_like(row)
            filtered_p[indices[mask]] = row[indices[mask]]
            row = filtered_p / filtered_p.sum()

        # sample
        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return samples


def generate_random_inputs(batch_size, vocab_size=248320, distribution="peaked", device="cuda"):
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
    top_k = torch.randint(
        10, min(500, vocab_size // 2), (batch_size,), dtype=torch.int32, device=device
    )
    top_p = torch.rand(batch_size, dtype=torch.float32, device=device) * 0.5 + 0.5  # [0.5, 1.0)
    return probs, top_k, top_p


def test_correctness(batch_size=4, vocab_size=248320, num_trials=10000):
    print(f"\n{'='*60}")
    print(f"Testing top_k_top_p_sampling_from_probs v248320")
    print(f"batch_size={batch_size}, num_trials={num_trials}")
    print(f"{'='*60}")

    device = "cuda"
    torch.manual_seed(42)

    probs, top_k, top_p = generate_random_inputs(batch_size, vocab_size, "peaked", device)

    ref_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device=device)

    for trial in range(num_trials):
        progress_interval = max(1000, num_trials // 5)
        if trial % progress_interval == 0:
            print(f"  Trial {trial}/{num_trials}...")

        torch.manual_seed(42 + trial)
        ref_samples = run(probs, top_k, top_p)
        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1

        torch.manual_seed(42 + trial)
        fi_samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
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


def main():
    print("Testing Top-K Top-P Sampling from Probabilities (vocab_size=248320)")

    passed = 0
    test_configs = [(2, 248320, 10000), (4, 248320, 10000)]
    total = len(test_configs)

    for batch_size, vocab_size, num_trials in test_configs:
        try:
            test_correctness(batch_size, vocab_size, num_trials)
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")

    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} tests failed")


if __name__ == "__main__":
    main()
