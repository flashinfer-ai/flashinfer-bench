"""Reference test for top_k_top_p_sampling_from_probs_v262208 (Gemma 3 27B)."""
import flashinfer
import torch


VOCAB_SIZE = 262208


@torch.no_grad()
def run(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    assert vocab_size == VOCAB_SIZE

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

        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return samples


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
    top_k = torch.randint(10, min(500, VOCAB_SIZE // 2), (batch_size,), dtype=torch.int32, device=device)
    top_p = torch.rand(batch_size, device=device) * 0.8 + 0.1  # Range [0.1, 0.9]

    return probs, top_k, top_p


def test_correctness(batch_size=4, num_trials=5000):
    """Test correctness by comparing sampling distributions with FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing Top-K+Top-P Sampling v262208 (Gemma 3 27B): batch_size={batch_size}, num_trials={num_trials}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    torch.manual_seed(42)
    probs, top_k, top_p = generate_random_inputs(batch_size, "peaked", device)

    # Count frequencies for both implementations
    ref_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)

    print(f"Running {num_trials} trials to compare distributions...")
    for trial in range(num_trials):
        ref_samples = run(probs.clone(), top_k, top_p)
        fi_samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)

        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1
            fi_counter[i, fi_samples[i]] += 1

    # Compare frequency distributions
    ref_freq = ref_counter.float() / num_trials
    fi_freq = fi_counter.float() / num_trials

    nonzero_mask = (probs > 1e-6)
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
    print("Testing Top-K+Top-P Sampling v262208 (Gemma 3 27B)")

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
