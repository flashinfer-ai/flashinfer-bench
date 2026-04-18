"""Reference test for top_p_sampling_from_probs_v163840 (Kimi K2.5)."""

import math
from pathlib import Path

import flashinfer
import torch

from flashinfer_bench.data import Definition, load_json_file

DEFINITIONS_DIR = Path(__file__).parent.parent.parent / "definitions"

VOCAB_SIZE = 163840


def load_definition(name: str) -> Definition:
    for op_dir in DEFINITIONS_DIR.iterdir():
        if op_dir.is_dir():
            def_file = op_dir / f"{name}.json"
            if def_file.exists():
                return load_json_file(Definition, def_file)
    raise FileNotFoundError(f"Definition {name} not found in {DEFINITIONS_DIR}")


def compile_reference(reference_code: str):
    namespace = {"torch": torch, "math": math}
    exec(reference_code, namespace)
    return namespace["run"]


def generate_random_inputs(batch_size, distribution="peaked", device="cuda"):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, skipping test")
        return False

    definition = load_definition("top_p_sampling_from_probs_v163840")
    run = compile_reference(definition.reference)

    torch.manual_seed(42)
    probs, top_p = generate_random_inputs(batch_size, "peaked", device)

    ref_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)
    fi_counter = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.int32, device=device)

    for _ in range(num_trials):
        ref_samples = run(probs.clone(), top_p)
        fi_samples = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)

        for i in range(batch_size):
            ref_counter[i, ref_samples[i]] += 1
            fi_counter[i, fi_samples[i]] += 1

    ref_freq = ref_counter.float() / num_trials
    fi_freq = fi_counter.float() / num_trials

    nonzero_mask = probs > 1e-6
    freq_diff = torch.abs(ref_freq[nonzero_mask] - fi_freq[nonzero_mask]).max().item()

    passed = freq_diff < 0.05
    print(
        f"batch_size={batch_size}: max_freq_diff={freq_diff:.4f} "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    return passed


def main():
    test_configs = [(1, 5000), (4, 5000), (8, 3000)]
    passed = sum(1 for b, t in test_configs if test_correctness(b, t))
    print(f"\nSummary: {passed}/{len(test_configs)} tests passed")


if __name__ == "__main__":
    main()
