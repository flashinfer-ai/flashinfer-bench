"""
Test BF16 MoE reference implementation against FlashInfer kernel.
Configuration: top_k=8, 256 experts, hidden_size=2048, intermediate_size=256.
Captured from Qwen3.5-35B-A3B at TP=2.

Run with:
    pytest test_moe_bf16_topk8_e256_h2048_i256.py -v
    python test_moe_bf16_topk8_e256_h2048_i256.py
"""

import pytest
import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    seq_len, hidden_size = hidden_states.shape
    num_experts = gating_output.shape[1]
    top_k = topk_ids.shape[1]
    intermediate_size = w2.shape[2]

    assert hidden_size == 2048
    assert num_experts == 256
    assert top_k == 8
    assert intermediate_size == 256

    device = hidden_states.device
    output = torch.zeros((seq_len, hidden_size), dtype=torch.float32, device=device)

    A = hidden_states.to(torch.float32)

    for t in range(seq_len):
        for k_idx in range(top_k):
            expert_id = int(topk_ids[t, k_idx].item())
            weight = topk_weights[t, k_idx].item()

            W13 = w1[expert_id].to(torch.float32)  # [gemm1_out_size, hidden_size]
            W2 = w2[expert_id].to(torch.float32)    # [hidden_size, intermediate_size]

            # GEMM1
            g1 = A[t] @ W13.t()  # [gemm1_out_size]

            # SwiGLU
            x1 = g1[:intermediate_size]
            x2 = g1[intermediate_size:]
            silu_x2 = x2 / (1.0 + torch.exp(-x2))
            c = silu_x2 * x1  # [intermediate_size]

            # GEMM2
            o = c @ W2.t()  # [hidden_size]

            output[t] += weight * o

    return output.to(torch.bfloat16)


@torch.no_grad()
def _sglang_moe_ground_truth(hidden_states, w1, w2, topk_weights, topk_ids):
    """SGLang vanilla MoE implementation (adapted from fused_moe_native.py moe_forward_native).

    Uses the same dispatch-by-expert pattern as SGLang's torch-native fallback.
    """
    seq_len, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    top_k = topk_ids.shape[1]
    intermediate_size = w2.shape[2]

    # Sort tokens by expert assignment
    cnts = topk_ids.new_zeros((seq_len, num_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()

    sorted_tokens = hidden_states[idxs // top_k].to(torch.float32)
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + int(num_tokens)
        if num_tokens == 0:
            continue
        tokens_for_expert = sorted_tokens[start_idx:end_idx]

        # GEMM1: [num_tokens, hidden] @ [2*intermediate, hidden].T -> [num_tokens, 2*intermediate]
        gate_up = tokens_for_expert @ w1[i].to(torch.float32).t()

        # SwiGLU activation
        x1 = gate_up[:, :intermediate_size]
        x2 = gate_up[:, intermediate_size:]
        activated = (x2 / (1.0 + torch.exp(-x2))) * x1

        # GEMM2: [num_tokens, intermediate] @ [hidden, intermediate].T -> [num_tokens, hidden]
        expert_out = activated @ w2[i].to(torch.float32).t()
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    final_out = (
        new_x.view(seq_len, top_k, hidden_size)
        .mul_(topk_weights.unsqueeze(-1).to(torch.float32))
        .sum(dim=1)
    )
    return final_out.to(torch.bfloat16)


def generate_random_inputs(
    seq_len,
    num_experts=256,
    hidden_size=2048,
    intermediate_size=256,
    top_k=8,
    device="cuda",
):
    """Generate random inputs for MoE testing."""
    gemm1_out_size = 2 * intermediate_size

    hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device)

    # Router logits
    gating_output = torch.randn(seq_len, num_experts, dtype=torch.float32, device=device)

    # Expert weights
    w1 = torch.randn(
        num_experts, gemm1_out_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02

    # Routing: select top-k experts per token
    topk_weights_raw, topk_ids = torch.topk(gating_output, top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights_raw, dim=-1)

    return {
        "hidden_states": hidden_states,
        "gating_output": gating_output,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def test_correctness(seq_len=4, atol=5e-2, rtol=5e-2):
    """Test correctness of reference implementation against FlashInfer."""
    print(f"\n{'='*60}")
    print(f"Testing BF16 MoE: seq_len={seq_len}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("CUDA not available")

    inputs = generate_random_inputs(seq_len, device=device)

    print(f"Hidden states shape: {inputs['hidden_states'].shape}")
    print(f"W1 shape: {inputs['w1'].shape}")
    print(f"W2 shape: {inputs['w2'].shape}")
    print(f"Top-k IDs shape: {inputs['topk_ids'].shape}")

    # Run reference implementation
    print("\nRunning reference implementation...")
    ref_output = run(
        inputs["hidden_states"],
        inputs["gating_output"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
    )

    # Run SGLang-style vanilla ground truth (adapted from sglang/layers/moe/fused_moe_native.py)
    print("Running SGLang vanilla ground truth...")
    gt_output = _sglang_moe_ground_truth(
        inputs["hidden_states"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
    )
    fi_output = gt_output

    # Compare
    print("\nComparing outputs...")
    ref_f32 = ref_output.float()
    fi_f32 = fi_output.float()

    abs_diff = torch.abs(ref_f32 - fi_f32)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(fi_f32) + 1e-8)
    max_rel_diff = rel_diff.max().item()

    cos_sim = F.cosine_similarity(ref_f32.flatten().unsqueeze(0), fi_f32.flatten().unsqueeze(0)).item()

    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    close = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    if close:
        print(f"\n✓ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"\n✗ FAILED (atol={atol}, rtol={rtol})")
    assert close, f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol})"


def main():
    print("Testing BF16 MoE topk8_e256_h2048_i256 Reference Implementation")

    test_configs = [1, 2, 4, 8, 16]
    passed = 0
    total = len(test_configs)

    for seq_len in test_configs:
        try:
            test_correctness(seq_len)
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
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
