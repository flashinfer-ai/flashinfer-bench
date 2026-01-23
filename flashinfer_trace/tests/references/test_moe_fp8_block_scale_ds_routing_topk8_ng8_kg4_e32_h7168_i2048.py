"""
Test MoE FP8 Block Scale reference implementation against FlashInfer.

This test validates that the reference implementation from the definition
matches the FlashInfer kernel implementation.
"""

import json
from pathlib import Path

import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from safetensors.torch import load_file
from test_utils import get_reference_run

# Load reference implementation from definition
run = get_reference_run("moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048")

TRACE_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_JSONL_PATH = (
    TRACE_ROOT
    / "workloads"
    / "moe"
    / "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"
)


# -----------------------------
# Helpers: FP8 block quantization (dequant scale semantics) - Vectorized
# -----------------------------
def _fp8_block_quant_1d(x_bf16: torch.Tensor, block: int = 128):
    """
    Quantize [T, H] activations into FP8 with per-(token, 128-col) block scales.
    Returns:
      x_fp8: [T, H] (float8_e4m3fn)
      scales_TxNb: [T, H/128] (float32)  -- dequant scales (float ≈ fp8 * scale)
    """
    assert x_bf16.dim() == 2
    T, H = x_bf16.shape
    assert H % block == 0
    nb = H // block

    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    x_f32 = x_bf16.to(torch.float32)

    # Reshape to [T, nb, block] for vectorized block operations
    x_blocked = x_f32.view(T, nb, block)

    # Compute per-block amax: [T, nb]
    amax = torch.amax(torch.abs(x_blocked), dim=2)

    # Compute scales (dequant scale = amax / max_fp8)
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))

    # Quantize: x_fp8 = x / scale
    x_scaled = x_blocked / scales.unsqueeze(2)
    x_fp8 = x_scaled.view(T, H).to(torch.float8_e4m3fn)

    return x_fp8, scales  # scales in [T, H/128]


def _fp8_block_quant_2d(w_bf16: torch.Tensor, block: int = 128):
    """
    Quantize weights with 2D block scales over the last two dims.
      w_bf16: [*, R, C]  (R and C are multiples of 128)
    Returns:
      w_fp8: [*, R, C] (float8_e4m3fn)
      scales: [*, R/128, C/128] (float32) -- dequant scales

    Fully vectorized implementation for speed.
    """
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r = R // block
    nb_c = C // block

    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    w_f32 = w_bf16.to(torch.float32).contiguous()

    # Reshape to [*, nb_r, block, nb_c, block] for vectorized block operations
    # Original shape: [*, R, C] -> [*, nb_r, block, nb_c, block]
    new_shape = (*prefix, nb_r, block, nb_c, block)
    w_blocked = w_f32.view(new_shape)

    # Compute per-block amax: [*, nb_r, nb_c]
    # Reduce over the block dimensions (dims -3 and -1 after reshape)
    amax = torch.amax(torch.abs(w_blocked), dim=(-3, -1))  # [*, nb_r, nb_c]

    # Compute scales (dequant scale = amax / max_fp8)
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))

    # Expand scales back to block shape for division
    # scales: [*, nb_r, nb_c] -> [*, nb_r, 1, nb_c, 1]
    scales_expanded = scales.unsqueeze(-2).unsqueeze(-1)  # [*, nb_r, 1, nb_c, 1]

    # Quantize: w_fp8 = w / scale
    w_scaled = w_blocked / scales_expanded
    w_fp8 = w_scaled.view(*prefix, R, C).to(torch.float8_e4m3fn)

    return w_fp8, scales


# read jsonl file to locate the workload record at index
def _load_workload_record(workload_index: int):
    if not WORKLOAD_JSONL_PATH.exists():
        raise FileNotFoundError(f"Workload JSONL not found: {WORKLOAD_JSONL_PATH}")

    record = None
    with WORKLOAD_JSONL_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            if idx == workload_index:
                record = json.loads(stripped)
                break

    if record is None:
        raise IndexError(f"No workload entry at index {workload_index}")

    return record


def _load_workload_tensors(record: dict, *, device: str):
    HIDDEN_SIZE = 7168
    BLOCK_SIZE = 128

    workload = record["workload"]
    inputs_spec = workload["inputs"]

    tensor_cache = {}

    def fetch_tensor(spec: dict):
        if spec["type"] != "safetensors":
            raise ValueError(f"Unsupported spec type: {spec['type']}")

        file_path = Path(spec["path"])
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path

        if file_path not in tensor_cache:
            tensor_cache[file_path] = load_file(file_path)

        tensors = tensor_cache[file_path]
        tensor_key = spec["tensor_key"]
        if tensor_key not in tensors:
            raise KeyError(f"Tensor key '{tensor_key}' not found in {file_path}")
        return tensors[tensor_key]

    seq_len = workload["axes"]["seq_len"]

    routing_logits = fetch_tensor(inputs_spec["routing_logits"]).to(torch.float32).to(device)
    routing_bias = fetch_tensor(inputs_spec["routing_bias"]).to(device)
    if routing_bias.dtype != torch.bfloat16:
        routing_bias = routing_bias.to(torch.bfloat16)

    hidden_states = fetch_tensor(inputs_spec["hidden_states"]).to(device)
    hidden_states_scale = fetch_tensor(inputs_spec["hidden_states_scale"]).to(torch.float32)
    expected_scale_shape = (HIDDEN_SIZE // BLOCK_SIZE, seq_len)
    if hidden_states_scale.shape == (seq_len, HIDDEN_SIZE // BLOCK_SIZE):
        hidden_states_scale = hidden_states_scale.permute(1, 0).contiguous()
    if hidden_states_scale.shape != expected_scale_shape:
        raise ValueError(
            f"Unexpected hidden_states_scale shape: {hidden_states_scale.shape}, expected {expected_scale_shape}"
        )
    hidden_states_scale = hidden_states_scale.to(device)

    local_expert_offset = int(inputs_spec["local_expert_offset"]["value"])
    routed_scaling_factor = float(inputs_spec["routed_scaling_factor"]["value"])

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "local_expert_offset": local_expert_offset,
        "routed_scaling_factor": routed_scaling_factor,
    }, {"seq_len": seq_len, "uuid": workload.get("uuid", "unknown")}


def prepare_inputs_from_workload(workload_index: int, *, device: str):
    HIDDEN_SIZE = 7168
    INTERMEDIATE_SIZE = 2048
    NUM_EXPERTS_GLOBAL = 256
    NUM_EXPERTS_LOCAL = 32

    record = _load_workload_record(workload_index)
    real_inputs, metadata = _load_workload_tensors(record, device=device)

    seq_len = metadata["seq_len"]

    base_inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=NUM_EXPERTS_GLOBAL,
        num_local_experts=NUM_EXPERTS_LOCAL,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        use_bias=True,
        local_expert_offset=real_inputs["local_expert_offset"],
        routed_scaling_factor=real_inputs["routed_scaling_factor"],
        device=device,
    )

    for key in ("routing_logits", "routing_bias", "hidden_states", "hidden_states_scale"):
        base_inputs[key] = real_inputs[key]

    base_inputs["local_expert_offset"] = real_inputs["local_expert_offset"]
    base_inputs["routed_scaling_factor"] = real_inputs["routed_scaling_factor"]

    return base_inputs, {**metadata, "workload_index": workload_index}


def _compare_reference_vs_kernel(
    inputs: dict, *, seq_len: int, atol: float, rtol: float, percent: float
):
    HIDDEN_SIZE = 7168
    INTERMEDIATE_SIZE = 2048
    NUM_EXPERTS_GLOBAL = 256
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    print("Running reference...")
    ref_out = run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    print("Running FlashInfer kernel...")
    fi_out = trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"].to(torch.float32),
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"].to(torch.float32),
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"].to(torch.float32),
        num_experts=NUM_EXPERTS_GLOBAL,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=inputs["local_expert_offset"],
        local_num_experts=inputs["local_num_experts"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        routing_method_type=2,  # DeepSeek-V3 routing
        use_shuffled_weight=False,
        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
    )

    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    rel_diff = abs_diff / (fi_f32.abs() + 1e-8)

    print("\nComparison stats:")
    print(f"Max abs diff:  {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Max rel diff:  {rel_diff.max().item():.6e}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - fi_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    allclose = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    print(f"\nAllclose(atol={atol}, rtol={rtol}): {allclose}")

    if not allclose:
        flat = abs_diff.flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print("\nTop-5 absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            t = idx // HIDDEN_SIZE
            h = idx % HIDDEN_SIZE
            print(
                f"  [t={t}, h={h}]: ref={ref_f32.flatten()[idx].item():.6e}, "
                f"fi={fi_f32.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}"
            )

    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


# -----------------------------
# Random input generator for MoE DS-V3
# -----------------------------
@torch.no_grad()
def generate_random_inputs_moe(
    seq_len: int,
    *,
    num_experts_global: int = 256,
    num_local_experts: int = 32,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    use_bias: bool = True,
    local_expert_offset: int = 0,
    routed_scaling_factor: float = 2.5,
    device: str = "cuda",
):
    assert hidden_size % 128 == 0 and intermediate_size % 128 == 0
    T, H, I = seq_len, hidden_size, intermediate_size
    E_global, E_local = num_experts_global, num_local_experts

    # Inputs for routing
    routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)

    if use_bias:
        routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
    else:
        routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device=device)

    # Boost logits AND bias for local expert range to ensure they get selected
    # DeepSeek V3 routing uses s_with_bias = sigmoid(logits) + bias for group selection
    # Both logits and bias need boosting to guarantee local experts are selected
    local_end = min(local_expert_offset + E_local, E_global)
    routing_logits[:, local_expert_offset:local_end] += 10.0
    # Ensure bias is positive for local experts (add large positive value)
    routing_bias[local_expert_offset:local_end] = (
        routing_bias[local_expert_offset:local_end].abs() + 5.0
    )

    # Activations: start from bf16, then FP8 block-quant with dequant scales
    a_bf16 = 2.0 * torch.randn(T, H, dtype=torch.bfloat16, device=device)
    a_fp8, a_scales_TxNb = _fp8_block_quant_1d(a_bf16, block=128)  # scales: [T, H/128]
    hidden_states = a_fp8
    hidden_states_scale = a_scales_TxNb.transpose(0, 1).contiguous()  # [H/128, T]

    # Weights per local expert
    # W13: [E_local, 2I, H], W2: [E_local, H, I]
    w13_bf16 = torch.randn(E_local, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_bf16 = torch.randn(E_local, H, I, dtype=torch.bfloat16, device=device)

    w13_fp8, w13_scales = _fp8_block_quant_2d(w13_bf16, block=128)  # scales: [E, (2I)/128, H/128]
    w2_fp8, w2_scales = _fp8_block_quant_2d(w2_bf16, block=128)  # scales: [E, H/128, I/128]

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13_fp8,
        "gemm1_weights_scale": w13_scales,
        "gemm2_weights": w2_fp8,
        "gemm2_weights_scale": w2_scales,
        "local_expert_offset": int(local_expert_offset),
        "local_num_experts": E_local,
        "routed_scaling_factor": float(routed_scaling_factor),
    }


# -----------------------------
# Test driver
# -----------------------------
def test_correctness_moe(
    seq_len: int = 32,
    *,
    local_expert_offset: int = 0,
    use_bias: bool = True,
    atol: float = 1e-1,
    rtol: float = 2e-1,
    percent: float = 0.85,
):
    print("\n" + "=" * 70)
    print(
        f"Testing MoE FP8 Block-Scale: seq_len={seq_len}, offset={local_expert_offset}, use_bias={use_bias}"
    )
    print("=" * 70)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping test.")
        return True

    if trtllm_fp8_block_scale_moe is None:
        print("WARNING: flashinfer fused_moe kernel not available.")
        return False

    device = "cuda"
    torch.manual_seed(42)

    # Constants (DeepSeek-V3)
    E_GLOBAL = 256
    E_LOCAL = 32
    H = 7168
    I = 2048
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Generate random but consistent inputs
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
        use_bias=use_bias,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=2.5,
        device=device,
    )

    # Run reference (returns bf16)
    print("Running reference...")
    ref_out = run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    # Run FlashInfer fused kernel
    print("Running FlashInfer kernel...")
    fi_out = trtllm_fp8_block_scale_moe(
        routing_logits=inputs["routing_logits"].to(torch.float32),
        routing_bias=inputs["routing_bias"],  # bf16
        hidden_states=inputs["hidden_states"],  # fp8
        hidden_states_scale=inputs["hidden_states_scale"],  # [H/128, T]
        gemm1_weights=inputs["gemm1_weights"],  # fp8
        gemm1_weights_scale=inputs["gemm1_weights_scale"].to(torch.float32),
        gemm2_weights=inputs["gemm2_weights"],  # fp8
        gemm2_weights_scale=inputs["gemm2_weights_scale"].to(torch.float32),
        num_experts=E_GLOBAL,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        local_expert_offset=inputs["local_expert_offset"],
        local_num_experts=inputs["local_num_experts"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        routing_method_type=2,  # DeepSeek-V3 routing
        use_shuffled_weight=False,
        tune_max_num_tokens=max(8, min(seq_len * TOP_K, 8192)),
    )

    # Compare
    ref_f32 = ref_out.float()
    fi_f32 = fi_out.float()

    abs_diff = (ref_f32 - fi_f32).abs()
    rel_diff = abs_diff / (fi_f32.abs() + 1e-8)

    print("\nComparison stats:")
    print(f"Max abs diff:  {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Max rel diff:  {rel_diff.max().item():.6e}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6e}")

    # Cosine similarity and MSE
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), fi_f32.flatten(), dim=0
    ).item()
    mse = torch.mean((ref_f32 - fi_f32) ** 2).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"MSE: {mse:.6e}")

    # Strict allclose
    allclose = torch.allclose(ref_f32, fi_f32, atol=atol, rtol=rtol)
    print(f"\nAllclose(atol={atol}, rtol={rtol}): {allclose}")

    if not allclose:
        # Show top-5 largest absolute errors
        flat = abs_diff.flatten()
        k = min(5, flat.numel())
        topv, topi = torch.topk(flat, k)
        print("\nTop-5 absolute error locations:")
        for rank in range(k):
            idx = topi[rank].item()
            t = idx // H
            h = idx % H
            print(
                f"  [t={t}, h={h}]: ref={ref_f32.flatten()[idx].item():.6e}, "
                f"fi={fi_f32.flatten()[idx].item():.6e}, diff={topv[rank].item():.6e}"
            )

    left = (ref_f32 - fi_f32).abs()
    right = atol + rtol * fi_f32.abs()
    ok = left <= right
    hit_ratio = ok.float().mean().item()
    print(f"\nHit ratio: {hit_ratio * 100:.2f}%  (need >= {percent * 100:.2f}%)")

    return hit_ratio >= percent


def test_moe_with_real_workload():
    device = "cuda"
    torch.manual_seed(42)

    # Select workload index deterministically for reproducibility
    workload_index = 0

    inputs, meta = prepare_inputs_from_workload(workload_index, device=device)

    atol = 1e-1
    rtol = 2e-1
    percent = 0.85

    ok = _compare_reference_vs_kernel(
        inputs, seq_len=meta["seq_len"], atol=atol, rtol=rtol, percent=percent
    )

    assert ok, (
        f"FlashInfer output mismatched reference for workload index {workload_index} "
        f"(uuid={meta['uuid']})."
    )


def main():
    print("Testing FP8 Block-Scale MoE (DeepSeek-V3) Reference vs FlashInfer")

    configs = [
        # (seq_len, local_expert_offset, use_bias)
        (1, 0, False),
        (4, 0, True),
        (8, 64, True),
        (16, 32, True),
        (64, 128, True),
        (256, 64, True),
        (1024, 32, True),
    ]

    passed = 0
    for T, off, use_bias in configs:
        try:
            ok = test_correctness_moe(
                seq_len=T, local_expert_offset=off, use_bias=use_bias, percent=0.85
            )
            passed += int(ok)
        except Exception as e:
            print(f"\n× Test crashed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {passed}/{len(configs)} tests passed")
    print("=" * 70)

    print("Testing with real workload...")
    test_moe_with_real_workload()


if __name__ == "__main__":
    main()
