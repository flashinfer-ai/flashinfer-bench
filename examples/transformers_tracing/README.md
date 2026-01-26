# Transformers Model Tracing

This directory contains scripts for tracing transformers models with flashinfer-bench to collect operator workload traces.

## Important: How Tracing Works

The flashinfer-bench tracing system requires **pre-existing definitions** to match workloads against. When you run tracing:

1. The system loads definitions from a dataset directory (e.g., `flashinfer_trace/`)
2. As the model runs, adapters intercept operations and try to match them to definitions
3. Only operations with matching definitions are recorded as workload traces
4. Operations without matching definitions are logged as "Definition X not found" (this is expected)

## Currently Supported Operators

The `flashinfer_trace/` dataset in this repository includes definitions for:

| Operator | Definition Pattern | Status |
|----------|-------------------|--------|
| Attention (GQA) | `gqa_ragged_prefill_causal_h{H}_kv{KV}_d{D}` | ✓ Supported |
| RMSNorm | `rmsnorm_h{H}` | ✓ Supported (h128, h512, h1536, h2048, h4096, h7168) |
| Linear/GEMM | `gemm_n{N}_k{K}` | ✓ Supported (various sizes) |
| Sampling | `top_k_sampling_from_probs_v{V}` | ✓ Supported |
| MoE | `moe_*` | ✓ Supported |
| MLA | `mla_paged_*` | ✓ Supported |

The following operators are **intercepted** but don't have definitions yet:

| Operator | Definition Pattern | Status |
|----------|-------------------|--------|
| Embedding | `embedding_v{V}_d{D}` | ⚠ Needs definitions |
| RoPE | `rope_h{H}_d{D}` | ⚠ Needs definitions |
| SiLU | `silu_h{H}` | ⚠ Needs definitions |
| GELU | `gelu_h{H}` | ⚠ Needs definitions |
| Softmax | `softmax_d{D}` | ⚠ Needs definitions |
| Top-K | `topk_d{D}_k{K}` | ⚠ Needs definitions |

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers accelerate torch flashinfer-bench
```

### 2. Trace a Model

The script automatically finds the `flashinfer_trace/` dataset in the repository:

```bash
cd /path/to/flashinfer-bench/examples/transformers_tracing

# Trace LLaMA 3.1 8B (uses default dataset path)
python trace_models.py --model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 16

# Or specify the dataset path explicitly
python trace_models.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset /path/to/flashinfer-bench/flashinfer_trace \
    --max-new-tokens 16
```

### 3. Check the Output

Workload traces are saved to `flashinfer_trace/workloads/`:

```bash
# List traced workloads
ls flashinfer_trace/workloads/
```

Expected output:
- `gqa_ragged/*.jsonl` - Attention workloads
- `rmsnorm/*.jsonl` - RMSNorm workloads  
- `gemm/*.jsonl` - Linear/GEMM workloads

### 4. Understanding "Definition not found" Messages

When running tracing, you'll see messages like:
```
Definition embedding_v128256_d4096 not found
Definition rope_h32_d128 not found
Definition silu_h14336 not found
```

**This is expected!** These messages indicate that the adapters are intercepting the operations, but no matching definitions exist in the dataset. To trace these operations, you would need to add the corresponding definitions to `flashinfer_trace/definitions/`.

## Tracing Large Models

For large models (70B+), you'll need:

1. **Multi-GPU setup** with sufficient VRAM
2. **accelerate** for device mapping:

```bash
pip install accelerate

python trace_models.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-new-tokens 16
```

3. **FP8 quantization** for memory efficiency:

```bash
python trace_models.py --model RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8
```

## Alternative: Environment Variables

You can enable tracing via environment variables:

```bash
export FIB_ENABLE_TRACING=1
export FIB_DATASET_PATH=/path/to/flashinfer-bench/flashinfer_trace

python your_inference_script.py
```

## Adding New Definitions

To trace additional operators, add JSON definition files to `flashinfer_trace/definitions/`. See existing definitions for the expected format:

```bash
# Example: Check existing rmsnorm definition
cat flashinfer_trace/definitions/rmsnorm/rmsnorm_h4096.json
```

Each definition specifies:
- `name`: Unique identifier matching the adapter's generated name
- `op_type`: Category for organizing traces
- `axes`: Shape parameters (const or var)
- `inputs`: Expected input tensors with shapes and dtypes
- `outputs`: Expected output tensors
- `reference`: Python reference implementation for validation

## Troubleshooting

### No traces being saved

1. **Check dataset path**: Ensure you're using a path with definitions
   ```bash
   ls /path/to/dataset/definitions/
   ```

2. **Check if definitions match**: The model's tensor shapes must match definition axes
   - Example: `rmsnorm_h4096` only matches hidden_size=4096
   - LLaMA-8B uses hidden_size=4096 ✓
   - Different models may need different definitions

### PyTorch version issues

Some adapters require specific PyTorch versions:
- `torch.nn.functional.rms_norm` requires PyTorch 2.4+
- Flash Attention requires PyTorch 2.0+

### Memory issues

For large models:
1. Use FP8 quantized versions when available
2. Use `--dtype bfloat16` to reduce memory
3. Reduce `--max-new-tokens` for initial testing

## Supported Models

| Model Key | Model ID | Architecture |
|-----------|----------|--------------|
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Dense |
| `llama-3.1-70b` | `meta-llama/Llama-3.1-70B-Instruct` | Dense |
| `llama-3.1-70b-fp8` | `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` | Dense + FP8 |
| `qwen3-30b-moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE (3B active) |
| `qwen3-30b-moe-fp8` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | MoE + FP8 |
