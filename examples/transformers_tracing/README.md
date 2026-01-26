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

| Operator | Definition Pattern | Supported Sizes |
|----------|-------------------|-----------------|
| **Attention (GQA)** | `gqa_ragged_prefill_causal_h{H}_kv{KV}_d{D}` | h32_kv4_d128, h32_kv8_d128, h64_kv8_d128, h64_kv8_d64 |
| **RMSNorm** | `rmsnorm_h{H}` | h128, h512, h1536, h2048, h2880, h4096, h7168, h8192 |
| **Fused Add+RMSNorm** | `fused_add_rmsnorm_h{H}` | h2048, h2880, h4096, h7168, h8192 |
| **Embedding** | `embedding_v{V}_d{D}` | v128256_d4096, v128256_d8192, v151936_d2048, v201088_d2880 |
| **RoPE** | `rope_h{H}_d{D}` | h32_d128, h64_d128, h64_d64 |
| **SiLU** | `silu_h{H}` | h768, h2880, h14336, h28672 |
| **Softmax** | `softmax_d{D}` | d128256, d151936, d201088 |
| **Top-K** | `topk_d{D}_k{K}` | d128256_k50, d151936_k50, d201088_k50 |
| **Multinomial** | `sampling_multinomial_v{V}` | v128256, v151936, v201088 |
| **Linear/GEMM** | `gemm_n{N}_k{K}` | various sizes |
| **MoE** | `moe_*` | various configurations |
| **MLA** | `mla_paged_*` | various configurations |

### Model Coverage

| Model | Operators Covered |
|-------|-------------------|
| **LLaMA-3.1-8B** | embedding, rope, silu, softmax, topk, multinomial, attention, rmsnorm |
| **LLaMA-3.1-70B** | embedding, rope, silu, softmax, topk, multinomial, attention, rmsnorm |
| **Qwen3-30B-A3B** | embedding, rope, silu, softmax, topk, multinomial, attention, rmsnorm, moe |
| **gpt-oss-120b** | embedding, rope, softmax, topk, multinomial, attention, rmsnorm, moe (uses custom GLU activation) |

The following operators are **intercepted** but may need additional definitions for specific model configurations:

| Operator | Definition Pattern | Status |
|----------|-------------------|--------|
| GELU | `gelu_h{H}` | ⚠ Add definitions as needed |
| GELU (tanh) | `gelu_tanh_h{H}` | ⚠ Add definitions as needed |

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

## What to Expect

After running tracing, you should see:

1. **"Tracing 'definition_name'" messages** - Operations being traced
2. **"Definition X not found" messages** - Expected for operators without definitions
3. **"Flush done. N entries selected"** - How many workloads were actually saved

**Important**: The "Total trace files: N" count includes pre-existing traces in the workloads directory. New traces are **appended** to existing JSONL files, so the file count may not change. Check the line count in JSONL files to see new entries:

```bash
# Count total workload entries
wc -l flashinfer_trace/workloads/*/*.jsonl
```

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

3. **Check the flush output**: Look for "Flush done. N entries selected" in the log.
   - If N=0, no workloads matched the filter criteria
   - The filter policy deduplicates by average sequence length to avoid redundant traces

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
