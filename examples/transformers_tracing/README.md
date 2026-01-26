# Transformers Model Tracing

This directory contains scripts for tracing transformers models with flashinfer-bench to collect operator workload traces.

## Supported Models

| Model Key | Model ID | Architecture |
|-----------|----------|--------------|
| `qwen3-30b-moe` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE (3B active) |
| `qwen3-30b-moe-fp8` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | MoE + FP8 |
| `llama-3.1-70b` | `meta-llama/Llama-3.1-70B-Instruct` | Dense |
| `llama-3.1-70b-fp8` | `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` | Dense + FP8 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | MoE |

## Traced Operators

The integration traces the following operators:

| Operator | Definition Pattern | Description |
|----------|-------------------|-------------|
| Attention | `gqa_ragged_prefill_causal_h{H}_kv{KV}_d{D}` | Multi-head/GQA attention |
| RMSNorm | `rmsnorm_h{H}` | RMS layer normalization |
| RoPE | `rope_h{H}_d{D}` | Rotary position embedding |
| Embedding | `embedding_v{V}_d{D}` | Token embedding lookup |
| SiLU | `silu_h{H}` | SiLU activation (SwiGLU) |
| GELU | `gelu_h{H}` | GELU activation |
| MoE | `moe_{impl}_e{E}_h{H}_i{I}_topk{K}` | Mixture of Experts |
| Linear | `gemm_n{N}_k{K}` | Linear/GEMM operations |
| Softmax | `softmax_d{D}` | Softmax (generation) |
| Top-K | `topk_d{D}_k{K}` | Top-k sampling |

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers accelerate torch flashinfer-bench
```

### 2. Trace a Model

```bash
# Trace LLaMA 3.1 8B (smaller model for testing)
python trace_models.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output ./traces \
    --max-new-tokens 50

# Trace with custom prompts
python trace_models.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output ./traces \
    --prompts "Hello world" "What is AI?"
```

### 3. Verify Traces

```bash
# List all traced operators
python verify_traces.py --traces ./traces --list

# Verify for a specific model
python verify_traces.py --traces ./traces --model llama-3.1-70b

# Verbose output with definition names
python verify_traces.py --traces ./traces --model llama-3.1-70b -v
```

## Tracing Large Models

For large models (70B+), you'll need:

1. **Multi-GPU setup** with sufficient VRAM
2. **accelerate** for device mapping:

```bash
# Install accelerate
pip install accelerate

# The script automatically uses device_map="auto" for multi-GPU
python trace_models.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output ./traces/llama-70b
```

3. **FP8 quantization** for memory efficiency:

```bash
python trace_models.py \
    --model RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 \
    --output ./traces/llama-70b-fp8
```

## Alternative: Environment Variables

You can also enable tracing via environment variables:

```bash
export FIB_ENABLE_TRACING=1
export FIB_DATASET_PATH=./traces

python your_inference_script.py
```

This automatically enables tracing for any transformers model.

## Verification Report Example

```
======================================================================
TRACE VERIFICATION REPORT
Model: meta-llama/Llama-3.1-70B-Instruct
Traces path: ./traces
======================================================================

Total definitions traced: 45
Total trace entries: 1250

----------------------------------------------------------------------
OPERATOR COVERAGE:
----------------------------------------------------------------------
  ✓ attention       (required)   [120 traces]
  ✓ rmsnorm         (required)   [240 traces]
  ✓ rope            (required)   [120 traces]
  ✓ embedding       (required)   [5 traces]
  ✓ silu            (required)   [240 traces]
  ✓ linear          (required)   [480 traces]
  ✓ softmax         (optional)   [45 traces]

----------------------------------------------------------------------
STATUS: ✓ ALL REQUIRED OPERATORS COVERED
======================================================================
```

## Troubleshooting

### Missing Operators

If some operators are not traced:

1. **Check PyTorch version**: Some adapters require PyTorch 2.4+ (e.g., `torch.nn.functional.rms_norm`)
2. **Check model implementation**: Some models may use custom implementations
3. **Run more prompts**: Some operators only trigger during generation (softmax, top-k)

### Memory Issues

For large models:

1. Use FP8 quantized versions when available
2. Use `--dtype bfloat16` to reduce memory
3. Ensure sufficient GPU VRAM (70B requires ~140GB for bf16, ~70GB for FP8)

### CUDA Errors

Tracing only works on CUDA. Ensure:

1. CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Compatible PyTorch/CUDA versions installed

## Contributing

To add support for new models:

1. Add the model to `MODEL_CONFIGS` in `trace_models.py`
2. Add verification expectations to `MODEL_OPERATOR_EXPECTATIONS` in `verify_traces.py`
3. Test with the new model
