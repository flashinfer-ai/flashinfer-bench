# FlashInfer-Bench Automated Model Addition Guide

This document explains how to automatically extract model information from HuggingFace and SGLang codebase, and integrate them into FlashInfer-Bench.

## Project Overview

FlashInfer-Bench is a GPU kernel optimization benchmarking framework for:
- Standardizing FlashInfer Trace format
- Real-time workload tracing and collection
- Automated kernel optimization and replacement
- Performance leaderboards and tracking

### Core Concepts

1. **Definition**: Specifies an operation's interface (inputs/outputs, axes, reference implementation)
2. **Solution**: Concrete implementation of a Definition (Python/Triton/CUDA)
3. **Workload**: Specific input configuration and test case
4. **Trace**: Execution record containing correctness and performance data
5. **Model**: Hierarchical module structure mapping model components to Definitions

## Project Architecture

```
flashinfer-bench/
├── flashinfer_bench/           # Main source code
│   ├── data/                   # Definition, Solution, Trace data structures
│   ├── compile/                # Build system (Python/Triton/CUDA)
│   ├── bench/                  # Benchmarking engine
│   ├── apply/                  # Kernel auto-replacement API
│   └── integration/            # FlashInfer integration
├── web/apps/web/data/          # Model definitions (TypeScript)
│   └── models.ts               # DeepSeek/Llama/Qwen model definitions
├── examples/                   # Example code
│   └── kernel_generator/       # AI-driven kernel generator
└── skills/                     # Automated workflow scripts
    └── add-new-model/          # New model addition workflow
```

## Model Definition Structure

Models are defined in `web/apps/web/data/models.ts` with a hierarchical structure:

```typescript
{
  id: "model-id",                    // Unique model identifier
  name: "Display Name",              // Display name
  description: "Model description",  // Description
  modules: {                         // Hierarchical modules
    ModuleName: {
      count: 32,                     // Repetition count
      parent: "ParentModule",        // Parent module
      type: "block" | "layer",       // Type
      definitions: [                 // Associated kernel definitions
        "rmsnorm_h4096",
        "gqa_paged_decode_h32_kv8_d128_ps1"
      ]
    }
  }
}
```

### Existing Model Examples

- **DeepSeek V3/R1**: MLA architecture, 61 layers, MoE
- **Llama 3.1 8B**: Standard GQA architecture, 32 layers
- **Qwen3 30B A3B**: MoE architecture, 32 layers

## Supported Operation Types

FlashInfer-Bench supports the following op_types (corresponding to different Definition types):

| Operation Type | Description | Example |
|---------------|-------------|---------|
| `rmsnorm` | RMS Layer Normalization | `rmsnorm_h4096` |
| `gemm` | General Matrix Multiplication | `gemm_n_6144_k_4096` |
| `gqa_ragged` | Group Query Attention (ragged) | `gqa_ragged_prefill_causal_h32_kv8_d128` |
| `gqa_paged` | Group Query Attention (paged) | `gqa_paged_decode_h32_kv8_d128_ps1` |
| `mla_paged` | Multi-Head Latent Attention (paged) | `mla_paged_decode_h16_ckv512_kpe64_ps1` |
| `dsa_paged` | DeepSeek Sparse Attention (paged) | `dsa_sparse_decode_h16_ckv512_kpe64_topk256_ps1` |
| `gdn` | Gated Delta Net (linear attention) | `gdn_decode_qk16_v32_d128_k_last` |
| `moe` | Mixture of Experts | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| `sampling` | Sampling operations | - |

## Automated Workflow

### Using Skills to Add New Models

We provide a suite of skills to automate the model addition process:

```bash
# Add new model (complete workflow)
claude-code run add-new-model --model-name kimi-k2

# Or execute step by step:

# 1. Extract model config from HuggingFace
claude-code run extract-model-from-hf --model-id moonshot-ai/kimi-k2

# 2. Find baseline implementation from SGLang
claude-code run find-sglang-baseline --model-name kimi

# 3. Generate model definition file
claude-code run generate-model-definition --config config.json --output web/apps/web/data/models.ts

# 4. Collect real-world workloads from inference runs
claude-code run collect-workloads --op-type mla_paged --model-name deepseek-v3 --num-samples 100
```

### Manual Model Addition Process

If you need to add manually, follow these steps:

#### 1. Obtain Model Architecture Information

Get the model's `config.json` from HuggingFace:

```python
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(
    repo_id="moonshot-ai/kimi-k2",
    filename="config.json"
)
```

Key configuration items:
- `num_hidden_layers`: Number of layers
- `hidden_size`: Hidden dimension
- `num_attention_heads`: Number of attention heads
- `num_key_value_heads`: Number of KV heads (for GQA)
- `intermediate_size`: MLP intermediate size

#### 2. Find SGLang Baseline Implementation

Locate the corresponding model implementation in SGLang codebase:

```bash
# Clone SGLang
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Search for model implementation
grep -r "class.*ForCausalLM" python/sglang/srt/models/
```

Files to examine:
- `python/sglang/srt/models/{model_name}.py`
- Kernel calls in forward pass
- attention/MLP/normalization implementations

#### 3. Create Model Definition

Add new model to `web/apps/web/data/models.ts`:

```typescript
{
  id: "kimi-k2",
  name: "Kimi K2",
  description: "Moonshot AI Kimi K2 model",
  modules: {
    // Reference DeepSeek/Llama/Qwen structure
    // Fill based on config.json and SGLang implementation
  }
}
```

#### 4. Map Modules to Definitions

**IMPORTANT**: When creating definitions for new kernels, always refer to the HuggingFace model page (`https://huggingface.co/{org}/{model-name}`) to obtain authoritative model constants from `config.json`. Cross-reference with SGLang implementation for runtime-specific values like `page_size`.

Associate each module with corresponding Definitions:

- **Normalization layers**: `rmsnorm_h{hidden_size}`, `fused_add_rmsnorm_h{hidden_size}`
- **Attention layers**:
  - GQA: `gqa_paged_decode_h{num_heads}_kv{kv_heads}_d{head_dim}_ps1`
  - MLA: `mla_paged_decode_h{num_heads}_ckv{ckv_dim}_kpe{kpe_dim}_ps1`
  - DSA: `dsa_sparse_decode_h{num_heads}_ckv{ckv_dim}_kpe{kpe_dim}_topk{topk}_ps1` (sparse MLA)
  - GDN: `gdn_decode_qk{q_heads}_v{v_heads}_d{head_dim}` (linear attention)
- **GEMM layers**: `gemm_n_{out_dim}_k_{in_dim}`
- **MoE layers**: `moe_fp8_block_scale_ds_routing_topk{topk}_ng{num_groups}_kg{group_size}_e{num_experts}_h{hidden}_i{intermediate}`

#### 5. Validation and Testing

```bash
# Start web interface to view model
cd web/apps/web
pnpm install
pnpm dev

# Run benchmarks
flashinfer-bench run --local /path/to/dataset --definitions <your-definitions>
```

## Skills Detailed Documentation

### add-new-model

Main workflow skill that integrates all steps.

**Parameters**:
- `--model-name`: Model name (e.g., "kimi-k2")
- `--hf-repo-id`: HuggingFace repo ID (e.g., "moonshot-ai/kimi-k2")
- `--sglang-path`: SGLang codebase path (optional, default ./sglang)

**Output**:
- Updated `web/apps/web/data/models.ts`
- Model architecture analysis report
- Definition mapping suggestions

### extract-model-from-hf

Extract model configuration from HuggingFace.

**Parameters**:
- `--model-id`: HuggingFace model ID
- `--output`: Output JSON path (optional)

**Output**:
- `model_config.json`: Model configuration file
- `model_architecture.json`: Parsed architecture information

### find-sglang-baseline

Find baseline implementation from SGLang codebase.

**Parameters**:
- `--model-name`: Model name or class name keyword
- `--sglang-path`: SGLang codebase path

**Output**:
- `sglang_implementation.json`: Implementation details
- Model file paths and key code snippets

### generate-model-definition

Generate TypeScript model definition.

**Parameters**:
- `--config`: Model config JSON path
- `--sglang-impl`: SGLang implementation JSON path (optional)
- `--output`: Output path (default web/apps/web/data/models.ts)

**Output**:
- Updated models.ts file
- TypeScript code for module definitions

### collect-workloads

Auto-collect real-world workloads from SGLang inference runs using FlashInfer Level 10 logging API.

**Parameters**:
- `--definition-names`: List of specific definition names to collect workloads for (optional)
- `--op-type`: Collect workloads for all definitions of a specific op_type (optional)
- `--all`: Collect workloads for ALL definitions (optional)
- `--model-name`: Model to run inference on (required, e.g., "deepseek-v3", "llama-3.1-8b")
- `--dataset`: Path to ShareGPT-format JSONL dataset (optional)
- `--num-samples`: Number of inference samples to process (default: 100)
- `--submit-pr`: Whether to submit PR to flashinfer-trace repo (default: true)

**Output**:
- Workload JSONL files in `flashinfer_trace/workloads/{op_type}/{definition_name}.jsonl`
- Safetensors files for large tensors (if applicable)
- Pull request to `flashinfer-ai/flashinfer-trace` dataset repo

**Workflow**:
1. Setup FlashInfer Level 10 logging (tensor dump mode)
2. Run SGLang inference with ShareGPT dataset
3. Dump tensors locally from FlashInfer logs
4. Sanitize tensors according to kernel definitions
5. Convert to workload JSONL format with deduplication
6. Submit PR to flashinfer-trace HuggingFace dataset repo

## Common Model Architecture Patterns

### Standard Transformer (e.g., Llama)

```
Model
├── embed_tokens
├── layers (n x DecoderLayer)
│   ├── input_layernorm → rmsnorm
│   ├── self_attn
│   │   ├── qkv_proj → gemm
│   │   ├── rotary_emb
│   │   ├── attn → gqa_paged_decode/prefill
│   │   └── o_proj → gemm
│   ├── post_attention_layernorm → rmsnorm
│   └── mlp
│       ├── gate_up_proj → gemm
│       ├── act_fn
│       └── down_proj → gemm
├── norm → rmsnorm
└── lm_head
```

### MoE Architecture (e.g., DeepSeek, Qwen)

MLP layer replaced with:
```
mlp
├── moe_gate → moe (routing)
├── moe_topk → moe (selection)
└── moe_experts → moe (execution)
```

### MLA Architecture (e.g., DeepSeek V3)

Attention replaced with:
```
self_attn
├── fused_qkv_a_proj_with_mqa
├── q_a_layernorm → rmsnorm
├── q_b_proj → gemm
├── kv_a_layernorm → rmsnorm
├── kv_b_proj → gemm
├── rotary_emb
├── attn_mla → mla_paged_decode/prefill
└── o_proj → gemm
```

### GDN Architecture (e.g., Qwen3 Next)

Linear attention layers using Gated Delta Net:
```
self_attn
├── q_proj → gemm
├── k_proj → gemm
├── v_proj → gemm
├── gating_proj → gemm (produces a, b for gating)
├── attn_gdn → gdn_prefill/gdn_decode
└── o_proj → gemm
```

Where GDN maintains a recurrent state [B, H, K, V] and uses:
- `A_log`: learnable log decay parameter
- `a`: input-dependent decay (combined with dt_bias)
- `b`: update gate input (transformed via sigmoid)

## Extension and Contributing

### Adding New Operation Types

To support new operation types (beyond existing rmsnorm/gemm/gqa/mla/moe):

1. Create operation documentation in `docs/op_type_schema/`
2. Create Definition JSON (input/output/axes specification)
3. Provide Python reference implementation
4. Create Solution (Triton/CUDA optimized implementation)
5. Optional: Create FlashInfer Adapter

### Contributing to Official Dataset

Optimized kernels can be submitted to:
```
https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
```

## Maintaining Documentation

**When making architectural changes, update these files accordingly:**
- `CLAUDE.md` - project overview, supported op_types, architecture patterns
- `.claude/skills/*.md` - skill-specific documentation

## References

- [FlashInfer Documentation](https://docs.flashinfer.ai)
- [FlashInfer Logging API](https://docs.flashinfer.ai/logging.html)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [HuggingFace Hub](https://huggingface.co/models)
- [flashinfer-ai/flashinfer-trace Dataset](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
- [Definition Schema Documentation](docs/flashinfer_trace/definition.md)
- [Operation Type Schema](docs/op_type_schema/)

## Troubleshooting

### Issue: Cannot find corresponding Definition

**Solution**:
1. Check existing definitions directory for similar definitions
2. Use kernel_generator to generate new optimized implementations
3. Start from flashinfer baseline solution

### Issue: Model architecture doesn't match existing patterns

**Solution**:
1. Reference the most similar existing model (DeepSeek/Llama/Qwen)
2. Create new operation type schema
3. Contact FlashInfer team for support

### Issue: Model not implemented in SGLang

**Solution**:
1. Check HuggingFace transformers library
2. Use generic transformer architecture as baseline
3. Infer architecture from model config

## Example: Adding Kimi K2

Complete example workflow:

```bash
# 1. Run automated workflow
claude-code run add-new-model \
  --model-name kimi-k2 \
  --hf-repo-id moonshot-ai/kimi-k2

# 2. View generated definition
cat web/apps/web/data/models.ts

# 3. Start web interface for validation
cd web/apps/web && pnpm dev

# 4. Collect real-world workloads
claude-code run collect-workloads \
  --model-name kimi-k2 \
  --op-type gqa_paged \
  --num-samples 200

# 5. Run benchmarks with collected workloads
flashinfer-bench run --local ./data --definitions <generated-defs>
```

Expected output:
- New kimi-k2 entry in `web/apps/web/data/models.ts`
- `model_analysis_kimi-k2.json` containing architecture analysis
- List of Definition mapping suggestions
- Workload JSONL files in `flashinfer_trace/workloads/{op_type}/`
- Pull request to `flashinfer-ai/flashinfer-trace` dataset
