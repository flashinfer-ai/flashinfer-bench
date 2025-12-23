# Find SGLang Baseline Implementation

Find and analyze baseline kernel implementations from SGLang codebase.

## Description

This skill searches the SGLang codebase for model implementations, extracts kernel call patterns, and identifies baseline implementations for comparison.

## Parameters

- `model_name` (required): Model name or keyword to search (e.g., "kimi", "llama", "deepseek")
- `sglang_path` (optional): Path to SGLang repository (default: "./sglang")
- `output_dir` (optional): Directory to save output files (default: "./model_analysis")

## Usage

```bash
/find-sglang-baseline --model-name kimi
/find-sglang-baseline --model-name llama --sglang-path ~/repos/sglang
/find-sglang-baseline --model-name deepseek-v3 --output-dir ./deepseek_analysis
```

## What This Skill Does

1. Clones SGLang repository if not present at specified path
2. Searches for model implementation files:
   - `python/sglang/srt/models/{model_name}.py`
   - Related model files based on keyword matching
3. Analyzes the model implementation to find:
   - Forward pass structure
   - Kernel calls (FlashInfer, vLLM, custom)
   - Attention mechanism implementation
   - MLP/FFN implementation
   - Normalization layers
4. Extracts code patterns for:
   - Attention: `flashinfer.batch_decode`, `flashinfer.batch_prefill`
   - Normalization: `rms_norm`, `layer_norm`
   - GEMM: Linear layers, matrix multiplications
   - MoE: Expert routing and execution
5. Generates output files:
   - `sglang_implementation.json`: Implementation details
   - `kernel_calls.json`: List of kernel calls with parameters
   - `code_snippets/`: Extracted relevant code sections

## Output Format

### sglang_implementation.json

```json
{
  "model_name": "kimi",
  "model_file": "python/sglang/srt/models/kimi.py",
  "class_name": "KimiForCausalLM",
  "forward_pass_structure": {
    "embedding": "embed_tokens",
    "layers": [
      {
        "name": "input_layernorm",
        "type": "RMSNorm",
        "kernel": "rms_norm"
      },
      {
        "name": "self_attn",
        "type": "Attention",
        "kernels": ["flashinfer.batch_decode", "flashinfer.batch_prefill"]
      },
      {
        "name": "post_attention_layernorm",
        "type": "RMSNorm",
        "kernel": "rms_norm"
      },
      {
        "name": "mlp",
        "type": "MLP",
        "kernels": ["linear"]
      }
    ],
    "final_norm": "norm",
    "lm_head": "lm_head"
  },
  "kernel_calls": [
    {
      "function": "flashinfer.batch_decode",
      "location": "line 234",
      "parameters": {
        "q": "query_states",
        "k": "key_cache",
        "v": "value_cache",
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128
      }
    }
  ]
}
```

### kernel_calls.json

```json
[
  {
    "module": "input_layernorm",
    "kernel": "rms_norm",
    "baseline_impl": "sglang.srt.layers.layernorm.RMSNorm",
    "flashinfer_equivalent": "rmsnorm_h{hidden_size}",
    "code_location": "models/kimi.py:145-147"
  },
  {
    "module": "attn",
    "kernel": "flashinfer.batch_decode",
    "baseline_impl": "flashinfer.batch_decode_with_shared_prefix_paged_kv_cache",
    "flashinfer_equivalent": "gqa_paged_decode_h32_kv8_d128_ps1",
    "code_location": "models/kimi.py:234-242"
  }
]
```

## Requirements

- Git (to clone SGLang if needed)
- Python packages: `ast` (standard library)
- Network access to clone SGLang repository

## Implementation

When executed, this skill will:

1. **Repository Setup**:
   - Check if SGLang exists at specified path
   - Clone from `https://github.com/sgl-project/sglang.git` if missing
   - Verify it's the latest version (optional: `git pull`)

2. **Model File Discovery**:
   - Search `python/sglang/srt/models/` for matching files
   - Use fuzzy matching on model_name keyword
   - Fall back to searching all model files if no exact match

3. **Code Analysis**:
   - Parse Python AST to find model class
   - Extract `forward()` method
   - Identify kernel calls by pattern matching:
     - `flashinfer.*` calls
     - `torch.nn.functional.*` calls
     - Custom kernel invocations
   - Track parameter values and tensor shapes

4. **Baseline Identification**:
   - Map SGLang kernels to FlashInfer operations
   - Extract dimension information from code
   - Generate Definition name suggestions

5. **Output Generation**:
   - Save JSON files with structured data
   - Extract and save relevant code snippets
   - Generate mapping from SGLang kernels to Definitions

## Notes

- SGLang implements multiple backends (FlashInfer, vLLM, custom)
- Look for FlashInfer integration in `python/sglang/srt/layers/`
- Some models may have custom implementations not directly mappable
- The skill prioritizes FlashInfer-based implementations when available
- Review extracted kernel calls for accuracy

## Integration with Other Skills

Use output from this skill as input to `generate-model-definition`:

```bash
/find-sglang-baseline --model-name kimi
/generate-model-definition \
  --config model_analysis/model_config.json \
  --sglang-impl model_analysis/sglang_implementation.json
```
