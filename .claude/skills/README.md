# FlashInfer-Bench Skills

Automated workflows for adding new models to FlashInfer-Bench.

## Quick Start

Add a new model with one command:

```bash
/add-new-model --model-name kimi-k2 --hf-repo-id moonshot-ai/kimi-k2
```

## Available Skills

### add-new-model
Main workflow - automates the complete model addition process.
[Documentation](./add-new-model.md)

### extract-model-from-hf
Extract model configuration from HuggingFace.
[Documentation](./extract-model-from-hf.md)

### find-sglang-baseline
Find baseline implementation from SGLang codebase.
[Documentation](./find-sglang-baseline.md)

### generate-model-definition
Generate TypeScript model definition file.
[Documentation](./generate-model-definition.md)

## Workflow

```
HuggingFace → extract → architecture.json
                              ↓
SGLang → find-baseline → implementation.json
                              ↓
                    generate-definition
                              ↓
                         models.ts
```

## Examples

```bash
# Add Llama model
/add-new-model --model-name llama-3.3-70b --hf-repo-id meta-llama/Llama-3.3-70B-Instruct

# Add MoE model
/add-new-model --model-name qwen-2.5-72b --hf-repo-id Qwen/Qwen2.5-72B-Instruct

# Skip SGLang search
/add-new-model --model-name custom --hf-repo-id org/model --skip-sglang true
```

## See Also

[CLAUDE.md](../../CLAUDE.md) - Complete guide to model addition
