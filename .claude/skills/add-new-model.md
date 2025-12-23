# Add New Model

Automated workflow to add a new model to FlashInfer-Bench by extracting configuration from HuggingFace and baseline implementation from SGLang.

## Description

This is the main orchestration skill that automates the entire process of adding a new model to FlashInfer-Bench. It combines the extract-model-from-hf, find-sglang-baseline, and generate-model-definition skills into a single streamlined workflow.

## Parameters

- `model_name` (required): Short model identifier (e.g., "kimi-k2", "llama-3.3-70b")
- `hf_repo_id` (required): HuggingFace repository ID (e.g., "moonshot-ai/kimi-k2")
- `display_name` (optional): Full display name (default: auto-generated from model_name)
- `description` (optional): Model description (default: auto-generated)
- `sglang_path` (optional): Path to SGLang repository (default: "./sglang")
- `output_dir` (optional): Working directory for intermediate files (default: "./model_analysis_{model_name}")
- `skip_sglang` (optional): Skip SGLang baseline search (default: false)

## Usage

```bash
# Basic usage (recommended)
/add-new-model --model-name kimi-k2 --hf-repo-id moonshot-ai/kimi-k2

# With custom display name and description
/add-new-model \
  --model-name llama-3.3-70b \
  --hf-repo-id meta-llama/Llama-3.3-70B-Instruct \
  --display-name "Llama 3.3 70B" \
  --description "Meta's Llama 3.3 70B parameter instruction-tuned model"

# Using existing SGLang repository
/add-new-model \
  --model-name qwen-2.5-72b \
  --hf-repo-id Qwen/Qwen2.5-72B-Instruct \
  --sglang-path ~/repos/sglang

# Skip SGLang search (use only HuggingFace config)
/add-new-model \
  --model-name custom-model \
  --hf-repo-id org/custom-model \
  --skip-sglang true
```

## What This Skill Does

This skill orchestrates a complete workflow:

### Phase 1: Extract Model Configuration (extract-model-from-hf)
1. Download `config.json` from HuggingFace
2. Parse model architecture information
3. Identify model type and key parameters
4. Generate dimension calculations
5. Save `model_config.json` and `model_architecture.json`

### Phase 2: Find Baseline Implementation (find-sglang-baseline)
1. Clone/update SGLang repository if needed
2. Search for model implementation files
3. Analyze forward pass and kernel calls
4. Extract baseline kernel implementations
5. Save `sglang_implementation.json` and `kernel_calls.json`

### Phase 3: Generate Model Definition (generate-model-definition)
1. Load data from Phase 1 and 2
2. Determine architecture pattern (Standard/MoE/MLA/Custom)
3. Generate hierarchical module structure
4. Map modules to FlashInfer Definitions
5. Generate TypeScript code
6. Update `web/apps/web/data/models.ts`

### Phase 4: Validation and Summary
1. Validate generated TypeScript syntax
2. Check that all parent references exist
3. Verify Definition naming conventions
4. Generate summary report with:
   - Model architecture overview
   - List of modules and their definitions
   - Suggested next steps
5. Save validation results

## Workflow Diagram

```
User Input
  ├─ model_name
  ├─ hf_repo_id
  └─ optional parameters
        ↓
┌─────────────────────────────────────┐
│  Phase 1: Extract from HuggingFace │
├─────────────────────────────────────┤
│  - Download config.json             │
│  - Parse architecture               │
│  - Calculate dimensions             │
│  Output:                            │
│    ├─ model_config.json             │
│    └─ model_architecture.json       │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 2: Find SGLang Baseline     │
├─────────────────────────────────────┤
│  - Clone/update SGLang repo         │
│  - Search model implementation      │
│  - Extract kernel calls             │
│  Output:                            │
│    ├─ sglang_implementation.json    │
│    └─ kernel_calls.json             │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 3: Generate Definition      │
├─────────────────────────────────────┤
│  - Detect architecture pattern      │
│  - Build module hierarchy           │
│  - Map to Definitions               │
│  Output:                            │
│    └─ Updated models.ts             │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 4: Validation & Summary     │
├─────────────────────────────────────┤
│  - Validate TypeScript              │
│  - Check references                 │
│  - Generate report                  │
│  Output:                            │
│    └─ model_addition_summary.md     │
└─────────────────────────────────────┘
```

## Output Files

All outputs are saved to `{output_dir}/`:

### Intermediate Files
- `model_config.json`: Raw HuggingFace config
- `model_architecture.json`: Parsed architecture info
- `sglang_implementation.json`: SGLang implementation details (if not skipped)
- `kernel_calls.json`: Extracted kernel calls (if not skipped)
- `module_mapping.json`: Module to Definition mappings

### Final Outputs
- `web/apps/web/data/models.ts`: Updated with new model (IMPORTANT!)
- `{output_dir}/model_addition_summary.md`: Detailed report

### Summary Report Contents

The `model_addition_summary.md` includes:

```markdown
# Model Addition Summary: {model_name}

## Model Information
- **Model ID**: {model_name}
- **HuggingFace Repo**: {hf_repo_id}
- **Display Name**: {display_name}
- **Architecture Type**: {architecture_type}

## Architecture Overview
- **Layers**: {num_layers}
- **Hidden Size**: {hidden_size}
- **Attention Heads**: {num_heads}
- **KV Heads**: {num_kv_heads}
- **Intermediate Size**: {intermediate_size}
- **Attention Type**: GQA/MLA/MHA
- **MLP Type**: Dense/MoE

## Generated Modules

### Normalization Layers
- input_layernorm: rmsnorm_h{size}, fused_add_rmsnorm_h{size}
- post_attention_layernorm: fused_add_rmsnorm_h{size}
- ...

### Attention Layers
- qkv_proj: gemm_n_{dim}_k_{dim}
- attn: gqa_paged_decode_h{heads}_kv{kv}_d{dim}_ps1, ...
- o_proj: gemm_n_{dim}_k_{dim}

### MLP Layers
- gate_up_proj: gemm_n_{dim}_k_{dim}
- down_proj: gemm_n_{dim}_k_{dim}

## Validation Results
✓ TypeScript syntax valid
✓ All parent references exist
✓ No circular dependencies
✓ Definition naming conventions followed

## Next Steps
1. Review generated model in web/apps/web/data/models.ts
2. Start web UI: `cd web/apps/web && pnpm dev`
3. Verify model appears in UI
4. Create corresponding Definition JSON files if needed
5. Run benchmarks: `flashinfer-bench run --local ./data`

## Notes
- Review Definition mappings for accuracy
- Some custom architectures may need manual adjustment
- Ensure Definition JSONs exist in dataset
```

## Error Handling

The skill handles common errors gracefully:

### Network Issues
- **Error**: Cannot download from HuggingFace
- **Handling**: Retry with exponential backoff, provide offline fallback

### Missing SGLang Implementation
- **Error**: Model not found in SGLang
- **Handling**: Continue with HuggingFace config only, use generic pattern

### Invalid Configuration
- **Error**: Missing required fields in config.json
- **Handling**: Attempt to infer from model name, prompt for manual input

### TypeScript Generation Errors
- **Error**: Invalid TypeScript syntax
- **Handling**: Save raw output, provide manual fix suggestions

## Requirements

- Python packages:
  - `huggingface_hub`
  - `requests`
  - `json` (standard library)
- Git (for SGLang cloning)
- Node.js and pnpm (for TypeScript validation)
- Network access to HuggingFace Hub and GitHub

## Implementation Details

The skill is implemented as a Python script that:

1. **Validates inputs**:
   - Check model_name format (lowercase, hyphens)
   - Validate HuggingFace repo ID format
   - Create output directory

2. **Executes sub-skills**:
   - Call extract-model-from-hf with appropriate parameters
   - Call find-sglang-baseline (unless skipped)
   - Call generate-model-definition with combined data
   - Each sub-skill runs with error handling

3. **Aggregates results**:
   - Collect outputs from all phases
   - Combine into comprehensive summary
   - Perform final validation

4. **Reports progress**:
   - Print status updates for each phase
   - Show progress bars for long operations
   - Display summary upon completion

## Advanced Usage

### Batch Processing Multiple Models

```bash
# Create a script to add multiple models
for model in "kimi-k2:moonshot-ai/kimi-k2" "llama-3.3-70b:meta-llama/Llama-3.3-70B-Instruct"; do
  IFS=':' read -r name repo <<< "$model"
  /add-new-model --model-name "$name" --hf-repo-id "$repo"
done
```

### Custom Architecture Patterns

For models with unique architectures not matching standard patterns:

```bash
# Add model first
/add-new-model --model-name custom-arch --hf-repo-id org/custom-arch

# Then manually edit web/apps/web/data/models.ts
# Refer to model_architecture.json for guidance
```

### Integration with CI/CD

```yaml
# .github/workflows/add-model.yml
name: Add Model
on:
  workflow_dispatch:
    inputs:
      model_name:
        required: true
      hf_repo_id:
        required: true

jobs:
  add-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Add model
        run: |
          claude-code run add-new-model \
            --model-name ${{ inputs.model_name }} \
            --hf-repo-id ${{ inputs.hf_repo_id }}
      - name: Create PR
        uses: peter-evans/create-pull-request@v5
        with:
          title: "Add model: ${{ inputs.model_name }}"
          body-path: model_analysis_${{ inputs.model_name }}/model_addition_summary.md
```

## Troubleshooting

### Issue: Model architecture not recognized

**Solution**: The skill falls back to generic transformer structure. Review and manually adjust the generated modules.

### Issue: Definitions don't exist in dataset

**Solution**: The summary report lists all required definitions. Create Definition JSON files for missing ones, or use kernel_generator to generate implementations.

### Issue: SGLang model not found

**Solution**: Use `--skip-sglang true` to bypass SGLang search. The skill will use HuggingFace config only.

### Issue: TypeScript validation fails

**Solution**: Check syntax in generated models.ts. The skill saves intermediate output for manual fixing.

## Notes

- Always review generated definitions before committing
- Test in web UI before running benchmarks
- For production use, create corresponding Definition and Solution JSONs
- The skill aims for 90%+ automation; some manual refinement expected
- Report issues or suggest improvements via GitHub issues

## See Also

- [extract-model-from-hf](./extract-model-from-hf.md)
- [find-sglang-baseline](./find-sglang-baseline.md)
- [generate-model-definition](./generate-model-definition.md)
- [CLAUDE.md](../CLAUDE.md) - Complete guide to model addition
