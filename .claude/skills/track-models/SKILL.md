---
name: track-models
description: Track popular/new open-source LLMs and update docs/model_coverage.mdx with their kernel support status. Use when discovering new models to add to the coverage tracker, checking if a specific model is covered, or refreshing model coverage documentation.
---

# Track Models

Discover popular or newly released open-source LLMs, determine their kernel coverage in FlashInfer-Bench, and update `docs/model_coverage.mdx` with summary and detailed per-kernel status tables.

## Usage

```bash
# Auto-discover new popular models not yet tracked
/track-models --discover

# Add a specific model by name
/track-models --model-name mistral-7b
/track-models --model-name gemma-3-27b --hf-repo-id google/gemma-3-27b-it

# Refresh coverage status for all already-tracked models
/track-models --refresh-status

# Do everything: discover new models + refresh existing ones
/track-models --discover --refresh-status
```

## Parameters

- `--discover` (optional): Auto-discover new popular models not yet in `model_coverage.mdx`
- `--model-name` (optional): Specific model to add or refresh (e.g., "mistral-7b", "gemma-3-27b")
- `--hf-repo-id` (optional): Override the HuggingFace repo ID (e.g., "google/gemma-3-27b-it")
- `--refresh-status` (optional): Re-check definition files for all already-tracked models and update ✅/❌ status

## Prerequisites

- `docs/model_coverage.mdx` must exist (it does — managed by this skill)
- `tmp/flashinfer-trace/definitions/` must exist with current definition JSON files
  (this is the HuggingFace dataset clone — populated by `/clone-repos`; the in-repo
  `flashinfer_trace/` directory was removed in the trace-dataset refactor)
- Run `/clone-repos` first if you need SGLang or sgl-cookbook configs for a model that isn't in the existing patterns, or to ensure the trace dataset clone is current

---

## What This Skill Does

### Phase 1: Model Discovery

When `--discover` is set, find popular open-source LLMs not yet tracked in `model_coverage.mdx`.

#### Discovery Sources (in order)

1. **SGLang day-0 additions** (highest priority — newly supported models):
   ```bash
   # Models added to SGLang in the last 30 days (new .py files in models dir)
   git -C tmp/sglang log --since="30 days ago" --name-status --diff-filter=A \
       -- "python/sglang/srt/models/*.py" | grep "^A" | awk '{print $2}'
   ```
   These are models with brand-new SGLang support — they are the most likely to have novel
   kernels not yet tracked. Parse each new filename to derive the model slug.

2. **SGLang supported models list** (full catalog):
   ```bash
   ls tmp/sglang/python/sglang/srt/models/
   ```
   SGLang models are a curated list of production-quality LLMs. Every model in SGLang is a candidate.

3. **sgl-cookbook** (models with recommended serving configs):
   ```bash
   ls tmp/sgl-cookbook/data/models/generated/v0.5.6/
   ```
   Models with a YAML config are actively deployed — highest priority.

4. **Inference API provider model lists** (strong popularity signal):
   A model served by 2+ commercial inference APIs is definitively production-critical.
   Fetch or browse the following provider pages and extract the open-weight model list:

   | Provider | URL | Notes |
   |----------|-----|-------|
   | Together AI | https://www.together.ai/pricing | Full model catalog with pricing tiers |
   | Fireworks AI | https://fireworks.ai/pricing | Open-source models section |
   | Groq | https://groq.com/pricing | Smaller curated list, very high-traffic models |
   | OpenRouter | https://openrouter.ai/models | Largest aggregator; filter by `free` or sort by throughput |
   | DeepInfra | https://deepinfra.com/models | Broad open-source catalog |
   | Hyperbolic | https://app.hyperbolic.xyz/models | GPU-native provider |
   | Cerebras | https://inference.cerebras.ai | Wafer-scale inference, limited but popular subset |

   **Heuristic**: Any model listed by **≥ 2 providers** should be added to coverage tracking.
   Models on **≥ 3 providers** are the highest priority (most widely deployed in production).

   Known high-priority models from provider crawls (update this list over time):
   - **Llama 4 Scout** (17Bx16E) — Together, Groq, sgl-cookbook
   - **Llama 4 Maverick** (17Bx128E MoE) — Together, Groq
   - **Llama 3.1 405B** — Together, Fireworks
   - **gpt-oss 120B / gpt-oss 20B** — Together, Fireworks, Groq (OpenAI open-source)
   - **GLM-4.5 / GLM-4.6 / GLM-5** — Together, Fireworks; sgl-cookbook has `glm46.yaml`
   - **MiniMax M2 / M2.5** — Together, Fireworks
   - **Kimi K2 / K2.5** — Together, Fireworks, Groq
   - **DeepSeek-V3.1 / R1-0528** — Together, Fireworks (updated DeepSeek variants)
   - **Qwen3 235B A22B** — Together, Fireworks, Groq
   - **Mistral Small 3 (24B)** — Together; sgl-cookbook has `mistral.yaml`

4. **HuggingFace website** (primary web source — fetch directly):
   Use `WebFetch` to retrieve the HuggingFace trending models page and extract newly released
   open-weight LLMs. This gives the most up-to-date signal for models gaining traction before
   they appear on inference provider lists.

   ```
   # Fetch trending models (text/LLM category)
   WebFetch: https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
   ```

   Also check the HuggingFace "new" filter for models released in the last 7 days:

   ```
   WebFetch: https://huggingface.co/models?pipeline_tag=text-generation&sort=created
   ```

   Focus on models that are:
   - Open-weight (not API-only or gated without public access)
   - From established labs (Meta, Google, Mistral, Alibaba/Qwen, DeepSeek, Moonshot, etc.)
   - Have ≥ 1K downloads/week or trending in the top 50
   - Transformer-based decoder (not vision-only or encoder-only)

   For each candidate, fetch its model page to confirm the architecture:
   ```
   WebFetch: https://huggingface.co/{org}/{model}/blob/main/config.json
   ```

5. **HuggingFace trending** (supplemental web search):
   Search for "huggingface trending LLM open source {current_year}" to find newly popular models.
   Focus on models that are:
   - Open-weight (not API-only)
   - LLM/VLM with transformer-based decoder
   - Have ≥ 1K downloads/week or are from a major lab (Meta, Google, Mistral, Alibaba, etc.)

#### Filtering Already-Tracked Models

Read the current `docs/model_coverage.mdx` and extract all model names from the `## Summary` table. Skip any model whose name is already listed.

#### Architecture Priority

Prefer to discover models that use kernel types already supported by FlashInfer-Bench:
- GQA attention (most LLMs)
- MLA attention (DeepSeek family)
- GDN linear attention (Qwen3-Next family)
- MoE FFN (Mixtral, Qwen-MoE, DeepSeek, etc.)
- Mamba/SSM (NemotronH, GraniteMoe-Hybrid)
- Dense FFN (Llama, Gemma, Mistral)

---

### Phase 2: Architecture Extraction

For each new model, extract architecture details. Use the same approach as the `extract-kernel-definitions` skill.

#### Step 1: Get config.json from HuggingFace

```python
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id="meta-llama/Llama-3.1-8B", filename="config.json")
with open(config_path) as f:
    config = json.load(f)
```

Key fields to extract:

| Field | Used For |
|-------|---------|
| `hidden_size` | RMSNorm definition names, GEMM K dim |
| `num_hidden_layers` | Layer count |
| `num_attention_heads` | GQA/MLA q_heads |
| `num_key_value_heads` | GQA kv_heads |
| `head_dim` or `hidden_size / num_attention_heads` | Attention head dim |
| `intermediate_size` | GEMM N dim for MLP |
| `vocab_size` | Sampling definition names |
| `architectures` | Model class name → infer op types |
| `num_experts` / `num_local_experts` | MoE expert count |
| `num_experts_per_tok` | MoE topk |

#### Step 2: Determine attention type from architecture

Use the `architectures` field in config.json or the SGLang model class:

| Architecture String | Attention Type | Notes |
|--------------------|---------------|-------|
| `LlamaForCausalLM`, `MistralForCausalLM`, `GemmaForCausalLM`, `Qwen2ForCausalLM` | GQA | Standard |
| `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM` | MLA | Has `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim` |
| `Qwen3NextForCausalLM` | GDN + GQA hybrid | Has `gdn_*` config keys |
| `NemotronHForCausalLM` | GQA + Mamba2 hybrid | Has `ssm_*` config keys |
| `MixtralForCausalLM` | GQA + MoE | Has `num_local_experts` |

#### Step 3: Find sgl-cookbook TP/EP config

```bash
find tmp/sgl-cookbook/data/models/generated/v0.5.6/ -name "*.yaml" | xargs grep -l "{model_keyword}"
```

Parse the YAML to extract all unique `tp` and `ep` values. Use these to compute per-TP head counts.

If no sgl-cookbook config exists, use TP=1 (single GPU baseline).

---

### Phase 3: Map Kernels to Definitions

For each model, compute the full list of expected definition names, then check which ones exist in `tmp/flashinfer-trace/definitions/` (the HF dataset clone — the only location where definition JSONs live).

#### 3a: Compute expected definitions

Follow these rules (same as CLAUDE.md):

**RMSNorm** (not TP-dependent):
- `rmsnorm_h{hidden_size}`
- `fused_add_rmsnorm_h{hidden_size}`
- If MLA: also `rmsnorm_h{q_lora_rank}` and `rmsnorm_h{kv_lora_rank}`

**GEMM** (not TP-dependent):
- `gemm_n{intermediate_size}_k{hidden_size}` (gate/up proj)
- `gemm_n{hidden_size}_k{intermediate_size}` (down proj)
- `gemm_n{hidden_size}_k{hidden_size}` (o_proj, if square)
- For MLA: `gemm_n{q_lora_rank * num_heads}_k{hidden_size}` etc. (check SGLang impl)

**GQA** (TP-dependent, per TP value):
- `gqa_paged_prefill_causal_h{q//TP}_kv{kv//TP}_d{head_dim}_ps1`
- `gqa_paged_prefill_causal_h{q//TP}_kv{kv//TP}_d{head_dim}_ps64`
- `gqa_paged_decode_h{q//TP}_kv{kv//TP}_d{head_dim}_ps1`
- `gqa_paged_decode_h{q//TP}_kv{kv//TP}_d{head_dim}_ps64`
- `gqa_ragged_prefill_causal_h{q//TP}_kv{kv//TP}_d{head_dim}`

**MLA** (TP-dependent, per TP value):
- `mla_paged_prefill_causal_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_ps1`
- `mla_paged_prefill_causal_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_ps64`
- `mla_paged_decode_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_ps1`
- `mla_paged_decode_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_ps64`

Where `ckv_dim = kv_lora_rank + qk_rope_head_dim` and `kpe_dim = qk_rope_head_dim`.

**DSA** (for DeepSeek V3.2-style, TP-dependent):
- `dsa_topk_indexer_fp8_h{num_heads//TP}_d{head_dim}_topk{topk}_ps64`
- `dsa_sparse_attention_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_topk{topk}_ps1`
- `dsa_sparse_attention_h{num_heads//TP}_ckv{ckv_dim}_kpe{kpe_dim}_topk{topk}_ps64`

**GDN** (for Qwen3-Next-style, TP-dependent, per TP value):
- `gdn_prefill_qk{q_heads//TP}_v{v_heads//TP}_d{head_dim}_k_last`
- `gdn_decode_qk{q_heads//TP}_v{v_heads//TP}_d{head_dim}_k_last`
- `gdn_mtp_qk{q_heads//TP}_v{v_heads//TP}_d{head_dim}_k_last`

**Mamba2 SSU** (for NemotronH-style, TP-dependent):
- `mamba_ssu_decode_h{nheads//TP}_d{head_dim}_s{dstate}_ng{ngroups//TP}`
- Constraints: `head_dim ∈ [64,128,256]`, `dstate ∈ [64,128,256]`, `nheads/ngroups ∈ [1,8,16]`

**MoE** (EP-dependent):
- `moe_fp8_block_scale_ds_routing_topk{topk}_ng{n_group}_kg{topk_group}_e{num_experts//EP}_h{hidden_size}_i{intermediate_size}`

**RoPE** (not TP-dependent):
- `rope_with_cos_sin_cache_neox_style_d{head_dim}_rd{rotary_dim}`
- `rope_with_cos_sin_cache_gptj_style_d{head_dim}_rd{rotary_dim}`

**Sampling** (not TP-dependent):
- `top_k_sampling_from_probs_v{vocab_size}`
- `top_k_top_p_sampling_from_probs_v{vocab_size}`
- `top_p_sampling_from_probs_v{vocab_size}`

#### 3b: Check definition existence

For each expected definition name, check:
```bash
find tmp/flashinfer-trace/definitions/ -name "{definition_name}.json"
```

Assign status:
- **✅** — JSON file exists in `tmp/flashinfer-trace/definitions/`
- **❌** — Name computed from model config, but no JSON file found (missing, needs to be created)
- **—** — Module exists in the model architecture but definition is not computed/mapped (unmapped)

#### 3c: Compute coverage summary

```
coverage = (count of ✅) / (count of ✅ + count of ❌)
```

- **✅ Fully covered** — all expected definitions exist (0 ❌)
- **🟡 Partial** — some definitions exist, some missing
- **❌ Not covered** — no definitions exist (all ❌ or —)

---

### Phase 4: Update `docs/model_coverage.mdx`

#### 4a: Add row to Summary table

Find the `## Summary` section and add (or update) the model row:

```markdown
| {Model Display Name} | {Architecture description} | {Coverage emoji + label} |
```

Architecture description examples:
- `GQA + Dense` (standard transformer)
- `GQA + MoE` (standard attention + mixture of experts)
- `MLA + Dense/MoE` (multi-head latent attention)
- `GDN + GQA + MoE` (hybrid linear + standard attention)
- `GQA + Mamba2` (hybrid attention + SSM)

#### 4b: Add detailed section

Append (or update) a new `## {Model Display Name}` section before the closing `---`:

```markdown
## {Model Display Name}

**Architecture**: {num_layers} decoder layers, {attention_type} attention, {ffn_type} FFN{extra details}

{If multiple TP configs, add note}: Standard serving configuration: **TP={N}** (or TP={N} / TP={M}).

| Definition | Op Type | Status |
|-----------|---------|:------:|
| `rmsnorm_h{hidden_size}` | rmsnorm | ✅ |
| `fused_add_rmsnorm_h{hidden_size}` | rmsnorm | ✅ |
| ... | ... | ... |
| MoE gate / topk / experts | moe | — |

**Coverage**: {N} / {M} definitions present. {Optional: note about missing definitions.}

---
```

#### 4c: Preserve existing sections

**Do NOT overwrite existing model sections** unless `--refresh-status` is set. When refreshing:
- Re-check all ✅/❌ in the existing table
- Update the coverage count line
- Update the summary table row

---

## Documentation Standards

### Model Display Names

Use consistent naming matching the summary table:
- `DeepSeek V3 / R1` (joint entry for same-architecture variants)
- `Llama 3.1 8B` (include size)
- `Qwen3 30B A3B` (include size + variant)
- `Mistral 7B v0.3`
- `Gemma 3 27B`

### Architecture Descriptions

Keep brief (5–8 words), following existing patterns:
- `{layer_count} decoder layers, {attention} attention, {ffn} FFN`
- Add `hybrid` prefix for mixed architectures (e.g., `48 layers, hybrid GDN+GQA attention, MoE FFN`)

### TP/EP Notes

When multiple TP configs exist, show all in the table using separate rows with TP label in Op Type column:

```markdown
| `gdn_prefill_qk16_v32_d128_k_last` | gdn TP=1 | ❌ |
| `gdn_prefill_qk8_v16_d128_k_last` | gdn TP=2 | ✅ |
| `gdn_prefill_qk4_v8_d128_k_last` | gdn TP=4 | ✅ |
```

For MoE with EP:
```markdown
| `moe_..._e256_...` | moe EP=1 | ❌ |
| `moe_..._e32_...` | moe EP=8 | ✅ |
```

### Unmapped Modules

For modules that exist in the architecture but don't have definitions mapped, use `—`:

```markdown
| MoE gate / topk / experts | moe | — |
```

---

## Output

After running, report:
1. **Models added**: List of new models added to `model_coverage.mdx`
2. **Models refreshed**: List of models whose coverage status was updated
3. **Missing definitions**: List of ❌ definitions that need to be created (suggest running `/extract-kernel-definitions`)
4. **Summary delta**: What changed in the summary table (newly fully covered, newly partial, etc.)

---

## Common Model Architectures Reference

### GQA Standard (Llama / Mistral / Gemma / Qwen2.5)

Expected definitions:
- 2× RMSNorm
- 4× GEMM (qkv_proj, o_proj, gate_up, down)
- 5× GQA (paged decode ps1, paged decode ps64, paged prefill ps1, paged prefill ps64, ragged prefill)
- 3× Sampling (top_k, top_k_top_p, top_p)

### GQA + MoE (Mixtral / Qwen2-MoE / Qwen3-MoE)

Expected definitions: same as GQA Standard, plus:
- 1× MoE (per EP config)

### MLA + Dense/MoE (DeepSeek V2/V3/R1)

Expected definitions:
- 4× RMSNorm (hidden, q_lora, kv_lora, plus sometimes kv_lora variants)
- N× GEMM (MLA has more projections: qkv_a, q_b, kv_b, o_proj, gate/up/down)
- 4× MLA (paged decode ps1, paged decode ps64, paged prefill ps1, paged prefill ps64)
- 1× MLA ragged (prefill)
- 1× MoE (per EP config)
- 3× Sampling

### GDN + GQA + MoE (Qwen3-Next)

Expected definitions:
- 2× RMSNorm
- 3× GDN per TP config (prefill, decode, mtp)
- 5× GQA per TP config
- 1× MoE
- (Sampling: if tracked)

### GQA + Mamba2 (NemotronH / GraniteMoe-Hybrid)

Expected definitions:
- 2× RMSNorm
- N× GEMM
- 5× GQA
- 1× Mamba SSU per TP config
- 3× Sampling

> **Note**: Check FlashInfer Mamba SSU constraints before adding: `nheads/ngroups ∈ [1, 8, 16]`. Models that violate this (e.g., FalconH1 ngroups=1 nheads=128) cannot use FlashInfer SSU and should be marked ❌ for Mamba kernel with a note.

---

## Integration with Other Skills

```bash
# Discover new models and see what's missing
/track-models --discover

# Create missing kernel definitions for a new model
/extract-kernel-definitions --model-name mistral_7b

# Add reference tests for new definitions
/add-reference-tests --op-type gqa_paged

# Full workflow for a brand new model
/clone-repos
/track-models --model-name gemma-3-27b --hf-repo-id google/gemma-3-27b-it
/extract-kernel-definitions --model-name gemma3
/add-reference-tests --op-type gqa_paged

# Or use the end-to-end pipeline (discovery + definition + workload + PR submission)
/onboard-model --discover
/onboard-model --model-name gemma-3-27b --hf-repo-id google/gemma-3-27b-it
```

This skill is also called by `onboard-model` (Phase 1 and Phase 4b) to update
`docs/model_coverage.mdx` as part of the automated pipeline.

## Error Handling

### HuggingFace config not accessible
- **Cause**: Private model or network error
- **Handling**: Note the model as `❌ Not covered` with a "config unavailable" note in the detailed section. Proceed with other models.

### Architecture not recognized
- **Cause**: Novel architecture not in the known patterns
- **Handling**: Add model to summary as `❌ Not covered`, add a detailed section with `— (architecture not yet mapped)` for all kernel rows and a note explaining the unknown pattern.

### Definition file naming ambiguity
- **Cause**: Computed name doesn't match any definition (off-by-one in dims, different naming convention)
- **Handling**: Mark as ❌ and list the computed name in the detailed section. Cross-check with actual files in `tmp/flashinfer-trace/definitions/` before marking as missing.

### Model not in SGLang
- **Cause**: Model isn't implemented in SGLang yet
- **Handling**: Still add to coverage doc using HuggingFace config.json. Note "SGLang implementation pending" in the architecture description.

## Maintaining This Document

Update this file when:
- New op_types are added to FlashInfer-Bench (add to Phase 3 mapping and Common Architectures)
- Definition naming conventions change
- New major model families emerge (add to Common Architectures)

## See Also

- [extract-kernel-definitions](../extract-kernel-definitions/SKILL.md) — create missing definition JSON files
- [clone-repos](../clone-repos/SKILL.md) — clone SGLang, sgl-cookbook
- [add-reference-tests](../add-reference-tests/SKILL.md) — add tests for new definitions
- [docs/model_coverage.mdx](../../../docs/model_coverage.mdx) — the file this skill maintains
