---
name: discover-models
description: Discover candidate LLMs and produce a kernel inventory — required definitions, classified as existing/new and fi_supported/fi_missing — for onboarding. Use as Phase 1 of /onboard-model, or standalone to plan onboarding work.
---

# Discover Models

Identify target models and produce a per-kernel inventory:
- which definitions are needed,
- which already live in the HuggingFace dataset (`tmp/flashinfer-trace/definitions/`),
- which are new and supported by FlashInfer,
- which are new and missing from FlashInfer (so a kernel-request issue is needed).

Produces the `kernels` block of the onboard-model run manifest.

## Usage

```bash
# Auto-discover candidate models added to SGLang in the last 30 days
/discover-models --discover

# Plan a specific model
/discover-models --model-name qwen3-235b-a22b --hf-repo-id Qwen/Qwen3-235B-A22B

# Write the inventory to a manifest file (consumed by /onboard-model)
/discover-models --model-name kimi-k2 --manifest tmp/onboard_kimi-k2_20260427.json
```

## Parameters

- `--discover` (optional): Auto-discover candidates from SGLang day-0 additions and sgl-cookbook YAMLs.
- `--model-name` (optional): Specific model slug to plan (e.g. `qwen3-235b-a22b`).
- `--hf-repo-id` (optional): HuggingFace repo override (e.g. `Qwen/Qwen3-235B-A22B`). Inferred from `--model-name` if omitted.
- `--manifest` (optional): Path to an onboard-model run manifest. The skill writes the `model_slug`, `hf_repo_id`, `repo_shas`, and `kernels` array. If the file already exists, fields are merged; existing per-kernel statuses are preserved unless `--refresh` is set.
- `--refresh` (optional): Re-classify all kernels even if entries already exist in the manifest.

## Prerequisites

- `/clone-repos` has been run, so `tmp/sglang/`, `tmp/flashinfer/`, `tmp/sgl-cookbook/`, and `tmp/flashinfer-trace/` are present and current.
- `huggingface_hub` is installed and (for gated models) authenticated.

---

## Phase 1a: Discover candidate models

Run only when `--discover` is set.

**Day-0 SGLang additions** (highest priority — production-ready):

```bash
git -C tmp/sglang log --since="30 days ago" --name-status --diff-filter=A \
    -- "python/sglang/srt/models/*.py" | grep "^A" | awk '{print $2}'
```

Models with a brand-new `.py` under `python/sglang/srt/models/` in the last 30 days are
day-0 candidates. Parse the model class to derive a slug.

**sgl-cookbook new entries**:

```bash
git -C tmp/sgl-cookbook log --since="30 days ago" --name-status --diff-filter=A \
    -- "data/models/generated/v0.5.6/*.yaml" | grep "^A" | awk '{print $2}'
```

A new YAML signals a model with a recommended serving config.

**Filter already-tracked models**: read `docs/model_coverage.mdx` Summary table and skip any
candidate already listed.

## Phase 1b: Fetch model config from HuggingFace

For each candidate (or the specified `--model-name`):

```python
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id=hf_repo_id, filename="config.json")
with open(config_path) as f:
    config = json.load(f)
```

Key fields to extract: see `track-models` SKILL.md for the full `config.json → kernel param`
mapping table.

## Phase 1c: Determine required kernel definitions

Use the per-op-type formulas in `track-models` Phase 3a to compute the expected definition
names from the model config and the sgl-cookbook TP/EP values. Each formula yields a fully
qualified definition name like `gqa_paged_decode_h40_kv8_d128_ps1`.

## Phase 1d: Classify existing vs new

For each expected definition name, search the HuggingFace dataset clone (definitions live
only there after the trace-dataset refactor):

```bash
find tmp/flashinfer-trace/definitions/ -name "{definition_name}.json"
```

| Result | Classification |
|--------|---------------|
| File found | **existing** — no new definition needed |
| Not found | **new** — proceed to FlashInfer-availability classification |

## Phase 1e: Check FlashInfer kernel availability for new definitions

For each *new* definition, determine whether FlashInfer already implements the underlying
kernel.

| op_type | Check path in `tmp/flashinfer/` |
|---------|--------------------------------|
| `rmsnorm` | `flashinfer/norm.py` — grep for `rmsnorm` |
| `gqa_paged` | `flashinfer/decode.py`, `flashinfer/prefill.py` |
| `gqa_ragged` | `flashinfer/prefill.py` |
| `mla_paged` | `flashinfer/mla.py` |
| `dsa_paged` | `flashinfer/sparse.py` |
| `gdn` | `flashinfer/gdn.py` or `flashinfer/gdn/` |
| `moe` | `flashinfer/fused_moe/` — check the specific variant |
| `gemm` | always available via PyTorch |
| `sampling` | `flashinfer/sampling.py` |
| `mamba_ssu` | `flashinfer/mamba.py` — grep for `selective_state_update` |
| `rope` | `flashinfer/rope.py` — grep for `apply_rope_with_cos_sin_cache` |

Also check `tmp/flashinfer/tests/` for a corresponding test file — its presence is a strong
signal the kernel is implemented and tested.

A stronger signal that the kernel is **fully ready for the trace-dump path** (Path A in
[`extract-kernel-definitions`](../extract-kernel-definitions/SKILL.md)) is whether the
FlashInfer API carries an `@flashinfer_api(trace=...)` decorator (added by
[flashinfer-ai/flashinfer#2931](https://github.com/flashinfer-ai/flashinfer/pull/2931)).
Check with:

```bash
grep -rn "@flashinfer_api(trace=" tmp/flashinfer/flashinfer/ | grep -i "{module_or_api}"
```

If the API is decorated, Phase 2 can produce its Definition JSON automatically by running
a short SGLang inference pass with `FLASHINFER_TRACE_DUMP=1`. If FlashInfer has the kernel
but not the decorator, classification is still `fi_supported` but Phase 2 falls back to
manual extraction. Record the decorator-presence flag on the manifest entry as
`fi_trace_template` (`true`/`false`) so reviewers know which path to expect.

Classify each new definition:

- **fi_supported**: FlashInfer has the kernel → onboard-model Phase 2 (trace-dump if
  `fi_trace_template=true`, else manual extraction; see `extract-kernel-definitions`).
- **fi_missing**: FlashInfer does not have the kernel → onboard-model Phase 2 (manual
  extraction from SGLang + file kernel-request issue).

## Phase 1f: Check SGLang integration for fi_supported definitions

For each `fi_supported` definition, determine whether SGLang already routes through the
FlashInfer kernel. The result drives Phase 3 (workload collection).

```bash
# Use the fi_api tag from the definition (or the expected wrapper name) to grep:
grep -r "{flashinfer_api_name}" tmp/sglang/python/sglang/srt/ 2>/dev/null | grep -v __pycache__
```

Common mapping:

| fi_api | SGLang integration file | Search term |
|--------|------------------------|-------------|
| `flashinfer.mla.BatchMLAPagedAttentionWrapper` | `layers/attention/flashinfer_backend.py` | `BatchMLAPagedAttentionWrapper` |
| `flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` | `layers/attention/flashinfer_backend.py` | `BatchDecodeWithPagedKVCacheWrapper` |
| `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` | `layers/attention/flashinfer_backend.py` | `BatchPrefillWithPagedKVCacheWrapper` |
| `flashinfer.norm.rmsnorm` | `layers/layernorm.py` | `flashinfer.norm` |
| `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` | `layers/moe/fused_moe.py` | `trtllm_fp8_block_scale_moe` |
| `flashinfer.gdn.gated_delta_rule_decode` | `layers/attention/gdn_backend.py` | `gated_delta_rule_decode` |
| `flashinfer.mamba.selective_state_update` | `layers/mamba/mamba_mixer.py` | `selective_state_update` |

Classify:

- **sgl_integrated**: SGLang already calls this FlashInfer API → Phase 3 collects workloads directly.
- **sgl_missing**: SGLang does not yet wire this API → Phase 3 must submit an SGLang PR first.

For `fi_missing` definitions, SGLang integration is moot (no FlashInfer kernel to call) — set `sgl_status` to `n/a`.

## Phase 1g: Report

Print a classification table:

```
Model: Qwen3-235B-A22B
HF repo: Qwen/Qwen3-235B-A22B
Architecture: 94 layers, GQA + MoE

Kernel inventory:
  EXISTING (skip):
    ✅ rmsnorm_h7168
    ✅ moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  NEW — FlashInfer supported, SGLang integrated → ready for workload collection:
    🆕 gqa_paged_decode_h40_kv8_d128_ps1
    🆕 gqa_paged_decode_h40_kv8_d128_ps64
  NEW — FlashInfer supported, SGLang missing → submit SGLang PR first:
    🆕 dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
  NEW — FlashInfer MISSING → file kernel-request issue, skip workload collection:
    ❓ <new_op_type>_<params>
```

## Output: run-manifest contract

When `--manifest <path>` is set, write/update a JSON file with this shape (the same manifest
consumed by `/onboard-model` and `/submit-onboarding-prs`):

```json
{
  "model_slug": "qwen3-235b-a22b",
  "hf_repo_id": "Qwen/Qwen3-235B-A22B",
  "date": "2026-04-27",
  "repo_shas": {
    "sglang": "abc1234",
    "flashinfer": "def5678",
    "sgl_cookbook": "ghi9012",
    "flashinfer_trace": "jkl3456"
  },
  "kernels": [
    {
      "definition_name": "gqa_paged_decode_h40_kv8_d128_ps1",
      "op_type": "gqa_paged",
      "phase1_status": "new",
      "fi_status": "fi_supported",
      "fi_trace_template": true,
      "sgl_status": "sgl_integrated"
    },
    {
      "definition_name": "rmsnorm_h7168",
      "op_type": "rmsnorm",
      "phase1_status": "existing"
    },
    {
      "definition_name": "new_op_h512",
      "op_type": "new_op",
      "phase1_status": "new",
      "fi_status": "fi_missing",
      "sgl_status": "n/a"
    }
  ]
}
```

Existing entries written by later phases (`phase2_status`, `phase3_status`, `workload_entries`,
`fi_issue_url`, `phase4`) are preserved on update.

## See Also

- [onboard-model](../onboard-model/SKILL.md) — full pipeline that consumes this skill's output
- [track-models](../track-models/SKILL.md) — config-field and per-op-type formula reference
- [clone-repos](../clone-repos/SKILL.md) — must run first
- [submit-onboarding-prs](../submit-onboarding-prs/SKILL.md) — Phase 4 counterpart
