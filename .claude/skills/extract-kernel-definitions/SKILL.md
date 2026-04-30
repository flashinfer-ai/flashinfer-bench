---
name: extract-kernel-definitions
description: Generate Definition JSON files for the flashinfer-trace HuggingFace dataset by harvesting them from a short SGLang inference pass (FlashInfer's @flashinfer_api(trace=...) dumper) — or, as a fallback, by manually transcribing the schema from SGLang sources when FlashInfer doesn't yet have a trace template. Use when adding a new model, extracting GPU kernels (MLA, MoE, GQA, RMSNorm, GEMM, GDN, RoPE, sampling), or filling gaps in the dataset.
---

# Extract Kernel Definitions

Produce per-(op, shape) Definition JSONs and stage them in the HuggingFace dataset clone at
`tmp/flashinfer-trace/definitions/{op_type}/`. PR submission is **out of scope** here — see
[`submit-onboarding-prs`](../submit-onboarding-prs/SKILL.md) (Phase 4 of `/onboard-model`).

## Two paths

| Path | When to use | What you do |
|------|-------------|-------------|
| **A. Trace-dump (primary)** | Kernel is `fi_supported` per `/discover-models` — i.e. the FlashInfer API used by SGLang carries a `@flashinfer_api(trace=...)` template (see [coverage list](#flashinfer-trace-coverage)). | Run a short SGLang inference pass with `FLASHINFER_TRACE_DUMP=1`. The dumper writes one JSON per unique (op, shape) before the kernel runs (crash-safe, deduplicated). |
| **B. Manual extraction (fallback)** | Kernel is `fi_missing`, **or** the relevant FlashInfer API is not yet trace-instrumented. | Read the SGLang model source + sgl-cookbook serving config + HF model config; write the Definition JSON by hand using the [schema reference](#schema-reference). |

The trace-dump path is the default — it eliminates manual axis derivation and produces
JSONs that already carry `axes`, `inputs`, `outputs`, `tags` (`fi_api:*`,
`status:verified`), and a `reference` implementation.

> Background: the trace dumper was added in
> [flashinfer-ai/flashinfer#2931](https://github.com/flashinfer-ai/flashinfer/pull/2931).
> Schema and full env-var docs live at
> [`docs/fi_trace.rst`](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/fi_trace.rst)
> in the FlashInfer repo. SGLang harness reference:
> [`tests/trace/example_sglang.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/trace/example_sglang.py).

## Usage

```bash
# Path A — auto-dump every fi_supported definition for a model in one inference pass
/extract-kernel-definitions --model-name llama-3.2-3b --hf-repo-id meta-llama/Llama-3.2-3B-Instruct

# Path A — multi-config: one short run per (TP, EP) listed in sgl-cookbook
/extract-kernel-definitions --model-name qwen3-next --tp-list 2,4

# Path B — manual fallback for fi_missing kernels (or names that didn't appear in the dump)
/extract-kernel-definitions --model-name kimi-k2 --manual --op-types new_op_type
```

## Parameters

- `--model-name` (required): Model slug (e.g. `llama`, `deepseek-v3`, `qwen3-next`). Used to
  look up the SGLang model file and the sgl-cookbook YAML.
- `--hf-repo-id` (optional): HuggingFace repo override; inferred from `--model-name` if omitted.
- `--tp-list` (optional): Comma-separated TP values to run for; default reads
  sgl-cookbook YAML.
- `--ep-list` (optional): Comma-separated EP values for MoE models.
- `--manual` (optional): Force Path B (manual extraction) even for fi_supported ops.
- `--op-types` (optional): Comma-separated `op_type` filter when using `--manual` or for
  `--dry-run` reporting.
- `--dry-run` (optional): Report what would be dumped/written without running anything.
- `--skip-existing` (optional, default `true`): Skip any definition whose name already
  exists under `tmp/flashinfer-trace/definitions/`.

## Prerequisites

- `/clone-repos` has been run, so `tmp/sglang/`, `tmp/flashinfer/`, `tmp/sgl-cookbook/`,
  and `tmp/flashinfer-trace/` are present and current. The HF dataset clone at
  `tmp/flashinfer-trace/` is the only home for definitions — the in-repo
  `flashinfer_trace/` directory was removed in the trace-dataset refactor.
- For Path A: a working CUDA-enabled environment, GPU memory sufficient for the chosen
  model + TP, and `attention_backend="flashinfer"` available in the installed SGLang.
- For Path B: HuggingFace `config.json` access for the target model.

---

## Path A: trace-dump from a short SGLang pass

The dumper fires inside FlashInfer when both env vars are set **before** the FlashInfer
import. SGLang routes through `@flashinfer_api(trace=...)`-decorated APIs whenever
`attention_backend="flashinfer"` is selected, so a single short prefill+decode pass
exercises most ops at once.

### A1. Pick the serving config(s)

Open the sgl-cookbook YAML for the target model and list the unique TP/EP values — one
trace-dump pass per unique combination is enough to cover every shape variant.

```bash
ls tmp/sgl-cookbook/data/models/generated/v0.5.6/ | grep -i {model_name}
cat tmp/sgl-cookbook/data/models/generated/v0.5.6/{model_yaml}
```

If the model has no cookbook entry, default to TP=1 (single-GPU baseline) and skip EP.

### A2. Run the trace-dump pass

Use `tools/gpu-lock` so `CUDA_VISIBLE_DEVICES` is set correctly. The script below mirrors
[`tests/trace/example_sglang.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/trace/example_sglang.py)
in the FlashInfer repo — adapt the `model_path`, `tp_size`, and `attention_backend`:

```bash
DUMP_DIR=tmp/dumps/fi_trace_{model_slug}_tp{TP}_ep{EP}

tools/gpu-lock --gpus {TP} --exec-timeout 1800 -- python - <<EOF
import os, shutil
from pathlib import Path

# Must be set BEFORE flashinfer / sglang import.
os.environ["FLASHINFER_TRACE_DUMP"] = "1"
os.environ["FLASHINFER_TRACE_DUMP_DIR"] = "$DUMP_DIR"
os.environ.setdefault("SGLANG_SKIP_CUBIN_DOWNLOAD", "1")

dump = Path("$DUMP_DIR")
if dump.exists():
    shutil.rmtree(dump)

from sglang.srt.entrypoints.engine import Engine
engine = Engine(
    model_path="{hf_repo_id}",
    attention_backend="flashinfer",
    disable_cuda_graph=True,         # keep first call on the Python path
    mem_fraction_static=0.5,
    tp_size={TP},
    disable_radix_cache=True,
    log_level="warning",
)
engine.generate(
    ["The capital of France is"],
    {"temperature": 0.0, "max_new_tokens": 4, "top_k": 50, "top_p": 0.9},
)
engine.shutdown()
EOF
```

A few non-obvious requirements:

- **Set the env vars before import.** `FLASHINFER_TRACE_DUMP` and
  `FLASHINFER_TRACE_DUMP_DIR` are read at call time, but the `@flashinfer_api` decorator
  binding happens at import — set them in the shell or at the top of the entry script
  *before* any `import flashinfer` / `import sglang` runs.
- **Use `attention_backend="flashinfer"`.** Other SGLang backends bypass the FlashInfer
  APIs and produce no dumps.
- **Disable CUDA graphs (`disable_cuda_graph=True`)** for the trace pass. Cached graphs
  skip the Python path and therefore the dumper.
- **Page-size variants need separate runs.** SGLang's page size is fixed per server, so
  to capture both `_ps16` and `_ps64` shapes (for example) you must run twice with
  different `--page-size`. Enumerate the page sizes used by the target model.
- **MoE routing methods.** Each `routing_method_type` (Default, Renormalize, DeepSeekV3,
  Llama4, RenormalizeNaive, TopK) emits its own template; only the routing actually
  exercised by the model in your prompts will dump. For DeepSeek-V3 use a real DSv3 model
  to capture the `ds_routing` variant.
- **Quantized variants** (fp8/mxfp8/fp4 GEMM, fp8/fp4 block-scale MoE) require the model
  to actually use that quant config — load with the matching `--quantization` flag.

### A3. Dedupe and stage into the dataset

```bash
# 1. List what was dumped
ls "$DUMP_DIR"

# 2. For each {name}.json: sort it under the right op_type subdirectory.
#    The op_type field inside the JSON is the source of truth for the subfolder.
python - <<'EOF'
import json, shutil
from pathlib import Path
src = Path("$DUMP_DIR")
dst_root = Path("tmp/flashinfer-trace/definitions")
for p in src.glob("*.json"):
    op_type = json.loads(p.read_text())["op_type"]
    dst = dst_root / op_type / p.name
    if dst.exists():
        print(f"skip (exists): {dst.relative_to(dst_root)}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    print(f"added: {dst.relative_to(dst_root)}")
EOF
```

### A3b. Normalize the staged JSONs for `flashinfer-bench validate`

The trace dumper's output has three known mismatches with `flashinfer-bench`'s
`Definition` schema (all originate in the trace templates shipped with
[flashinfer-ai/flashinfer#2931](https://github.com/flashinfer-ai/flashinfer/pull/2931)):

1. `reference` declares the function as `def _<name>_reference(...)`, but
   `flashinfer-bench` requires a top-level `def run(...)`.
2. Plan-time index tensors (`kv_indptr`, `kv_indices`, `qo_indptr`) come back
   with `dtype: "unknown"` because the dumper inspects only `run()`'s kwargs,
   not the wrapper state set during `plan()`. The validator only accepts
   concrete dtypes from its enum.
3. In-place ops (e.g. `fused_add_rmsnorm`'s residual) declare the same name
   in both `inputs` and `outputs`. `flashinfer-bench` rejects overlapping
   I/O names; the dumper's `reference` function only returns the non-overlap
   outputs anyway, so it's safe to drop the duplicates from `outputs`.

Run this once per staging pass to make the JSONs validate:

```bash
python - <<'EOF'
import json, re
from pathlib import Path
INDEX_TENSOR_DTYPE = "int32"
KNOWN_INDEX_TENSORS = {
    "kv_indptr", "kv_indices", "qo_indptr",
    "paged_kv_indptr", "paged_kv_indices", "kv_last_page_len",
}
REF_RE = re.compile(r"^def\s+_[A-Za-z0-9_]+_reference\b", re.MULTILINE)
for p in Path("tmp/flashinfer-trace/definitions").rglob("*.json"):
    d = json.loads(p.read_text()); changed = False
    ref = d.get("reference", "")
    if ref and "def run(" not in ref:
        new_ref, n = REF_RE.subn("def run", ref, count=1)
        if n == 1: d["reference"], changed = new_ref, True
    for name, spec in d.get("inputs", {}).items():
        if isinstance(spec, dict) and spec.get("dtype") == "unknown" and name in KNOWN_INDEX_TENSORS:
            spec["dtype"], changed = INDEX_TENSOR_DTYPE, True
    overlap = set(d.get("inputs", {})) & set(d.get("outputs", {}))
    for name in overlap:
        d["outputs"].pop(name, None)
        changed = True
    if changed:
        p.write_text(json.dumps(d, indent=2) + "\n")
        print(f"normalized: {p}")
EOF
```

These three patches are mechanical — file a follow-up issue against
`flashinfer-ai/flashinfer` to emit `def run(...)`, resolve plan-time dtypes,
and drop in-place outputs (or rename them) inside the dumper itself, after
which A3b becomes a no-op.

### A3c. Validate

```bash
flashinfer-bench validate --dataset tmp/flashinfer-trace --disable-gpu
```

Newly staged definitions should report `[WARNING]` (missing descriptions on
axes/inputs/outputs are advisory) and not `[ERROR]`. Any `[ERROR]` on a
definition you just staged means A3b didn't normalize a new failure mode —
inspect the report (`tmp/flashinfer-trace/reports/report-*.txt`) and extend
the snippet.

That's it for Path A — once normalized, the staged JSONs carry `axes`,
`inputs`, `outputs`, `tags` (`fi_api:*`, `status:verified`), and a
`reference` implementation, so they're ready for the rest of the onboarding
pipeline (workloads → baseline → eval → Phase 4 PRs).

### Trade-off vs. tag enrichment

The dumper does **not** auto-emit `tp:N`, `ep:N`, `model:*`, or `quantization:*` tags —
those are workflow-level metadata, not kernel-shape metadata. After staging, append the
appropriate tags to the JSONs you just produced:

```bash
python - <<'EOF'
import json
from pathlib import Path
extra_tags = ["model:{model_slug}", "tp:{TP}"]   # add ep:{EP} for MoE
for p in Path("tmp/flashinfer-trace/definitions").rglob("*.json"):
    if p.stat().st_mtime < {dump_run_start_epoch}:
        continue
    j = json.loads(p.read_text())
    j["tags"] = sorted(set(j.get("tags", []) + extra_tags))
    p.write_text(json.dumps(j, indent=2) + "\n")
EOF
```

### FlashInfer trace coverage

Per
[`docs/fi_trace.rst`](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/fi_trace.rst),
the trace registry currently covers:

| FlashInfer module | API(s) | `op_type` |
|-------------------|--------|-----------|
| `flashinfer.norm` | `rmsnorm`, `fused_add_rmsnorm` (and gemma / quant variants) | `rmsnorm` |
| `flashinfer.sampling` | `top_k_sampling_from_probs`, `top_p_sampling_from_probs`, `top_k_top_p_sampling_from_probs`, `min_p_sampling_from_probs`, `chain_speculative_sampling` | `sampling` |
| `flashinfer.gemm` | `mm_bf16`, `mm_fp8`, `mm_mxfp8`, `mm_fp4` | `gemm_bf16` / `gemm_fp8` / `gemm_mxfp8` / `gemm_fp4` |
| `flashinfer.decode` | `BatchDecodeWithPagedKVCacheWrapper.run` | `gqa_paged` |
| `flashinfer.prefill` | `BatchPrefillWithPagedKVCacheWrapper.run`, `BatchPrefillWithRaggedKVCacheWrapper.run` | `gqa_paged` / `gqa_ragged` |
| `flashinfer.mla` | `BatchMLAPagedAttentionWrapper.run` | `mla_paged` |
| `flashinfer.gdn_decode` | `gated_delta_rule_decode`, `gated_delta_rule_mtp` | `gdn` |
| `flashinfer.gdn_prefill` | `chunk_gated_delta_rule` | `gdn` |
| `flashinfer.fused_moe` | `trtllm_fp8_block_scale_moe` × 6 routings, `trtllm_fp4_block_scale_moe` × 6 routings | `moe` |
| `flashinfer.rope` | `apply_rope_*` family | `rope` |
| `flashinfer.cascade` | `merge_state*` | `cascade` |
| `flashinfer.activation` | `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul` | `activation` |
| `flashinfer.quantization` | `fp4_quantize` | `quantize` |
| `flashinfer.page` | `append_paged_kv_cache` | `page` |

Anything outside this list falls through to Path B. To check up-to-date coverage:

```bash
grep -rn "@flashinfer_api(trace=" tmp/flashinfer/flashinfer/
```

---

## Path B: manual extraction (fallback)

Use this when:
- The kernel is `fi_missing` (no FlashInfer kernel exists yet — definition JSON will carry
  `status:unverified` plus a link to the kernel-request issue), or
- The kernel exists in FlashInfer but the API does not yet have a `@flashinfer_api(trace=...)`
  template (rare; check coverage list above).

### B1. Read the model + serving config

1. Locate the SGLang model file:
   ```bash
   ls tmp/sglang/python/sglang/srt/models/ | grep -i {model_name}
   ```
2. Find sgl-cookbook YAML and parse unique `tp` / `ep` values.
3. Pull `config.json` from HuggingFace (`hidden_size`, `num_attention_heads`,
   `num_key_value_heads`, `head_dim`, `intermediate_size`, `num_experts`, `num_experts_per_tok`,
   `vocab_size`, etc.). See `track-models` SKILL.md for the full field-to-axis mapping.

### B2. Compute kernel parameters per (TP, EP)

Apply the parallelism rules (TP/EP-affected kernels split head/expert counts; norm / GEMM
/ RoPE / sampling are parallelism-agnostic):

| op_type | TP affects | EP affects | Naming pattern |
|---------|-----------|-----------|---------------|
| `gqa_paged` | `q_heads/=TP`, `kv_heads/=TP` | — | `gqa_paged_{decode,prefill}_h{q}_kv{kv}_d{d}_ps{P}` |
| `gqa_ragged` | same as gqa_paged | — | `gqa_ragged_{prefill}_h{q}_kv{kv}_d{d}` |
| `mla_paged` | `q_heads/=TP` | — | `mla_paged_{decode,prefill}_h{q}_ckv{ckv}_kpe{kpe}_ps{P}` |
| `gdn` | `q_heads/=TP`, `v_heads/=TP` | — | `gdn_{decode,mtp,prefill}_qk{q}_v{v}_d{d}_k_last` |
| `mamba_ssu` | `nheads/=TP`, `ngroups/=TP` | — | `mamba_ssu_decode_h{n}_d{d}_s{s}_ng{g}` |
| `moe` | — | `num_experts/=EP` | `moe_{quant}_{routing}_topk{k}_e{local_e}_h{H}_i{I}` |
| `rmsnorm` | — | — | `rmsnorm_h{H}` / `fused_add_rmsnorm_h{H}` |
| `gemm` | — | — | `gemm_n{N}_k{K}` (or `gemm_{quant}_N{N}_K{K}`) |
| `rope` | — | — | `rope_with_cos_sin_cache_{neox,gptj}_style_d{d}_rd{rd}` |
| `sampling` | — | — | `{topk,topp,topk_topp}_sampling_from_probs_v{vocab}` |

Where `ckv = kv_lora_rank + qk_rope_head_dim` and `kpe = qk_rope_head_dim` for MLA.

### B3. Write Definition JSON

Hand-write the JSON under `tmp/flashinfer-trace/definitions/{op_type}/{name}.json`. Use
the canonical schema below. For `fi_missing` definitions add the status tag and the
issue back-pointer in the `description`.

For the `reference` field: write a plain-PyTorch `run(...)` implementation. Source it from
SGLang's vanilla forward (`tmp/sglang/python/sglang/srt/layers/...`) when FlashInfer
doesn't have it, otherwise mirror FlashInfer's own test harness. See `add-reference-tests`
for validation flow.

### B4. File a "missing trace template" issue against FlashInfer

Path B is friction we want to remove. Whenever you fall through to manual extraction
because **FlashInfer has the kernel but the API isn't decorated yet** (the
"decorator-gap" case — `fi_status=fi_supported` and `fi_trace_template=false` in the
manifest), file a follow-up issue in `flashinfer-ai/flashinfer` so the next onboarding
of the same op_type can use Path A. Skip this step for `fi_missing` kernels — those
already have a kernel-request issue from `/onboard-model` Phase 2a, and the trace
template will be added together with the kernel implementation.

```bash
gh issue create \
  --repo flashinfer-ai/flashinfer \
  --title "trace: add @flashinfer_api(trace=...) for {fi_api}" \
  --label "enhancement,flashinfer-trace" \
  --body "$(cat <<'EOF'
## Missing trace template

`{fi_api}` already has a working FlashInfer kernel but no `@flashinfer_api(trace=...)`
template, so flashinfer-bench cannot auto-dump its Definition JSON during a
`FLASHINFER_TRACE_DUMP=1` run. We had to fall back to manual extraction for
`{model_display_name}`.

### What we need

A `TraceTemplate` (see `flashinfer/trace/template.py` and the existing per-family
modules under `flashinfer/trace/templates/`) attached to the existing `@flashinfer_api`
decorator on `{fi_api}`. The template should declare the full set of axes / inputs /
outputs so `tests/trace/test_fi_trace_template_consistency.py` passes.

### Reference (manual extraction)

The Definition JSON we hand-wrote for this op while waiting for the trace template:
- `tmp/flashinfer-trace/definitions/{op_type}/{definition_name}.json` (will land in the
  next flashinfer-trace dataset PR — link from this comment once open).
- Used by: {model_display_name} ({hf_repo_id}).

### Acceptance

- [ ] `@flashinfer_api(trace=...)` attached to `{fi_api}`.
- [ ] Running `FLASHINFER_TRACE_DUMP=1 python tests/trace/example_sglang.py` (or the
      relevant model harness) emits a JSON for this op with the same `axes` /
      `inputs` / `outputs` / `name` we wrote by hand.
- [ ] `pytest tests/trace/ -v` passes.

### Related

- flashinfer-bench onboarding skill: `.claude/skills/extract-kernel-definitions/SKILL.md`
  Path B → step B4 (this issue is filed by that step).
EOF
)"
```

Record the issue URL on the manifest entry as `fi_trace_template_request_url` so
reviewers of the dataset PR can see the follow-up is in flight.

---

## Schema reference

This applies to both paths — it's the format the trace dumper produces (Path A) and the
format your hand-written JSON must match (Path B).

```json
{
  "name": "rmsnorm_h7168",
  "description": "Root Mean Square Normalization. Epsilon is fixed at 1e-6.",
  "op_type": "rmsnorm",
  "tags": [
    "fi_api:flashinfer.norm.rmsnorm",
    "status:verified",
    "model:{model_slug}",
    "tp:{N}"
  ],
  "axes": {
    "batch_size":  {"type": "var"},
    "hidden_size": {"type": "const", "value": 7168}
  },
  "constraints": ["..."],
  "inputs": {
    "hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
    "weight":        {"shape": ["hidden_size"],               "dtype": "bfloat16"}
  },
  "outputs": {
    "output": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}
  },
  "reference": "import torch\n\ndef run(...):\n    ..."
}
```

Field rules:

- **`name`** — Path A: auto-generated by the trace dumper from `op_type` / `name_prefix` +
  const-axis values. Path B: assemble per the [naming patterns](#b2-compute-kernel-parameters-per-tp-ep).
- **`op_type`** — selects the subdirectory under `definitions/` (`rmsnorm`, `gqa_paged`,
  `mla_paged`, `gdn`, `moe`, `gemm`, `gemm_fp8`, `sampling`, `rope`, …).
- **`tags`** — always include `fi_api:<qualified.name>` (e.g. `fi_api:flashinfer.norm.rmsnorm`)
  and `status:verified` (use `status:unverified` for fi_missing). Add `model:*`, `tp:N`,
  `ep:N`, `quantization:*` as applicable. Path A emits the first two automatically; the
  rest are workflow metadata you append after staging.
- **`axes`** — `var` axes vary at runtime (batch, sequence length, num_pages); `const` axes
  are model constants and carry a `"value"`. Const-axis values plus `name_prefix` produce
  the file name.
- **`constraints`** (optional) — string expressions like `"len_indptr == batch_size + 1"`,
  evaluated against axis values when validating workloads.
- **`inputs` / `outputs`** — each entry has `shape` (list of axis names) and `dtype`.
  Optional inputs: `"optional": true`. Output dtype may be inherited from an input via
  `"dtype_from": "{input_name}"` in trace templates (the dumper resolves it before
  writing).
- **`reference`** — pure-PyTorch `run()` for correctness checking. Required for
  `status:verified`. Path A emits this when the trace template includes one; Path B writes
  it by hand.

Examples of fully populated definitions live in
[`tests/trace/fi_trace_out/`](https://github.com/flashinfer-ai/flashinfer/tree/main/tests/trace/fi_trace_out)
in the FlashInfer repo — read these as canonical templates rather than re-deriving the
schema by hand.

---

## After staging

1. Validate: `flashinfer-bench validate --dataset tmp/flashinfer-trace --disable-gpu`.
2. Add reference tests for any newly staged definitions:
   `/add-reference-tests --definition-name {name}` (or `--op-type {op_type}`).
3. Move on to workload collection: `/collect-workloads --definition-names {names}`.
   Tip: `/collect-workloads` can also dump definitions in the same SGLang run by setting
   the trace env vars — useful for picking up shapes you missed in step A2.
4. PR submission is handled separately by `/submit-onboarding-prs` (Phase 4 of
   `/onboard-model`). Do **not** add definition JSONs to a `flashinfer_trace/...` path
   inside `flashinfer-bench` — that directory was removed in the refactor.

---

## Error handling

- **No JSONs appeared in the dump dir.** Either the env vars were set after the FlashInfer
  import, the SGLang attention backend isn't `flashinfer`, CUDA graphs were enabled, or the
  inference path didn't reach a decorated API. Re-check the env-var ordering, ensure
  `attention_backend="flashinfer"` and `disable_cuda_graph=True`, and add
  `print(flashinfer.norm.rmsnorm.fi_trace.__doc__)` to confirm the decorator is bound.
- **Names collide with existing definitions.** Path A is content-deterministic — if a
  staged file with the same name already exists and differs, the dump captured a different
  shape under the same const-axis values. Compare the JSONs; the existing one usually wins
  unless the new shape is the intended target (then update tags / file an issue rather
  than overwriting silently).
- **MoE routing variants didn't all dump.** Each `routing_method_type` is its own
  template; only the routings the model actually invokes will fire. Run a model with the
  required routing (e.g. real DeepSeek-V3 for `ds_routing`).
- **GPU OOM.** Reduce `mem_fraction_static`, increase `tp_size`, or use a smaller variant
  of the model — the trace pass needs only a couple of generated tokens.

## See also

- [discover-models](../discover-models/SKILL.md) — Phase 1 classifier; tells you which
  kernels are `fi_supported` (Path A) vs `fi_missing` (Path B).
- [add-reference-tests](../add-reference-tests/SKILL.md) — pytest validation against
  FlashInfer / SGLang ground truth.
- [collect-workloads](../collect-workloads/SKILL.md) — runs another SGLang pass and can
  dump definitions in the same run.
- [submit-onboarding-prs](../submit-onboarding-prs/SKILL.md) — Phase 4 PR flow.
- FlashInfer trace docs:
  [`docs/fi_trace.rst`](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/fi_trace.rst).
- Reference SGLang harness:
  [`tests/trace/example_sglang.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/trace/example_sglang.py).
