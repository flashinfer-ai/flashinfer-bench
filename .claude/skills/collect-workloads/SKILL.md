---
name: collect-workloads
description: Auto-collect workloads from SGLang inference runs using FlashInfer logging API. Dumps tensors, sanitizes them according to kernel definitions, and submits PR to flashinfer-trace workload repo.
---

# Collect Workloads

Collect real-world workloads by running SGLang inference with FlashInfer Level 10 logging, then sanitize and submit to the flashinfer-ai/flashinfer-trace HuggingFace dataset.

**No code changes to SGLang or FlashInfer are required.** Collection works entirely through FlashInfer's built-in logging API.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/collect_workloads.py` | Primary entry point: runs SGLang inference + sanitizes dumps |
| `scripts/sanitize_dumps.py` | Converts FlashInfer Level 10 dump dirs → JSONL + safetensors; called automatically by `collect_workloads.py`, or run manually to re-sanitize |

```bash
python scripts/collect_workloads.py sglang \
  --model-path /path/to/model \
  --definitions gqa_paged_decode_h32_kv8_d128_ps1 \
  --flashinfer-trace-dir tmp/flashinfer-trace \
  --replace
```

**Auto-detection from definition tags:**
- `tp:N` → sets `--tp N` (use `CUDA_VISIBLE_DEVICES=0,0` to simulate TP=2 on 1 GPU)
- `page_size` const axis → sets `--page-size N`

## Workflow

### Phase 0: Install Latest Packages

```bash
git -C tmp/flashinfer pull && git -C tmp/sglang pull
conda run -n flashinfer_bench pip install -e tmp/flashinfer --no-build-isolation
conda run -n flashinfer_bench pip install -e "tmp/sglang/python[all]"
```

### Phase 1: Resolve Target Definitions

- `--definitions <name> [name ...]`: specific definitions by name
- `--op-type <type>`: all definitions under `definitions/{op_type}/`
- `--all`: all definitions in the repo

### Phase 2: FlashInfer Logging Configuration

Parses `fi_api:<dotted.api.name>` tags from each definition to build `FLASHINFER_DUMP_INCLUDE`:
- Wrapper class APIs (e.g. `BatchDecodeWithPagedKVCacheWrapper`) → include `.run`, and `.plan` if the definition has `int32`/`int64` inputs
- **`BatchPrefillWithRaggedKVCacheWrapper`**: SGLang calls `.forward()`/`.forward_return_lse()` (not `.run()`) — those are automatically added to `FLASHINFER_DUMP_INCLUDE` for Ragged wrappers
- Plain function APIs (e.g. `rmsnorm`) → include by function name

Key env vars set automatically:
```bash
FLASHINFER_LOGLEVEL=10
FLASHINFER_DUMP_DIR=./workload_dumps_<timestamp>
FLASHINFER_DUMP_SAFETENSORS=1
FLASHINFER_DUMP_INCLUDE=<fi_api patterns>   # only log matching API calls
FLASHINFER_DUMP_EXCLUDE=*.__init__
FLASHINFER_DUMP_MAX_COUNT=50000
FLASHINFER_DUMP_MAX_SIZE_GB=30
```

### Phase 3: SGLang Inference

**Inference source**: real ShareGPT prompts (from `--dataset` path or downloaded from HuggingFace `anon8231489123/ShareGPT_Vicuna_unfiltered`). Falls back to synthetic prompts only if ShareGPT is unavailable.

**Batch sizes**: `[8, 32, 64, 128]` — powers of 2 matching SGLang CUDA graph capture points, run 4 rounds each with fresh ShareGPT slices for natural KV-length diversity.

**Three execution modes** (chosen automatically based on definition type):

| Mode | When | How |
|------|------|-----|
| SGLang offline Engine | Decode-only definitions | `engine.generate()` with exact batch size per call — guarantees decode sees `B` concurrent sequences |
| SGLang HTTP server (paged) | Paged-prefill definitions | Launches server with `--enable-deterministic-inference` to force `use_ragged=False`, sends prefix-sharing requests via `/v1/chat/completions` |
| SGLang HTTP server (ragged) | Ragged-prefill definitions (`BatchPrefillWithRaggedKVCacheWrapper`) | Launches server with `--disable-piecewise-cuda-graph` (no `--enable-deterministic-inference`), sends requests with `max_tokens=1` |

**Critical ragged prefill flags**: `--disable-cuda-graph` alone is insufficient. SGLang always captures a piecewise CUDA graph for prefill; during capture `is_in_piecewise_cuda_graph()=True` forces `use_ragged=False`, so the captured graph only uses `BatchPrefillWithPagedKVCacheWrapper`. Adding `--disable-piecewise-cuda-graph` prevents the capture, ensuring every prefill executes eagerly through `BatchPrefillWithRaggedKVCacheWrapper`. Do **not** add `--enable-deterministic-inference` for ragged — it forces `use_ragged=False` entirely.

### Phase 4: Tensor Dump Sanitization

`sanitize_dumps.py` processes dump dirs:
1. Matches dumps to definitions via `fi_api` function name
2. Pairs `plan()` dumps with the following `run()` dump (same PID) to get structural tensors
3. Maps plan kwargs: `paged_kv_indptr→kv_indptr`, `paged_kv_indices→kv_indices`, etc.
4. Tensor storage policy:
   - `int32`/`int64` (structural: indptrs, indices) → saved to safetensors blob
   - float activations (`q`, `k_cache`, `v_cache`) → `{"type": "random"}` (shapes validated but values irrelevant for benchmarking)
   - scalars (`sm_scale`) → `{"type": "scalar", "value": <float>}`
5. Trims `kv_indices` to `kv_indptr[-1]` (SGLang over-allocates KV pool)
6. Deduplicates: at most 2 entries per unique axes combination

### Phase 5: Baseline Evaluation

Runs the baseline solution against collected workloads before PR submission:
```bash
flashinfer-bench run --local {trace_dir} --definitions {def_name} --solutions baseline
# → writes {trace_dir}/traces/{def_name}_baseline.jsonl
```

All entries must have `evaluation.status == "PASSED"`. If any fail, do not submit PR 2.

### Phase 6: Submit PR 2 (HuggingFace flashinfer-trace)

One HuggingFace PR per definition. PR 1 (GitHub flashinfer-bench) must already be open.

**PR 2 contents:**
1. `solutions/baseline/{op_type}/{def_name}/flashinfer_wrapper_*.json` — FlashInfer API wrapper (calls `BatchDecodeWithPagedKVCacheWrapper` or `BatchPrefillWithPagedKVCacheWrapper`, **not** `reference_impl`)
2. `workloads/{op_type}/{def_name}.jsonl`
3. `blob/workloads/{op_type}/{def_name}/*.safetensors`
4. `definitions/{op_type}/{def_name}.json` (copied from PR 1)
5. `tests/references/test_{def_name}.py` (copied from PR 1)
6. `traces/{op_type}/{def_name}.jsonl` (baseline eval trace, all PASSED)

**PR description must include** the full stdout of `collect_workloads.py sglang` under `## SGLang Collection Log`. The log must show real ShareGPT inference with diverse `(batch_size, kv_length)` pairs — uniform tiny KV caches (e.g. `batch_size=4096` with 1-page contexts) indicate synthetic data, not real inference.

## Output Format

```
{flashinfer_trace_dir}/workloads/{op_type}/{def_name}.jsonl
{flashinfer_trace_dir}/blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors
```

Each JSONL line:
```json
{
  "definition": "gqa_paged_decode_h32_kv8_d128_ps1",
  "workload": {
    "uuid": "a1b2c3d4-...",
    "axes": {"len_indptr": 33, "num_kv_indices": 4096},
    "inputs": {
      "q": {"type": "random"},
      "k_cache": {"type": "random"},
      "v_cache": {"type": "random"},
      "kv_indptr": {"type": "safetensors", "path": "...", "tensor_key": "kv_indptr"},
      "kv_indices": {"type": "safetensors", "path": "...", "tensor_key": "kv_indices"},
      "kv_last_page_len": {"type": "safetensors", "path": "...", "tensor_key": "kv_last_page_len"},
      "sm_scale": {"type": "scalar", "value": 0.0883}
    }
  }
}
```

## Error Handling

**No tensor dumps generated**: verify `FLASHINFER_LOGLEVEL=10` is set before any FlashInfer import; check `FLASHINFER_DUMP_INCLUDE` matches actual API function names; confirm `--attention-backend flashinfer`.

**`run()` not captured**: look for `cudaErrorStreamCaptureUnsupported` in SGLang log. Fix: always pass both `--disable-cuda-graph` and `--disable-piecewise-cuda-graph`.

**Ragged prefill yields 0 workloads**: two possible causes — (1) `--enable-deterministic-inference` is set, which forces `use_ragged=False` globally — never set this for ragged definitions; (2) piecewise CUDA graph is active (default even without `--enable-deterministic-inference`), so `is_in_piecewise_cuda_graph()=True` during capture forces `use_ragged=False`, and the cached graph always routes to `BatchPrefillWithPagedKVCacheWrapper`. Fix: add `--disable-piecewise-cuda-graph`. The script auto-detects ragged prefill definitions and adds this flag automatically.

**Constant axis mismatch across TP**: use `--skip-const-axis-check` when collecting TP=1 dumps for a TP=2 definition (structural tensors are identical across TP).

**SGLang not wired to target FlashInfer API**: grep for the `fi_api` function name in `tmp/sglang/python/sglang/srt/`. If missing, the `onboard-model` skill handles submitting a SGLang PR to wire it in.

## Prerequisites

Run `/clone-repos` first. Then:
```bash
/clone-repos
/extract-kernel-definitions --model-name <model>
/collect-workloads --op-type <op_type> --model-path /path/to/model
```
