---
name: sanitize-fi-log
description: Convert FlashInfer level-3 API logs into per-API workload JSONL files (random-input form). Use when you have a FLASHINFER_LOGLEVEL>=3 log and want workload entries without dumping real tensors. Pair with FlashInfer trace-template JSONs to produce schema-aligned workloads.
---

# Sanitize FlashInfer Level-3 Logs into Workloads

Convert a FlashInfer level-3 (or higher) API log into per-API workload JSONL files
where every tensor input is marked `type: random` and shapes/dtypes are derived
from the log. Optionally align each call to a FlashInfer trace-dumped definition
JSON to produce a schema-correct workload (named inputs, resolved axes,
`definition` field set).

## When to use this vs `collect-workloads`

| Situation | Skill |
|---|---|
| You have a `FLASHINFER_LOGLEVEL=3` log already and just need workload metadata (shapes/dtypes), not real tensors. | **this skill** |
| You want runnable workloads with real tensors saved as safetensors blobs (for correctness eval, replay, eval traces). | `collect-workloads` (level-10 dumps) |
| You need to capture a workload distribution from a live SGLang run end-to-end (run inference → blobs → push to HF). | `collect-workloads` |

The level-3 path is faster and cheaper (no tensor dumps), and useful when you
already have logs from a past run. The output is **random-input** workloads —
suitable for shape coverage and definition discovery, *not* for correctness
evaluation.

## Inputs

- A FlashInfer log file produced with `FLASHINFER_LOGLEVEL=3` (or 5/10).
- *(optional)* a directory of FlashInfer trace-dumped definition JSONs
  (those produced by `@flashinfer_api(trace=...)` + `FLASHINFER_TRACE_DUMP=1`,
  e.g. `tests/trace/fi_trace_*` in the FlashInfer repo).

## Command

```bash
python3 scripts/sanitize_fi_log.py \
    --log-file  /path/to/fi_apilog_<pid>.log \
    --output-dir /path/to/workloads_out/ \
    [--trace-dir /path/to/trace_jsons/] \
    [--strict] \
    [--include-defaults] \
    [--dedupe]
```

Flags:

- `--trace-dir` — when supplied, every log call is matched against the trace
  templates in that dir. On match, the workload uses the definition's named
  inputs and resolved axes (no embedded shape). On miss, it falls back to a
  raw form keyed by `arg_N` / `kwarg_NAME` with shape/dtype embedded inline.
- `--strict` — disable the lenient name-based fallback. Calls that don't
  *fully* align (rank match + const consistent) become raw. Use this when you
  want to know whether the *templates themselves* are correct (any raw entry
  is a real schema/runtime mismatch worth debugging).
- `--include-defaults` — also emit default-valued kwargs. Off by default
  because most defaults are signature noise (`enable_pdl=None`, `out=None`).
- `--dedupe` — collapse duplicate calls to one entry per file. Off by default;
  every call becomes its own workload entry.

Output filename:

- Matched call → `<definition_name>.jsonl` (e.g. `rmsnorm_h7168.jsonl`).
- Unmatched call → `<api_name>.jsonl` (e.g. `BatchMLAPagedAttentionWrapper.__init__.jsonl`).

## Output format

Each line of every JSONL is a `Trace`-shaped object with `solution`/`evaluation`
left null:

```jsonc
// Matched (with --trace-dir):
{
  "definition": "rmsnorm_h7168",
  "workload": {
    "uuid": "...",
    "axes": {"batch_size": 128},
    "inputs": {
      "hidden_states": {"type": "random"},
      "weight":        {"type": "random"}
    }
  },
  "solution": null,
  "evaluation": null
}

// Unmatched (raw):
{
  "definition": null,
  "workload": {
    "uuid": "...",
    "api": "some_unknown_api",
    "inputs": {
      "arg_0":      {"type": "random", "shape": [8, 4], "dtype": "float32"},
      "kwarg_alpha":{"type": "scalar", "value": 0.5}
    }
  },
  "solution": null,
  "evaluation": null
}
```

## Matching pipeline (when `--trace-dir` is supplied)

For each log call, candidate definitions are looked up by **suffix-matching**
the log's API name against every `fi_api:` tag in the trace dir. For each
candidate the matcher tries, in order:

1. **Strict — name match.** Map kwargs to def inputs whose names are equal.
2. **Strict — shape match.** Greedily assign log tensors whose rank/dtype/const-
   axis values uniquely identify a def input.
3. **Strict — positional fallback.** When some def inputs are unmatched and all
   remaining log args are positional, pair them in declaration order, accepting
   only if every pair is fully shape/dtype/const-consistent.
4. **Lenient — name match (skipped under `--strict`).** Map by kwarg name and
   tolerate per-input rank/const mismatches (still requires at least one
   cleanly rank+const-consistent input as a sanity guard). This handles
   templates where the schema doesn't perfectly model the runtime — e.g.
   MLA-style rank-collapsed K tensors.

A call is emitted as **raw** only if all four passes reject every candidate
(or if there's no candidate at all).

## Triaging raw entries

When a call is raw under `--strict`, it can be one of four root causes:

| Symptom | Likely cause | Fix |
|---|---|---|
| API doesn't appear in any `fi_api:` tag of any trace JSON. | Template missing. | Add a `TraceTemplate` in `flashinfer/flashinfer/trace/templates/` for that API and decorate the API with `@flashinfer_api(trace=...)`. |
| Template exists but log tensor has different rank than schema. | Schema doesn't model a runtime variant (e.g. MLA's rank-2 K, 4D kv_cache). | Adjust `dim_names` in the trace template to match the runtime. For dual-rank cases (3D vs 4D backwards-compat), pick the actual runtime form and add a leading const axis if needed. |
| Template's const axis disagrees with the log dim at that position. | Template's axis is bound to a wrong tensor dim, or two tensors share a var name unintentionally. | Often `workspace_buffer` shape declared as `[num_pages]` or `[num_q_tokens]` — workspace_buffer is a flat byte buffer; give it its own `workspace_size: Var` and dtype `uint8`. |
| Schema dim order is the inverse of runtime dim order. | Template was authored from the API docstring but the actual call passes the tensor transposed (or vice versa). | Verify against the *log* (authoritative) — flip the `dim_names` to match. If the docstring disagrees, flag it; the docstring may be wrong, or an upstream caller may be transposing before invocation. |

If a raw entry under `--strict` becomes matched under lenient mode (no `--strict`),
that confirms the issue is a template/runtime mismatch worth fixing on the
FlashInfer side. Lenient is acceptable as a stopgap — strict-clean is the goal.

## Recurring template bugs to grep for

When auditing FlashInfer trace templates against actual logs:

- **`workspace_buffer` aliasing** — any template declaring `Tensor(["num_pages"], ...)` or `Tensor(["num_q_tokens"], ...)` for `workspace_buffer` is wrong; the buffer is a flat byte array sized in MB, unrelated to the page/token vars.
- **MLA rank collapse** — any template that declares K/V tensors with a `num_k_heads` axis under MLA passes them rank-collapsed; either define a separate MLA template or drop the kv-head axis.
- **head_dim split (DeepSeek)** — DeepSeek MLA splits Q/K's head_dim from V's projection dim. Templates that use a single `head_dim` const for all three need `head_dim_qk` / `head_dim_v`.
- **bool scalars typed as int32** — cosmetic, but `is_causal` / `enable_pdl` / `return_lse` should be `Scalar("bool")` not `Scalar("int32")`.

## Worked example (DSR1 + SGLang on B200)

Given a level-3 log from an 8-rank DSR1 inference run plus the trace JSONs that
SGLang's run produced (auto-dumped under the same folder via
`FLASHINFER_TRACE_DUMP=1`):

```bash
python3 scripts/sanitize_fi_log.py \
    --log-file  /home/averyh/fi_log_runs/level3/good/2026-05-01T2113/fi_apilog_150365.log \
    --output-dir /home/averyh/fi_log_runs/level3/good/2026-05-01T2113 \
    --trace-dir  /home/averyh/fi_log_runs/level3/good/2026-05-01T2113 \
    --strict
# parsed 64673 API call(s); 36549 matched a definition, 28124 emitted as raw
```

Output is one JSONL per matched definition (e.g. `rmsnorm_h7168.jsonl`,
`mla_rope_quantize_fp8_h16_rope64.jsonl`, …) plus one raw JSONL per unmatched
API. Use `--strict` for an audit pass over your trace templates; drop it when
you just need workload data and don't care that some calls aren't perfectly
schema-aligned.
