# Definition Standards

Use these rules when writing reviewed FlashInfer-Bench Definition JSON files.

## Contract Boundary

A definition names a stable kernel/workload contract, not a model, Python class, or one
runtime call. Reuse one definition across models and implementations when their inputs,
outputs, semantics, and constant axes are the same.

Native definitions produced by a current FlashInfer trace template are authoritative. Do
not rename or rewrite them unless source evidence proves that the template is wrong.

## Canonical Naming

Prefer:

```text
{op_type}_{semantic_variant}_{key_const_axes}
```

Omit the variant when the base op type is sufficient. The name must:

- use lower snake_case;
- start with the stable semantic `op_type`;
- encode only semantic variants and reviewed constant axes;
- exclude runtime-variable axes such as batch size, token count, and sequence length;
- exclude model, vendor, framework, and Python class names. Those are provenance, not
  workload semantics.

Do not copy `Gemma4`, `Qwen`, `Llama`, `SGLang`, or a class name into `name`/`op_type`.
Describe the actual behavior instead, such as `scale_shift`, `cross_mixed`, `paged`, or
`fused_add`, and put model provenance in a `model:<slug>` tag. For example, use
`rmsnorm_scale_shift_h1536` and `rotary_embedding_cross_mixed_h8_kv1_d512`, not names that
contain `gemma4`.

The only exception is an already-established canonical contract in the official dataset or
public API. An Agent must not create a new model-named exception; it requires existing
source evidence and human approval.

Before creating a new name, compare existing definitions. If only constant axes differ,
reuse the same op type and variant and change only the encoded constants.

Common axis abbreviations:

- `h`: number of heads or hidden size, according to the op type;
- `kv`: number of KV heads;
- `d`: head dimension;
- `ps`: page size;
- `ckv`: compressed KV dimension;
- `kpe`: key positional encoding dimension;
- `i`: intermediate size;
- `e`: number of experts;
- `topk`: selected experts or sparse top-k;
- `v`: vocabulary size.

Examples:

```text
rmsnorm_h4096
fused_add_rmsnorm_h7168
gelu_tanh_and_mul_i6144
rotary_embedding_h8_kv1_d512
gqa_paged_decode_h32_kv8_d128_ps1
```

The file must be `definitions/{op_type}/{name}.json`; its `name` and `op_type` fields must
match that path.

## Schema and Reference

- Use `{"type": "var"}` for runtime axes and `{"type": "const", "value": N}` for
  fixed axes.
- Write a non-empty `description` for the definition and for every axis, input, and output.
  Missing nested descriptions are submission errors, not advisory warnings.
- Tensor inputs and outputs require `shape` and `dtype`.
- Python scalar inputs use `"shape": null`, not an empty shape.
- Never publish `dtype: "unknown"`. Resolve index-tensor and plan-state dtypes from the
  traced API, source, or ground-truth test before submission.
- Input and output names must not overlap. For an in-place kernel, keep the mutated value as
  an input and declare only the values actually returned by `run(...)` as outputs.
- Optional inputs use `"optional": true`. Output dtypes inherited from inputs must resolve
  to a concrete supported dtype before publication.
- The top-level `reference` must define `run(...)`, preserve source-backed argument order
  and semantics, and return exactly the declared outputs.
- A `status:verified` native definition keeps the FlashInfer trace-template reference.
- A reviewed non-FI definition may use a pure PyTorch reference transcribed from the exact
  SGLang source. It must not call the optimized SGLang or FlashInfer kernel.

## Submission Evidence

- Every new published definition requires
  `output/tests/references/test_<definition_name>.py`.
- Prefer the matching FlashInfer implementation as test ground truth. Use the exact SGLang
  implementation only when FlashInfer has no equivalent kernel.
- The test must exercise the definition's `run(...)`, compare every declared output, and
  use tolerances appropriate for the declared dtype.
- Capture-only `sglang_module:`, `sglang_callable:`, and `sglang_input:` tags belong in the
  evidence sidecar, not in the published definition.
- Run `check-submission` after workload collection; schema-valid workloads do not by
  themselves make a definition ready for publication.

## Final Naming Review

Before finishing, verify for every file:

1. Does the name describe a reusable semantic contract rather than its model/class origin?
2. Does it start with the generic op type?
3. Are all suffixes semantic variants or constant axes?
4. Would an existing definition with the same contract already cover it?
5. Do `op_type`, `name`, directory, filename, axes, and reference agree?
6. Are all descriptions present and all dtypes concrete?
7. Does the reference test compare every declared output against valid ground truth?
