# Definition Analysis

Analyze model kernel coverage and write reviewed FlashInfer-Bench Definition JSON files.

## Mandatory References

Before inspecting or editing definitions, read and follow both current workflow references:

- `.claude/skills/onboard-model/references/definition_standards.md`
- `.claude/skills/onboard-model/references/non_fi_capture.md`

Treat these files as normative. Do not use legacy proposal, merge, promote, manifest, or
old workload-collection skills. In particular, apply the canonical naming review to every
existing and newly written non-FI definition before finishing.

## Rules

- Analyze the supplied evidence, then write or rewrite Definition JSON directly. Do not
  create a separate patch artifact.
- Edit only files under the definitions directory.
- Preserve valid native FlashInfer definitions unless source evidence proves they are wrong.
- Inspect every observed SGLang module and identify kernel-level tensor operations not
  represented by a native FlashInfer definition.
- Write source-backed non-FI definitions for missing kernel-level operations. Do not finish
  with only native FI definitions when the inventory contains custom activation,
  normalization, rotary, or other direct tensor-kernel modules.
- Do not create definitions for generic linear layers, whole decoder blocks, model
  containers, logits processors, samplers, or attention orchestration wrappers unless the
  source proves a distinct standalone kernel contract.
- Every collectable definition must choose exactly one backend.
- For FlashInfer, use `fi_api:<exact decorated callable>`; never tag a wrapper class.
- For non-FI torch modules, use
  `sglang_module:<exact fully-qualified class observed in
  reports/evidence/sglang_execution_inventory.json>`.
- For non-FI plain functions, use `sglang_callable:<exact fully-qualified callable>` only
  with source evidence.
- For non-FI definitions, never put the current model family, framework, or class name in
  `name` or `op_type`. Name the observable semantic behavior and use `model:<slug>` only as
  provenance.
- Definition input names default to forward or callable argument names.
- Infer every input dtype from runtime evidence or the source callable contract. Do not
  assume module attributes use the model activation dtype; current SGLang rotary CUDA
  kernels require a `float32` cosine-sine cache even when query/key use `bfloat16`.
- If names differ, add
  `sglang_input:<definition_input>=arg:<runtime_argument>`.
- If an observed forward signature is `(*args, **kwargs)`, map positional tensors with
  zero-based tags such as `sglang_input:x=arg:0`; inventory keys `arg_0`, `arg_1`, and so on
  describe those positions.
- Module attributes may use
  `sglang_input:<definition_input>=attr:<attribute_path>`.
- Definition analysis uses executed module paths and signatures from the inventory. During the
  later workload stage, SGLang Dumper captures raw inputs for reviewed `sglang_module:`
  definitions; do not invent Dumper evidence during definition analysis.
- Keep the formal Definition schema and a top-level reference `run(...)` function.
- Write a non-empty `description` for the definition and for every axis, input, and output.
  The deterministic definition review treats missing descriptions as errors.
- The reference `run(...)` return count must equal the declared outputs count.
- For GQA, verify query and KV head counts and keep encoded name axes in sync.
- Do not edit `config/`, `output/`, `reports/`, tests, or repository source.
- Do not launch GPU stages and do not invent source evidence.
- `sglang_module:`, `sglang_callable:`, `sglang_input:`, and
  `status:source_reviewed` are capture-only metadata. Workload export moves them to
  `reports/evidence/capture_metadata.json`; they must not appear in submitted definitions.
