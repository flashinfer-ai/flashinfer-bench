# Non-FI Capture Rules

Use these rules only for kernel-level operations that are observed in SGLang and do not
already have an equivalent native FlashInfer definition.

## Evidence

Use the supplied `reports/evidence/sglang_execution_inventory.json` as execution and
source evidence. SGLang's
tensor logger report is output-only comparison evidence; it does not prove complete input
capture and must not be used as the workload input source.

Do not create definitions for generic linear layers, whole decoder blocks, model
containers, attention orchestration wrappers, logits processors, or samplers unless the
source proves a distinct standalone kernel contract.

## Capture Backend

Every collectable definition chooses exactly one backend:

- `fi_api:<exact decorated callable>` for a native FlashInfer API;
- `sglang_module:<exact observed torch.nn.Module class>` for a non-FI module;
- `sglang_callable:<exact source-backed callable>` for a non-FI function.

Never combine these backend kinds in one definition. Never invent a FlashInfer API or use
a wrapper class as an `fi_api` tag.

## Input Mapping

Definition input names default to the exact `forward` or callable argument names. Add a
mapping only when needed:

```text
sglang_input:<definition_input>=arg:<runtime_argument>
sglang_input:<definition_input>=arg:<zero_based_position>
sglang_input:<definition_input>=attr:<module_attribute_path>
```

Use positional mappings when the observed signature is `(*args, **kwargs)`. Use attribute
mappings only for stable module state that is required by the contract, such as a weight,
epsilon, cache, or semantic flag.

Scalar attributes must remain scalar definition inputs with `"shape": null`. Tensor
attributes retain their tensor shape and dtype. Do not turn implementation metadata into
workload inputs unless the reference semantics consume it.

## Definition Scope

- Prefer a generic reusable op type and name. The SGLang class path belongs in the capture
  tag, not automatically in the definition name.
- Never put the current model family in a non-FI `name` or `op_type`. Encode the observable
  semantic difference and keep model provenance in a `model:<slug>` tag.
- Create separate definitions when source-backed constant axes or semantics differ.
- Reuse an existing semantic contract when only the implementation class differs.
- Preserve mutation, output count, dtype behavior, slicing, and optional-input behavior
  from source.

Before finishing, confirm that each non-FI definition names an observed exact capture
point, can obtain every required input, has a pure PyTorch reference, and follows
`references/definition_standards.md`.
