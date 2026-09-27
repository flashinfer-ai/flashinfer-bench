# Model Onboarding

Model onboarding has two GPU capture stages and one reviewed Definition source:

```text
dump-definition -> review/edit definitions -> dump-workload -> review dataset
```

`definitions/` is the single reviewed source. There is no proposal/config copy. Workload
export removes capture-only metadata from the publication copy under `output/definitions/`;
the reviewed files remain unchanged.

## 1. Dump and Review Definitions

```bash
flashinfer-bench onboarding dump-definition \
  --run qwen3/20260711 \
  --model-name Qwen/Qwen3-1.7B \
  --agent codex
```

The command runs a bounded SGLang request matrix and writes:

```text
runs/qwen3/20260711/
├── config/run_config.json
├── definitions/<op_type>/<definition_name>.json
└── reports/
    ├── definition_report.json
    ├── definition_review.md
    └── evidence/
        ├── sglang_execution_inventory.json
        └── request_manifest.json
```

Two evidence sources are combined:

- FlashInfer native tracing writes definitions tagged with `fi_api:`.
- SGLang worker capture records the module classes, paths, signatures, input shapes, and
  bounded source snippets that actually executed.

With `--agent codex`, the agent reads this evidence and writes Definition JSON directly.
It may preserve native FlashInfer definitions or add non-FI definitions; it may edit only
`definitions/`.

Each collectable definition chooses exactly one capture backend:

```text
fi_api:<exact decorated FlashInfer callable>
sglang_module:<exact fully-qualified torch.nn.Module class>
sglang_callable:<exact fully-qualified plain callable>
```

Non-FI definition inputs default to runtime argument names. Explicit mappings are available
when names differ or an input comes from module state:

```text
sglang_input:<definition_input>=arg:<runtime_argument>
sglang_input:<definition_input>=attr:<module_attribute>
```

The deterministic review checks the Definition schema, path/name consistency, reference
`run(...)`, capture-tag exclusivity, observed SGLang evidence, exact FlashInfer callables,
and GQA head/name invariants. Review and edit `definitions/` in place. Existing reviewed
definitions are overwritten only when `--overwrite` is explicit.

## 2. Dump Workloads

```bash
flashinfer-bench onboarding dump-workload \
  --run qwen3/20260711
```

Before GPU execution, the command validates every definition and records a SHA-256 digest.
The workload pass replays the request manifest from the definition stage:

- `fi_api:` definitions use FlashInfer's native logger and
  `flashinfer_bench.tracing.sanitize`;
- `sglang_module:` definitions use SGLang Dumper inputs plus a thin
  Definition/path/attribute adapter;
- `sglang_callable:` definitions keep a reviewed callable wrapper because SGLang Dumper
  hooks modules, not arbitrary Python functions;
- all capture paths converge on the existing `TracingRuntime` and standard dataset layout.

Outputs:

```text
runs/qwen3/20260711/
├── definitions/                 # reviewed input, unchanged
├── output/                      # latest complete snapshot
│   ├── definitions/
│   ├── workloads/
│   └── blob/
└── reports/
    ├── run_report.json
    ├── review.md
    └── evidence/capture_metadata.json
```

The command fails when a collectable definition produces no workload, the definition digest
changes during the run, or canonical dataset validation fails. With `--agent codex`, a failed
run may update `definitions/`; review the changed snapshot and rerun `dump-workload`.

`output/` is replaced as one snapshot and never appends to stale output. Capture-only
`sglang_*` tags remain in reviewed definitions but are removed from publication copies and
recorded in `reports/evidence/capture_metadata.json`.

## Runtime Config

The first command writes `config/run_config.json`. `model_name` is required; `tp_size`
defaults to 1. CLI values override or initialize persisted values. Common optional fields:

```json
{
  "batch_sizes": [1, 2, 4, 8],
  "max_new_tokens": 16,
  "max_new_workloads": 20,
  "disable_cuda_graph": true,
  "force_flashinfer_backends": true,
  "mem_fraction_static": 0.7,
  "engine_kwargs": {},
  "isl": 1024,
  "osl": 8,
  "random_range_ratio": 1.0,
  "seed": 0
}
```

Both stages use the same persisted synthetic request matrix. It includes 128-token and `isl`
inputs across configured batch sizes, one bounded long-context request, and one shared-prefix
batch. `osl` is the generated-token length; `random_range_ratio=1.0` uses exact lengths.
This reuses controllable request generation without running an HTTP throughput benchmark.

## Failure Rules

- `definitions failed review`: fix findings in `reports/definition_review.md`.
- `no reviewed definition contains a supported capture tag`: add an evidenced capture point.
- a collectable definition has no workload: confirm that its capture point executed and its
  runtime inputs satisfy the Definition.
- dataset validation fails: inspect `reports/run_report.json` and `reports/review.md`.
- definition digest mismatch: review the changed Definition snapshot and rerun.

## Internal Call Chain

```text
cli.py
  -> definition_stage.py / workload_stage.py
  -> planning.py
  -> runners/stage_runner.py
  -> runners/remote_pipeline.py
       -> runners/sglang_runner.py
       -> tracing/flashinfer_logging.py -> tracing/sanitize.py
       -> tracing/SGLang capture -> tracing/TracingRuntime
  -> data.validate
```

Run both stages in an environment that provides SGLang, FlashInfer, model access, and the
requested GPU. Cluster submission, job recovery, reference-test generation, and dataset PR
submission are separate deployment or publication workflows.
