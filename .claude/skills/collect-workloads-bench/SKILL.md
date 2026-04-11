# collect-workloads-bench

Collect real workloads using `examples/sglang_bench/bench_sharegpt.py` as the server
launcher. This is an improved alternative to `collect_workloads.py sglang` that:

- Pulls model-specific server flags (TP, EP, attention-backend, etc.) from `examples/sglang_bench/model_configs.json`
- Uses ShareGPT prompts with batch sweeps for diverse (batch_size, kv_len) coverage
- Handles server lifecycle cleanly via `popen_launch_server` / `kill_process_tree`
- Still uses the same FlashInfer Level-10 tensor dump + `sanitize_dumps.py` pipeline

## When to use

Use when:
- The model is listed in `examples/sglang_bench/model_configs.json` (or can be added)
- You need a **paged prefill** collection (add `--disable-radix-cache` flag; it's already
  present for qwen3-235b-a22b)
- You need TP/EP parallelism without hand-coding server flags

## Workflow

### 1. Ensure model is in model_configs.json

Edit `examples/sglang_bench/model_configs.json` to add/confirm entry:
```json
"<model-key>": {
    "server_flags": [
        "--trust-remote-code",
        "--attention-backend", "flashinfer",
        "--tp-size", "<N>",
        "--disable-radix-cache",       // add for paged-prefill definitions
        "--disable-cuda-graph"
    ]
}
```
For paged prefill: always include `--disable-radix-cache` and `--enable-deterministic-inference`.

### 2. Set FlashInfer Level-10 dump env vars and run bench_sharegpt.py (incremental)

Run one batch size at a time to avoid node overload. After each run, sanitize to collect
2–3 workloads, then delete the dump dir before the next run.

**Multi-node auto-detection**: when model config TP > local GPU count (e.g. TP=8 config but
only 4 GPUs visible via `nvidia-smi`), `bench_sharegpt.py` automatically triggers multi-node
mode. Peer nodes are discovered from `SLURM_JOB_ID`; workers are launched via SSH.
FLASHINFER dump env vars are forwarded only to the head node (rank 0) — workers don't dump.
Passwordless SSH between allocated nodes is required (standard in SLURM environments).

```bash
DUMP_DIR=/tmp/flashinfer_dumps_<name>
TRACE_DIR=<trace_dir>
LOG=/tmp/bench_sharegpt_<name>.log

cd /home/averyh/flashinfer-bench

for BS in 1 64 128 256; do
  echo "=== Collecting batch_size=$BS ===" | tee -a $LOG
  rm -rf $DUMP_DIR && mkdir -p $DUMP_DIR

  tools/gpu-lock --gpus <N> --exec-timeout 1800 -- \
    conda run -n flashinfer_bench env \
      FLASHINFER_LOGLEVEL=10 \
      FLASHINFER_DUMP_DIR=$DUMP_DIR \
      FLASHINFER_DUMP_SAFETENSORS=1 \
      FLASHINFER_DUMP_INCLUDE="<fi_include_pattern>,<fi_plan_pattern>" \
      FLASHINFER_DUMP_EXCLUDE="*.__init__" \
      FLASHINFER_DUMP_MAX_COUNT=100 \
      FLASHINFER_DUMP_MAX_SIZE_GB=2 \
      FLASHINFER_USE_CUDA_NORM=1 \
      FLASHINFER_DISABLE_VERSION_CHECK=1 \
      SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
      SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
    python examples/sglang_bench/bench_sharegpt.py \
      --model <model-key> \
      --model-path <model-path> \
      --batch-sizes $BS \
      --num-batches 4 \
      --disable-cuda-graph \
      2>&1 | tee -a $LOG

  # Kill any lingering processes (including remote workers if multi-node)
  pkill -f bench_sharegpt.py || true
  pkill -f sglang.launch_server || true

  # Sanitize (no --replace so workloads from previous batch sizes are kept)
  conda run -n flashinfer_bench python scripts/sanitize_dumps.py \
    --dump-dir $DUMP_DIR \
    --definitions <def_name> \
    --flashinfer-trace-dir $TRACE_DIR

  # Delete dumps before next run to free disk space
  rm -rf $DUMP_DIR
done
```

#### Multi-node flags (bench_sharegpt.py)

| Flag | Default | Purpose |
|------|---------|---------|
| `--peer-node-addr HOST [HOST ...]` | auto (SLURM) | Override peer node hostname(s)/IP(s) |
| `--dist-init-port PORT` | `20010` | PyTorch dist rendezvous port (must differ from `--port`) |
| `--no-multinode` | off | Disable auto multi-node even when config TP > local GPUs |
| `--conda-env ENV` | `$CONDA_DEFAULT_ENV` | Conda env to activate on peer nodes via SSH |

**Two independent decisions**:

1. **What TP to use** — always from model config:
   - Config has `--tp-size N` → use N as-is
   - Config has no `--tp-size` → fall back to local GPU count
   - **Config TP is never overridden**

2. **How many nodes to use** — derived from TP and local GPU count:
   - `needed_nodes = ceil(config_tp / local_gpus)`
   - Only launch multi-node if `needed_nodes > 1` — peer nodes are looked up from SLURM/`--peer-node-addr` only when needed

**Examples** (4 GPUs per node, 2-node SLURM allocation):
| Config TP | needed_nodes | Launch mode |
|-----------|-------------|-------------|
| `--tp-size 4` | 1 | Single-node, TP=4 (fits on one node) |
| `--tp-size 8` | 2 | Multi-node, TP=8 (4 GPUs × 2 nodes) |
| *(no flag)* | 1 | Single-node, TP=4 (local GPU count fallback) |

**Manual peer override** (when not in SLURM or peer auto-detection fails):
```bash
python examples/sglang_bench/bench_sharegpt.py \
  --model deepseek-v3 --model-path /path/to/model \
  --peer-node-addr nvl72155-T14 \
  --dist-init-port 20010 \
  ...
```

**Note**: Omit `--replace` during the loop so each batch size's workloads are appended.
Use `--replace` only on the first run if you want to start fresh.

**FLASHINFER_DUMP_INCLUDE pattern** — always include both `plan*` and `run*`:
- For paged prefill: `"BatchPrefillWithPagedKVCacheWrapper.plan*,BatchPrefillWithPagedKVCacheWrapper.run*"`
- For paged decode: `"BatchDecodeWithPagedKVCacheWrapper.plan*,BatchDecodeWithPagedKVCacheWrapper.run*"`

Including `plan*` alongside `run*` is required so `sanitize_dumps.py` can pair each `run()` dump with
its `plan()` dump — this enables automatic skipping of const-axis checks on tensors not captured
in the `run()` dump (k_cache, v_cache). If you only captured `run*` dumps, add
`--skip-const-axis-check` to `sanitize_dumps.py`.

### 3. Post-process dumps with sanitize_dumps.py

```bash
conda run -n flashinfer_bench python scripts/sanitize_dumps.py \
  --dump-dir $DUMP_DIR \
  --definitions <def_name> \
  --flashinfer-trace-dir <trace_dir> \
  --replace
```

**If sanitize reports 0 workloads with const axis warnings**, a definition constant is wrong.
Diagnose and fix with:

```python
# Inspect actual tensor shapes from a dump
from pathlib import Path
import safetensors.torch, json

dump_dir = Path(DUMP_DIR)
first = sorted(dump_dir.iterdir())[0]
inputs = safetensors.torch.load_file(first / "inputs.safetensors")
for k, v in inputs.items():
    print(f"  {k}: {v.shape} {v.dtype}")
meta = json.loads((first / "metadata.jsonl").read_text().splitlines()[0])
print("tensor_details:", json.dumps(meta.get("tensor_details", {}), indent=2))
```

Then update the definition JSON to match the actual shapes:
1. Fix the const axis value (e.g. `head_dim: 64 → 128`)
2. Rename the definition file/name to reflect the corrected value (e.g. `_d64_` → `_d128_`)
3. Fix the `assert` in the `reference` implementation
4. Rename the workload `.jsonl` and blob dir to match the new definition name
5. Delete the old (wrong) definition JSON
6. Update any sibling definitions for the same model (e.g. `_ps1` / `_ps64` variants share the same `head_dim`)
7. Re-run `sanitize_dumps.py` with the corrected definition name

### 4. Verify NOT synthetic

```bash
python3 -c "
import json
lines = open('<trace_dir>/workloads/gqa_paged/<def_name>.jsonl').readlines()
entries = [json.loads(l) for l in lines]
axes = list(entries[0]['workload']['axes'].keys())
print('axes:', axes, '| total:', len(entries))
# Check for diversity - look at first axis that varies
first_key = axes[0]
vals = sorted(set(e['workload']['axes'][first_key] for e in entries))
print(f'{first_key} unique:', len(vals), '| sample:', vals[:10])
if len(vals) < 5: print('WARNING: low diversity'); exit(1)
print('REAL workloads OK')
"
```

### 5. Run baseline eval + commit

```bash
cd /home/averyh/flashinfer-bench
conda run -n flashinfer_bench python -m flashinfer_bench run \
  --local <trace_dir> --definitions <def_name> --save-results \
  --warmup-runs 0 --iterations 1 --num-trials 1
```

Then commit and force-push to HF PR2:
```bash
cd <trace_dir>
git add workloads/ blob/ traces/ solutions/
git commit -m "workloads: SGLang-collected <def_name> via bench_sharegpt.py"
git push origin HEAD:refs/pr/<pr2_num> --force
```

Post the bench_sharegpt.py log to the HF PR2 discussion under `## SGLang Collection Log`.

## Notes

- For `--enable-deterministic-inference` (paged prefill): add it to the model's `server_flags`
  in `model_configs.json`, or pass it via bench_sharegpt.py's `server_args` list
- The existing `collect_workloads.py` dump-processing pipeline (sanitize_dumps.py) is unchanged
- `bench_sharegpt.py` passes `**os.environ` to the server process, so all `FLASHINFER_*` vars
  set in the outer shell are inherited by the SGLang server automatically
