---
name: collect-workloads-bench
description: Run SGLang inference via examples/sglang_bench/bench_serving.py (model-specific server flags, multi-node auto-detection) and convert the output into workload JSONL files. Defaults to level-3 logging + the sanitize-fi-log skill; opt in to level-10 logging + the sanitize-fi-dumps skill when real tensors are required.
---

# collect-workloads-bench

Collect real workloads using `examples/sglang_bench/bench_serving.py` as the
server launcher.  This is an improved alternative to `collect_workloads.py
sglang` that:

- Pulls model-specific server flags (TP, EP, attention-backend, etc.) from
  `examples/sglang_bench/model_configs.json`.
- Drives the server with random or ShareGPT prompts (default `--dataset random`,
  controlled via `--isl`/`--osl`) for diverse `(batch_size, kv_len)` coverage.
- Handles server lifecycle cleanly via `popen_launch_server` /
  `kill_process_tree`.
- Auto-detects multi-node configurations (SLURM peer discovery + remote worker
  launch via SSH).

This skill owns the **SGLang launcher** half of the workflow.  The
**conversion-to-workload** half is delegated to one of two sibling skills,
chosen by which FlashInfer log level you ran with:

| Path | Log level | Sanitize skill | Workload tensors |
|---|---|---|---|
| **Default** | `FLASHINFER_LOGLEVEL=3` | [`/sanitize-fi-log`](../sanitize-fi-log/SKILL.md) | All `{"type": "random"}` |
| Opt-in | `FLASHINFER_LOGLEVEL=10` | [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md) | Structural ints saved as safetensors |

Pick level 10 when the workloads need to drive
`flashinfer-bench run --solutions baseline` (i.e. real eval traces for a
HuggingFace `flashinfer-trace` PR).  For shape-coverage / fast iteration,
stay on level 3.

## When to use

Use when:
- The model is listed in `examples/sglang_bench/model_configs.json` (or can be
  added).
- You need a **paged prefill** collection (add `--disable-radix-cache` flag —
  it's already present for `qwen3-235b-a22b`).
- You need TP/EP parallelism without hand-coding server flags.
- You need multi-node auto-detection (SLURM allocations spanning >1 node).

## Default flow — level 3 (lightweight)

Set `FLASHINFER_LOGLEVEL=3` and let `bench_serving.py` produce one log per rank.
Then sanitize one log with [`/sanitize-fi-log`](../sanitize-fi-log/SKILL.md).
Note that **kernel call distributions are identical across ranks** for a given
TP run, so a single rank's log gives you the full per-definition workload set —
no need to merge.

```bash
LOG_DIR=/tmp/fi_log_runs/<name>_$(date +%Y%m%d_%H%M%S)
TRACE_DIR=<flashinfer_trace_dir>      # used to source definition JSONs (auto-dumped below)

mkdir -p $LOG_DIR
cd $LOG_DIR

tools/gpu-lock --gpus <N> --exec-timeout 1800 -- \
  conda run -n flashinfer_bench env \
    FLASHINFER_LOGLEVEL=3 \
    FLASHINFER_LOGDEST=fi_apilog_%i.log \
    FLASHINFER_TRACE_DUMP=1 \
    FLASHINFER_TRACE_DUMP_DIR=$LOG_DIR \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  python /home/averyh/flashinfer-bench/examples/sglang_bench/bench_serving.py \
    --model      <model-key> \
    --model-path <model-path> \
    --batch-sizes 64 128 \
    --num-batches 2 \
    --disable-cuda-graph

# Convert one rank's log into per-definition workload JSONLs.
# All ranks' logs are equivalent — pick the first.
ONE_LOG=$(ls $LOG_DIR/fi_apilog_*.log | head -1)

python3 /home/averyh/flashinfer-bench/scripts/sanitize_fi_log.py \
    --log-file   $ONE_LOG \
    --output-dir $LOG_DIR \
    --trace-dir  $LOG_DIR \
    --strict
```

Output: one JSONL per matched definition (e.g. `rmsnorm_h7168.jsonl`,
`gemm_fp8_nt_groupwise_n2112_k7168.jsonl`).  See
[`/sanitize-fi-log`](../sanitize-fi-log/SKILL.md) for triage of any raw entries
under `--strict`.

The trace JSONs FlashInfer auto-dumps via `FLASHINFER_TRACE_DUMP=1` land in
`$LOG_DIR` alongside the logs, which is also `--trace-dir` for the sanitizer —
keep them all in one place so the sanitizer can pair each log call to its
definition without copying files around.

## Opt-in flow — level 10 (real tensors)

Use this when the workloads will be published with eval traces (HF PR 2).
Hand off to [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md) for the
sanitize step; only the SGLang launching half lives here.

### 1. Ensure model is in model_configs.json

Edit `examples/sglang_bench/model_configs.json` to add/confirm the entry:

```json
"<model-key>": {
    "server_flags": [
        "--trust-remote-code",
        "--attention-backend", "flashinfer",
        "--tp-size", "<N>",
        "--disable-radix-cache",
        "--disable-cuda-graph"
    ]
}
```

For paged prefill: always include `--disable-radix-cache` and
`--enable-deterministic-inference`.

### 2. Set FlashInfer Level-10 dump env vars and run bench_serving.py (incremental)

Run one batch size at a time to avoid node overload.  After each run, sanitize
to collect 2–3 workloads, then delete the dump dir before the next run.

**Multi-node auto-detection**: when model config TP > local GPU count (e.g.
TP=8 config but only 4 GPUs visible via `nvidia-smi`), `bench_serving.py`
automatically triggers multi-node mode.  Peer nodes are discovered from
`SLURM_JOB_ID`; workers are launched via SSH.  FlashInfer dump env vars are
forwarded only to the head node (rank 0) — workers don't dump.  Passwordless
SSH between allocated nodes is required (standard in SLURM environments).

```bash
DUMP_DIR=/tmp/flashinfer_dumps_<name>
TRACE_DIR=<trace_dir>
LOG=/tmp/bench_serving_<name>.log

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
    python examples/sglang_bench/bench_serving.py \
      --model <model-key> \
      --model-path <model-path> \
      --batch-sizes $BS \
      --num-batches 4 \
      --disable-cuda-graph \
      2>&1 | tee -a $LOG

  # Kill any lingering processes (including remote workers if multi-node)
  pkill -f bench_serving.py || true
  pkill -f sglang.launch_server || true

  # Sanitize (no --replace so workloads from previous batch sizes are kept).
  # See /sanitize-fi-dumps for plan/run pairing details.
  conda run -n flashinfer_bench python scripts/sanitize_dumps.py \
    --dump-dir $DUMP_DIR \
    --definitions <def_name> \
    --flashinfer-trace-dir $TRACE_DIR

  # Delete dumps before next run to free disk space
  rm -rf $DUMP_DIR
done
```

#### Multi-node flags (bench_serving.py)

| Flag | Default | Purpose |
|------|---------|---------|
| `--peer-node-addr HOST [HOST ...]` | auto (SLURM) | Override peer node hostname(s)/IP(s) |
| `--dist-init-port PORT` | `20010` | PyTorch dist rendezvous port (must differ from `--port`) |
| `--no-multinode` | off | Disable auto multi-node even when config TP > local GPUs |
| `--conda-env ENV` | `$CONDA_DEFAULT_ENV` | Fallback: conda env for peer SSH (only used when `sys.executable` is a relative path, i.e. non-NFS clusters) |

**Peer worker SSH environment**: on NFS-shared clusters (e.g. SLURM where
`/home` is shared), `bench_serving.py` passes `sys.executable` (absolute path)
as the Python binary on the remote node — no conda activation needed.  It also
passes:

- `CUDA_HOME=<conda_prefix>`: `deep_gemm` checks this first; conda prefix
  contains nvcc/include/lib.
- `PATH=<conda_bin>:/usr/...`: JIT build tools like `ninja` (needed by SGLang's
  JIT kernels) are found.
- `CPATH=<nvidia_include_dirs>`: pip-installed nvidia packages
  (`nvidia-cuda-nvrtc-cu13` etc.) put `nvrtc.h` and other CUDA headers under
  `site-packages/nvidia/cuXX/include`.  These are not on the system compiler's
  default include path, but FlashInfer's TRT-LLM FMHA JIT kernel (`fmha_gen`)
  `#includes <nvrtc.h>`.  Without CPATH, the peer node's JIT compilation fails
  with `nvrtc.h: No such file or directory`.  The head node's SGLang server also
  gets CPATH set via the `env` dict passed to `popen_launch_server`.
- `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_jit_peer`: redirects the peer
  node's FlashInfer JIT cache to local `/tmp`.  Without this, both head and
  peer nodes try to acquire the same NFS-path `FileLock` (at
  `~/.cache/flashinfer/.../cached_ops/tmp/fmha_gen.lock`), causing
  `OSError: [Errno 116] Stale file handle` on the peer.

SSH sessions don't inherit the conda env's `CUDA_HOME`, `PATH`, or `CPATH`, and
csh/tcsh don't support inline `KEY=value cmd` syntax, so `env` (an external
binary) is used for portability.  The `--conda-env` flag is only a fallback for
non-NFS setups where `sys.executable` resolves to a relative path.

**Pre-compile TRT-LLM FMHA on head node before first run** (optional but speeds
up server startup):

```bash
# Warm the NFS JIT cache so head-node TP ranks don't all compile simultaneously
CPATH="$(python -c "import sys,pathlib; print(':'.join(str(p) for p in pathlib.Path(sys.prefix).glob('lib/python*/site-packages/nvidia/cu*/include')))")" \
  python -c "from flashinfer.prefill import get_trtllm_gen_fmha_module; get_trtllm_gen_fmha_module()"
```

**Two independent decisions**:

1. **What TP to use** — always from model config:
   - Config has `--tp-size N` → use N as-is.
   - Config has no `--tp-size` → fall back to local GPU count.
   - **Config TP is never overridden.**

2. **How many nodes to use** — derived from TP and local GPU count:
   - `needed_nodes = ceil(config_tp / local_gpus)`.
   - Only launch multi-node if `needed_nodes > 1` — peer nodes are looked up
     from SLURM/`--peer-node-addr` only when needed.

**Examples** (4 GPUs per node, 2-node SLURM allocation):

| Config TP | needed_nodes | Launch mode |
|-----------|-------------|-------------|
| `--tp-size 4` | 1 | Single-node, TP=4 (fits on one node) |
| `--tp-size 8` | 2 | Multi-node, TP=8 (4 GPUs × 2 nodes) |
| *(no flag)* | 1 | Single-node, TP=4 (local GPU count fallback) |

**Manual peer override** (when not in SLURM or peer auto-detection fails):

```bash
SLURM_JOB_NODELIST=head-node,peer-node \
python examples/sglang_bench/bench_serving.py \
  --model deepseek-v3 --model-path /nfs/path/to/model \
  --peer-node-addr peer-node \
  --dist-init-port 20010 \
  ...
```

**Model path must be NFS-accessible on all nodes**: `--model-path` is passed
directly to the SGLang server on each node.  Local paths (e.g. `/data/...`)
that exist only on one node will cause `HFValidationError` or `FileNotFoundError`
on the peer.  Use NFS-shared paths (e.g. `/home/user/models/...`) or a shared
file system mount that exists on all allocated nodes.

**Note**: omit `--replace` during the loop so each batch size's workloads are
appended.  Use `--replace` only on the first run if you want to start fresh.

**FLASHINFER_DUMP_INCLUDE pattern** — always include both `plan*` and `run*`:

- For paged prefill:
  `"BatchPrefillWithPagedKVCacheWrapper.plan*,BatchPrefillWithPagedKVCacheWrapper.run*"`.
- For paged decode:
  `"BatchDecodeWithPagedKVCacheWrapper.plan*,BatchDecodeWithPagedKVCacheWrapper.run*"`.

Including `plan*` alongside `run*` is required so `sanitize_dumps.py` can pair
each `run()` dump with its `plan()` dump — this enables automatic skipping of
const-axis checks on tensors not captured in the `run()` dump (k_cache, v_cache).
If you only captured `run*` dumps, add `--skip-const-axis-check` to
`sanitize_dumps.py`.  Full pairing logic is in
[`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md).

### 3. Post-process dumps with sanitize_dumps.py

Owned by [`/sanitize-fi-dumps`](../sanitize-fi-dumps/SKILL.md) — see that skill
for the full command, flags, and tensor storage policy.  Quick reference:

```bash
conda run -n flashinfer_bench python scripts/sanitize_dumps.py \
  --dump-dir $DUMP_DIR \
  --definitions <def_name> \
  --flashinfer-trace-dir <trace_dir> \
  --replace
```

**If sanitize reports 0 workloads with const axis warnings**, a definition
constant is wrong.  Diagnose and fix with:

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

1. Fix the const axis value (e.g. `head_dim: 64 → 128`).
2. Rename the definition file/name to reflect the corrected value (e.g. `_d64_`
   → `_d128_`).
3. Fix the `assert` in the `reference` implementation.
4. Rename the workload `.jsonl` and blob dir to match the new definition name.
5. Delete the old (wrong) definition JSON.
6. Update any sibling definitions for the same model (e.g. `_ps1` / `_ps64`
   variants share the same `head_dim`).
7. Re-run `sanitize_dumps.py` with the corrected definition name.

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
git commit -m "workloads: SGLang-collected <def_name> via bench_serving.py"
git push origin HEAD:refs/pr/<pr2_num> --force
```

Post the bench_serving.py log to the HF PR2 discussion under
`## SGLang Collection Log`.

## Notes

- For `--enable-deterministic-inference` (paged prefill): add it to the model's
  `server_flags` in `model_configs.json`, or pass it via `bench_serving.py`'s
  `server_args` list.
- `bench_serving.py` passes `**os.environ` to the server process, so all
  `FLASHINFER_*` vars set in the outer shell are inherited by the SGLang
  server automatically.  This applies equally to level-3 and level-10 vars.
