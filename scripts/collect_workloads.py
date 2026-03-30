#!/usr/bin/env python3
"""
Collect real-world workloads by running SGLang inference (or direct FlashInfer API calls)
with FlashInfer Level 10 logging, then sanitizing the tensor dumps.

Two modes:
  sglang  - Launch SGLang server with a real model and run ShareGPT inference
  direct  - Call FlashInfer APIs directly with synthetic inputs (for testing)

Usage:
    # Direct mode: collect GDN MTP workloads without a model
    python collect_workloads.py direct \
        --definitions gdn_mtp_qk4_v8_d128_k_last \
        --definitions gdn_mtp_qk4_v8_d128_k_last \
        --replace

    # SGLang mode: run inference to collect workloads
    python collect_workloads.py sglang \
        --model-path /path/to/model \
        --definitions mla_paged_decode_h16_ckv512_kpe64_ps1 rmsnorm_h7168 \
        --num-samples 100

    # Direct mode: collect by op_type
    python collect_workloads.py direct \
        --op-type gdn
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_definitions(
    trace_dir: Path, def_names: list[str] | None, op_type: str | None
) -> list[Path]:
    """Resolve definition files from names or op_type."""
    def_files = []
    if op_type:
        op_dir = trace_dir / "definitions" / op_type
        if not op_dir.exists():
            print(f"ERROR: op_type directory not found: {op_dir}", file=sys.stderr)
            sys.exit(1)
        def_files += sorted(op_dir.glob("*.json"))

    if def_names:
        for name in def_names:
            found = list((trace_dir / "definitions").glob(f"**/{name}.json"))
            if not found:
                print(f"WARNING: definition not found: {name}", file=sys.stderr)
            else:
                def_files += found

    # Deduplicate preserving order
    seen = set()
    result = []
    for f in def_files:
        k = str(f)
        if k not in seen:
            seen.add(k)
            result.append(f)
    return result


_INT_DTYPES = {"int32", "int64"}


def _build_fi_include_pattern(def_files: list[Path]) -> str:
    """Parse fi_api tags from definition files to build FLASHINFER_DUMP_INCLUDE.

    For wrapper class APIs (uppercase last component), always includes .run.
    Also includes .plan when ANY definition with that API has int32/int64 inputs
    (those structural tensors come from plan(), not run()).
    For plain function APIs (lowercase last component), uses the function name directly.
    """
    from collections import defaultdict

    # Map each API dotted path → list of definition dicts that reference it
    api_to_defs: dict[str, list[dict]] = defaultdict(list)
    for path in def_files:
        with open(path) as f:
            defn = json.load(f)
        for tag in defn.get("tags", []):
            if tag.startswith("fi_api:"):
                api_to_defs[tag[len("fi_api:") :]].append(defn)

    if not api_to_defs:
        return ""

    patterns: list[str] = []
    for api in sorted(api_to_defs):
        last = api.split(".")[-1]
        if last[0].isupper():
            # Wrapper class: always capture .run
            patterns.append(f"{last}.run")
            # Also capture .plan if any definition has int32/int64 inputs
            # (structural tensors like indptrs/indices come from plan(), not run())
            needs_plan = any(
                any(
                    inp.get("shape") is not None and inp.get("dtype") in _INT_DTYPES
                    for inp in defn.get("inputs", {}).values()
                )
                for defn in api_to_defs[api]
            )
            if needs_plan:
                patterns.append(f"{last}.plan")
        else:
            # Plain function: match by function name
            patterns.append(last)

    # Deduplicate while preserving order
    seen: set[str] = set()
    return ",".join(p for p in patterns if not (p in seen or seen.add(p)))


# ──────────────────────────────────────────────────────────────────────────────
# Direct mode: call FlashInfer APIs with synthetic inputs
# ──────────────────────────────────────────────────────────────────────────────

# Default var-axis value sets per op_type.  Each entry is a list of dicts
# mapping axis name → value.  Only axes present in the definition are used;
# extras are silently ignored.  Constraints from the definition JSON (e.g.
# "seq_len > 1") are enforced automatically before calling the API.
OP_TYPE_VAR_CONFIGS: dict[str, list[dict]] = {
    "gdn": [
        {"batch_size": 1, "seq_len": 1, "pool_size": 1},
        {"batch_size": 1, "seq_len": 2, "pool_size": 1},
        {"batch_size": 1, "seq_len": 4, "pool_size": 1},
        {"batch_size": 4, "seq_len": 1, "pool_size": 4},
        {"batch_size": 4, "seq_len": 2, "pool_size": 4},
        {"batch_size": 8, "seq_len": 1, "pool_size": 8},
        {"batch_size": 8, "seq_len": 2, "pool_size": 8},
        {"batch_size": 8, "seq_len": 4, "pool_size": 8},
        {"batch_size": 16, "seq_len": 1, "pool_size": 16},
        {"batch_size": 16, "seq_len": 2, "pool_size": 16},
        {"batch_size": 32, "seq_len": 1, "pool_size": 32},
        {"batch_size": 32, "seq_len": 2, "pool_size": 32},
        {"batch_size": 64, "seq_len": 1, "pool_size": 64},
        {"batch_size": 64, "seq_len": 2, "pool_size": 64},
        {"batch_size": 1, "seq_len": 4, "pool_size": 49},
        {"batch_size": 4, "seq_len": 4, "pool_size": 49},
        {"batch_size": 8, "seq_len": 4, "pool_size": 49},
        {"batch_size": 16, "seq_len": 4, "pool_size": 49},
        {"batch_size": 32, "seq_len": 4, "pool_size": 49},
        {"batch_size": 2, "seq_len": 16, "pool_size": 4},
        {"batch_size": 4, "seq_len": 8, "pool_size": 8},
    ],
    "gqa_paged": [
        {"batch_size": 1, "num_pages": 16},
        {"batch_size": 4, "num_pages": 64},
        {"batch_size": 8, "num_pages": 128},
        {"batch_size": 16, "num_pages": 256},
        {"batch_size": 32, "num_pages": 512},
    ],
    "gqa_ragged": [
        {"batch_size": 1, "seq_len": 16},
        {"batch_size": 4, "seq_len": 128},
        {"batch_size": 8, "seq_len": 512},
        {"batch_size": 16, "seq_len": 64},
    ],
    "mla_paged": [
        {"batch_size": 1, "num_pages": 16},
        {"batch_size": 4, "num_pages": 64},
        {"batch_size": 8, "num_pages": 128},
        {"batch_size": 16, "num_pages": 256},
    ],
    "rmsnorm": [
        {"batch_size": 1, "seq_len": 1},
        {"batch_size": 4, "seq_len": 32},
        {"batch_size": 16, "seq_len": 128},
        {"batch_size": 64, "seq_len": 512},
    ],
    "gemm": [{"batch_size": 1}, {"batch_size": 8}, {"batch_size": 32}, {"batch_size": 128}],
    "sampling": [{"batch_size": 1}, {"batch_size": 8}, {"batch_size": 32}, {"batch_size": 128}],
    # fallback for unknown op_types
    "_default": [{"batch_size": 1}, {"batch_size": 4}, {"batch_size": 16}],
}

_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
    "float16": "torch.float16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "bool": "torch.bool",
}


def _check_constraint(constraint: str, axes: dict) -> bool:
    """Evaluate a simple axis constraint expression (e.g. 'seq_len > 1')."""
    try:
        return bool(eval(constraint, {}, axes))  # noqa: S307
    except Exception:
        return True  # unknown constraint → don't filter


def _build_generator_code(
    defn: dict, var_configs: list[dict], dump_dir: Path, include_pattern: str
) -> str:
    """
    Build a self-contained Python script that sets FlashInfer env vars before
    any import, generates synthetic inputs for each var-axis combo, and calls
    the API.  Must be executed in a subprocess so env vars are read at import.
    """
    fi_api = next(
        (t[len("fi_api:") :] for t in defn.get("tags", []) if t.startswith("fi_api:")), None
    )
    if not fi_api:
        return ""

    module_path, func_name = fi_api.rsplit(".", 1)

    # Resolve const axes
    const_axes = {
        name: spec["value"]
        for name, spec in defn.get("axes", {}).items()
        if spec.get("type") == "const"
    }
    var_axis_names = [
        name for name, spec in defn.get("axes", {}).items() if spec.get("type") != "const"
    ]
    constraints = defn.get("constraints", [])

    # Build list of concrete axis-value dicts, honouring constraints
    concrete_combos = []
    for combo in var_configs:
        axes = {**const_axes, **{k: combo.get(k, 1) for k in var_axis_names}}
        if all(_check_constraint(c, axes) for c in constraints):
            concrete_combos.append(axes)

    if not concrete_combos:
        return ""

    # Build input-generation code for each input tensor
    input_lines = []
    for inp_name, inp_spec in defn.get("inputs", {}).items():
        if inp_spec.get("optional"):
            continue
        shape = inp_spec.get("shape")
        dtype_str = inp_spec.get("dtype", "float32")
        torch_dtype = _DTYPE_MAP.get(dtype_str, "torch.float32")

        if shape is None:
            # Scalar — use 0.0 (scale=0 triggers 1/sqrt(head_size) default)
            input_lines.append(
                f"    {inp_name} = torch.tensor(0.0, dtype={torch_dtype}, device=device)"
            )
        else:
            shape_expr = (
                "["
                + ", ".join(f'axes["{d}"]' if isinstance(d, str) else str(d) for d in shape)
                + "]"
            )
            is_int = torch_dtype in ("torch.int32", "torch.int64")
            if is_int:
                # Index tensors: fill with arange % pool_size (or zeros if no pool)
                input_lines.append(
                    f"    _shape = {shape_expr}\n"
                    f"    {inp_name} = (torch.arange(_shape[0], dtype={torch_dtype}, device=device)"
                    f' % axes.get("pool_size", 1)).reshape(_shape) if len(_shape) == 1 '
                    f"else torch.zeros(_shape, dtype={torch_dtype}, device=device)"
                )
            else:
                input_lines.append(
                    f"    {inp_name} = torch.randn({shape_expr}, dtype={torch_dtype}, device=device)"
                )

    input_block = "\n".join(input_lines)
    kwarg_names = [n for n in defn.get("inputs", {}) if not defn["inputs"][n].get("optional")]
    kwargs_str = ", ".join(f"{n}={n}" for n in kwarg_names)

    combos_repr = repr(concrete_combos)

    return f"""\
import os, sys
os.environ["FLASHINFER_LOGLEVEL"] = "10"
os.environ["FLASHINFER_DUMP_DIR"] = {str(dump_dir)!r}
os.environ["FLASHINFER_DUMP_SAFETENSORS"] = "1"
os.environ["FLASHINFER_DUMP_INCLUDE"] = {include_pattern!r}
os.environ["FLASHINFER_DUMP_EXCLUDE"] = "*.__init__"
os.environ["FLASHINFER_DUMP_MAX_COUNT"] = "10000"
import torch
from {module_path} import {func_name}

device = "cuda" if torch.cuda.is_available() else "cpu"
combos = {combos_repr}
print(f"Generating {{len(combos)}} calls for {defn['name']}")
for axes in combos:
{input_block}
    try:
        result = {func_name}({kwargs_str})
        print(f"  OK  axes={{dict((k,v) for k,v in axes.items() if isinstance(v,int) and k not in {set(const_axes)})}}")
    except Exception as exc:
        print(f"  ERR {{exc}}")
"""


def run_direct_mode(def_files: list[Path], trace_dir: Path, dump_dir: Path, replace: bool) -> None:
    """Run direct API call mode: spawn a subprocess per definition with FlashInfer logging env."""
    include_pattern = _build_fi_include_pattern(def_files)
    if not include_pattern:
        print("WARNING: No fi_api tags found in any definition; capturing all calls")
        include_pattern = "*"

    print(f"\nPhase 2: FlashInfer Logging Configuration")
    print(f"  FLASHINFER_LOGLEVEL=10")
    print(f"  FLASHINFER_DUMP_DIR={dump_dir}")
    print(f"  FLASHINFER_DUMP_SAFETENSORS=1")
    print(f"  FLASHINFER_DUMP_INCLUDE={include_pattern}")

    print(f"\nPhase 3: Direct API Call Generation")
    for df in def_files:
        with open(df) as fh:
            defn = json.load(fh)

        op_type = defn.get("op_type", "_default")
        var_configs = OP_TYPE_VAR_CONFIGS.get(op_type, OP_TYPE_VAR_CONFIGS["_default"])

        code = _build_generator_code(defn, var_configs, dump_dir, include_pattern)
        if not code:
            print(f"  SKIP {defn['name']}: no fi_api tag")
            continue

        print(f"  Generating workloads for {defn['name']} ({op_type})")
        result = subprocess.run([sys.executable, "-c", code], capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: generator subprocess failed (rc={result.returncode})")
            sys.exit(1)

    print(f"\nPhase 4: Sanitizing Tensor Dumps")

    # Run sanitization
    sanitize_script = Path(__file__).parent / "sanitize_dumps.py"
    def_names = [f.stem for f in def_files]
    cmd = [
        sys.executable,
        str(sanitize_script),
        "--dump-dir",
        str(dump_dir),
        "--definitions",
        *def_names,
        "--flashinfer-trace-dir",
        str(trace_dir),
    ]
    if replace:
        cmd.append("--replace")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: sanitization failed", file=sys.stderr)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# SGLang mode: run inference with real model
# ──────────────────────────────────────────────────────────────────────────────


def _tp_ep_from_definitions(def_files: list[Path]) -> tuple[int, int]:
    """Extract tp and ep values from definition tags.

    Reads all definition JSONs, collects tp:N and ep:N tags, verifies they
    are consistent across definitions, and returns (tp, ep). Defaults to (1, 1).
    """
    tp_values, ep_values = set(), set()
    for f in def_files:
        try:
            defn = json.loads(f.read_text())
        except Exception:
            continue
        for tag in defn.get("tags", []):
            if tag.startswith("tp:"):
                try:
                    tp_values.add(int(tag.split(":", 1)[1]))
                except ValueError:
                    pass
            elif tag.startswith("ep:"):
                try:
                    ep_values.add(int(tag.split(":", 1)[1]))
                except ValueError:
                    pass

    if len(tp_values) > 1:
        print(
            f"WARNING: conflicting tp values across definitions: {tp_values}. Using max={max(tp_values)}"
        )
    if len(ep_values) > 1:
        print(
            f"WARNING: conflicting ep values across definitions: {ep_values}. Using max={max(ep_values)}"
        )

    tp = max(tp_values) if tp_values else 1
    ep = max(ep_values) if ep_values else 1
    return tp, ep


def _uses_paged_prefill(def_files: list[Path]) -> bool:
    """Return True if any definition uses BatchPrefillWithPagedKVCacheWrapper."""
    for path in def_files:
        with open(path) as f:
            defn = json.load(f)
        for tag in defn.get("tags", []):
            if "BatchPrefillWithPagedKVCacheWrapper" in tag:
                return True
    return False


def _run_sglang_offline_batched(
    model_path: str,
    tp: int,
    page_size: int,
    env: dict,
    dump_dir: Path,
    quantization: str | None = None,
    cpu_offload_gb: float = 0.0,
) -> None:
    """Run SGLang offline Engine in a subprocess to collect decode workloads with controlled batch sizes.

    Spawns a subprocess that:
      1. Sets FlashInfer logging env vars before any import (so TP workers inherit them)
      2. Creates sglang.Engine and calls engine.generate() for each (batch_size, prompt_tokens) round
      3. Each generate() call is a static batch → decode sees exactly batch_size=B sequences

    Rounds cover batch_size ∈ {1, 2, 4, 8, 16, 32, 64} with short/medium/long prompts
    to produce diverse (batch_size, num_kv_indices) workload combinations.
    """
    # Controlled (batch_size, prompt_tokens, max_tokens) rounds.
    # Prompt tokens control how many KV pages are occupied per sequence.
    # max_tokens=8 keeps each round fast while still providing kv_len diversity.
    # With 3 prompt lengths per batch_size and 8 decode steps each, we get
    # 24 distinct (batch_size, kv_len) combos per batch_size — enough for
    # the deduplication/diversity filter to select 20 representative entries.
    rounds = [
        (1, 50, 8),
        (1, 300, 8),
        (1, 800, 8),
        (2, 50, 8),
        (2, 300, 8),
        (2, 800, 8),
        (4, 50, 8),
        (4, 300, 8),
        (4, 800, 8),
        (8, 50, 8),
        (8, 300, 8),
        (8, 800, 8),
        (16, 50, 8),
        (16, 300, 8),
        (16, 800, 8),
        (32, 50, 8),
        (32, 300, 8),
        (64, 50, 8),
    ]

    # Write a self-contained script that is launched as a subprocess.
    # All FlashInfer env vars must be set before sglang is imported so that
    # the TP worker processes inherit them from the parent environment.
    # IMPORTANT: Must use `if __name__ == '__main__':` guard because SGLang uses
    # the `spawn` multiprocessing start method, which re-imports the main script
    # in each worker process. Without this guard the workers would also try to
    # create engines, triggering the "bootstrapping phase" RuntimeError.
    script = r"""
import os, sys, json, time

# ── FlashInfer logging setup (must come before any sglang/flashinfer import) ──
# These env vars are set here (at module level, before any import) so that
# TP worker processes spawned by sglang inherit them.
_env = json.loads(sys.argv[1])
for k, v in _env.items():
    os.environ[k] = v

if __name__ == '__main__':
    model_path  = sys.argv[2]
    tp          = int(sys.argv[3])
    page_size   = int(sys.argv[4])
    rounds      = json.loads(sys.argv[5])
    quant       = sys.argv[6] if sys.argv[6] != "none" else None
    cpu_offload = float(sys.argv[7])

    # ── SGLang offline engine ─────────────────────────────────────────────────
    import sglang

    engine_kwargs = dict(
        model_path=model_path,
        tp_size=tp,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        page_size=page_size,
        log_level="info",
    )
    if quant:
        engine_kwargs["quantization"] = quant
    if cpu_offload > 0:
        engine_kwargs["cpu_offload_gb"] = cpu_offload

    engine = sglang.Engine(**engine_kwargs)

    _BASE = (
        "You are an expert in GPU kernel optimization, CUDA programming, and "
        "transformer inference systems. Please provide a detailed technical "
        "explanation covering the following topic: "
    )
    _TOPICS = [
        "paged KV cache and memory fragmentation reduction",
        "tensor parallelism for multi-head attention",
        "FlashAttention tiling and IO-aware computation",
        "mixture-of-experts routing and load balancing",
        "RMSNorm versus LayerNorm computational differences",
        "speculative decoding draft model verification",
        "continuous batching in LLM serving systems",
        "quantization tradeoffs FP8 INT4 and GPTQ",
        "GQA grouped query attention and KV head sharing",
        "paged attention and vLLM memory management",
        "CUDA warp-level parallelism and shared memory",
        "multi-head latent attention MLA KV compression",
        "gated delta networks linear attention recurrence",
        "expert parallelism for MoE inference",
        "prefix caching and radix attention",
        "chunked prefill for decode-prefill balance",
        "ring attention and sequence parallelism",
        "FlashInfer batch decode wrapper plan and run",
    ]

    def _make_prompt(approx_tokens: int, idx: int) -> str:
        topic = _TOPICS[idx % len(_TOPICS)]
        text = _BASE + topic
        # Repeat to reach approximate token count (~6 chars/token heuristic)
        target_chars = approx_tokens * 6
        while len(text) < target_chars:
            text += " " + topic
        return text[:target_chars]

    prompt_idx = 0
    try:
        for B, prompt_tokens, max_tokens in rounds:
            prompts = [_make_prompt(prompt_tokens, prompt_idx + i) for i in range(B)]
            prompt_idx += B
            t0 = time.time()
            outputs = engine.generate(
                prompt=prompts,
                sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0},
            )
            elapsed = time.time() - t0
            n_ok = len(outputs) if outputs else 0
            print(f"  batch_size={B:2d}, prompt~{prompt_tokens:4d}t, max_tokens={max_tokens}: {n_ok}/{B} ok ({elapsed:.1f}s)", flush=True)
    finally:
        engine.shutdown()
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    # Pass FlashInfer env vars as JSON so the subprocess sets them before importing sglang
    fi_env_keys = [
        "FLASHINFER_LOGLEVEL",
        "FLASHINFER_DUMP_DIR",
        "FLASHINFER_DUMP_SAFETENSORS",
        "FLASHINFER_DUMP_INCLUDE",
        "FLASHINFER_DUMP_EXCLUDE",
        "FLASHINFER_DUMP_MAX_COUNT",
        "FLASHINFER_DUMP_MAX_SIZE_GB",
        "FLASHINFER_USE_CUDA_NORM",
    ]
    fi_env_json = json.dumps({k: env[k] for k in fi_env_keys if k in env})

    cmd = [
        sys.executable,
        script_path,
        fi_env_json,
        model_path,
        str(tp),
        str(page_size),
        json.dumps(rounds),
        quantization or "none",
        str(cpu_offload_gb),
    ]
    subprocess.run(cmd, check=True, env=env)

    import os as _os

    _os.unlink(script_path)


def run_sglang_mode(
    model_path: str,
    def_files: list[Path],
    trace_dir: Path,
    dump_dir: Path,
    num_samples: int,
    dataset_path: str | None,
    tp: int,
    replace: bool,
    quantization: str | None = None,
    cpu_offload_gb: float = 0.0,
    ep: int = 1,
    page_size: int = 1,
    skip_const_axis_check: bool = False,
) -> None:
    """Run SGLang inference with FlashInfer logging to collect workloads."""
    include_pattern = _build_fi_include_pattern(def_files)
    if not include_pattern:
        include_pattern = "*"

    # Paged prefill (BatchPrefillWithPagedKVCacheWrapper) requires --enable-deterministic-inference
    # so SGLang sets use_ragged=False and routes all prefill through the paged wrapper.
    force_paged_prefill = _uses_paged_prefill(def_files)
    if force_paged_prefill:
        print("  Detected paged prefill definition — will use --enable-deterministic-inference")

    env = os.environ.copy()
    env["FLASHINFER_LOGLEVEL"] = "10"
    env["FLASHINFER_DUMP_DIR"] = str(dump_dir)
    env["FLASHINFER_DUMP_SAFETENSORS"] = "1"
    env["FLASHINFER_DUMP_INCLUDE"] = include_pattern
    # Exclude only constructors; .plan is included selectively via FLASHINFER_DUMP_INCLUDE
    env["FLASHINFER_DUMP_EXCLUDE"] = "*.__init__"
    # Each batch-size round emits ~2 decode dumps per worker (1 plan + 1-8 run).
    # 18 rounds × 10 × 2 workers ≈ 360 — set a generous cap.
    env["FLASHINFER_DUMP_MAX_COUNT"] = "50000"
    env["FLASHINFER_DUMP_MAX_SIZE_GB"] = "30"
    # Use pre-compiled CUDA norm kernels — CuTe DSL norm requires CUDA toolkit 13.1+
    env["FLASHINFER_USE_CUDA_NORM"] = "1"
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    print(f"\nPhase 2: FlashInfer Logging Configuration")
    print(f"  FLASHINFER_DUMP_INCLUDE={include_pattern}")
    print(f"  FLASHINFER_DUMP_DIR={dump_dir}")

    print(f"\nPhase 3: SGLang Inference Execution")
    desc = f"tp={tp}"
    if ep > 1:
        desc += f", ep={ep}"
    if quantization:
        desc += f", quant={quantization}"

    # For decode-only collections (no paged prefill definitions), use the SGLang
    # offline Engine instead of the HTTP server.  The offline Engine processes each
    # call to engine.generate() as a single static batch, which guarantees that the
    # decode phase sees exactly batch_size=B concurrent sequences.
    # The HTTP server's continuous-batching scheduler processes requests one at a
    # time and does not reliably produce multi-sequence decode batches for small B.
    use_offline = not force_paged_prefill
    if use_offline:
        print(
            f"  Using SGLang offline Engine (decode-only, batch-controlled) — model={model_path}, {desc}"
        )
        _run_sglang_offline_batched(
            model_path,
            tp,
            page_size,
            env,
            dump_dir,
            quantization=quantization,
            cpu_offload_gb=cpu_offload_gb,
        )
    else:
        _server_port = int(os.environ.get("SGLANG_PORT", "30000"))
        print(f"  Launching SGLang server (model={model_path}, {desc})")
        # Start SGLang server
        server_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(_server_port),
            "--tp",
            str(tp),
            "--attention-backend",
            "flashinfer",
            "--disable-cuda-graph",
            "--log-level",
            "info",
            "--page-size",
            str(page_size),
            "--max-running-requests",
            "256",
            # Force use_ragged=False so all prefill goes through BatchPrefillWithPagedKVCacheWrapper
            "--enable-deterministic-inference",
        ]
        if ep > 1:
            server_cmd += ["--ep", str(ep)]
        if quantization:
            server_cmd += ["--quantization", quantization]
        if cpu_offload_gb > 0:
            server_cmd += ["--cpu-offload-gb", str(int(cpu_offload_gb))]

        server_log_path = dump_dir / "sglang_server.log"
        server_log_file = open(server_log_path, "w")
        print(f"  SGLang server log: {server_log_path}  (tail -f {server_log_path})")
        server_proc = subprocess.Popen(
            server_cmd, env=env, stdout=server_log_file, stderr=server_log_file
        )
        print("  Waiting for server to start (polling health endpoint)...")
        import requests as _requests

        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(10)
            if server_proc.poll() is not None:
                raise RuntimeError("SGLang server exited unexpectedly during startup")
            try:
                r = _requests.get(f"http://localhost:{_server_port}/health", timeout=5)
                if r.status_code == 200:
                    print("  Server is ready!")
                    break
            except Exception:
                pass
        else:
            server_proc.terminate()
            server_log_file.close()
            raise RuntimeError("SGLang server did not become ready within 30 minutes")

        try:
            _run_sglang_inference(num_samples, dataset_path, paged_prefill=True)
        finally:
            server_proc.terminate()
            server_proc.wait()
            server_log_file.close()
            print("  Server shutdown")

    print(f"\nPhase 4: Sanitizing Tensor Dumps")
    sanitize_script = Path(__file__).parent / "sanitize_dumps.py"
    def_names = [f.stem for f in def_files]
    cmd = [
        sys.executable,
        str(sanitize_script),
        "--dump-dir",
        str(dump_dir),
        "--definitions",
        *def_names,
        "--flashinfer-trace-dir",
        str(trace_dir),
    ]
    if replace:
        cmd.append("--replace")
    if skip_const_axis_check:
        cmd.append("--skip-const-axis-check")
    subprocess.run(cmd, check=True)


def _run_sglang_inference(
    num_samples: int, dataset_path: str | None, paged_prefill: bool = False
) -> None:
    """Send ShareGPT inference requests to a running SGLang server.

    Sends requests in concurrent bursts of varying sizes (1, 2, 4, 8, 16, 32)
    so that the server's continuous-batching scheduler produces decode steps
    with diverse batch sizes, not just batch_size=1.

    When paged_prefill=True, uses a prefix-sharing strategy to generate diverse
    workloads: a long shared system prompt is prepended to every request, producing
    varied (num_tokens, prefix_len) pairs for the paged prefill wrapper.
    """
    import threading

    import requests

    # Load dataset
    conversations = []
    if dataset_path:
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conversations.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        # Try to load from HuggingFace datasets
        try:
            from datasets import load_dataset

            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
            for item in ds:
                conversations.append(item)
        except Exception as e:
            print(f"  Warning: could not load ShareGPT dataset ({e}), using synthetic prompts")
            # Generate synthetic prompts of varying lengths to exercise different workload shapes
            _TOPICS = [
                "Explain the transformer attention mechanism in detail, covering query key value projections, softmax attention, and multi-head attention.",
                "Write a complete Python implementation of a binary search tree with insert, delete, and traversal methods.",
                "What are the main differences between CUDA and ROCm? Discuss kernel compilation, memory management, and ecosystem support.",
                "Describe the architecture of a modern large language model, covering tokenization, embeddings, transformer layers, and output projection.",
                "How does gradient checkpointing reduce GPU memory usage? Explain the tradeoff with recomputation overhead.",
                "Explain paged KV cache in LLM serving systems. How does PagedAttention work and what are its benefits?",
                "What is tensor parallelism and how does it work? Describe how attention heads and MLP weights are sharded.",
                "Write a detailed essay about the history of neural networks from perceptrons to transformers.",
                "Compare and contrast RMSNorm and LayerNorm. When should each be used and what are the computational differences?",
                "Explain the MoE (Mixture of Experts) architecture. How does routing work and what are the challenges in training and serving?",
                "What is speculative decoding and how does it speed up inference? Describe the draft model and verification process.",
                "Describe how FlashAttention reduces memory complexity. Explain tiling, recomputation, and IO-awareness.",
                "Write a tutorial on writing custom CUDA kernels for matrix operations, covering shared memory, warp synchronization, and coalesced access.",
                "Explain the Gated Delta Network linear attention mechanism. How does it compare to standard attention in terms of complexity?",
                "What are the challenges of serving very large language models? Discuss sharding strategies, quantization, and KV cache management.",
                "How does quantization affect model quality and throughput? Compare INT8, FP8, INT4, and GPTQ approaches.",
                "Describe KV cache compression techniques for LLMs, including quantization, eviction policies, and prefix sharing.",
                "What is the difference between prefill and decode in LLM serving? How does chunked prefill help balance GPU utilization?",
                "Explain multi-head latent attention (MLA) architecture from DeepSeek. How does it compress KV cache compared to GQA?",
                "How do you optimize batch size for LLM inference throughput? Discuss continuous batching, dynamic batching, and request scheduling.",
            ]
            for i in range(num_samples + 10):
                conversations.append(
                    {
                        "conversations": [
                            {"from": "human", "value": _TOPICS[i % len(_TOPICS)]},
                            {"from": "gpt", "value": ""},
                        ]
                    }
                )

    # Ensure we have enough samples; repeat if needed for burst scheduling
    while len(conversations) < num_samples:
        conversations.extend(conversations[: num_samples - len(conversations)])
    conversations = conversations[:num_samples]

    def _format_conv(conv: dict) -> list[dict] | None:
        msgs = conv.get("conversations", conv.get("messages", []))
        formatted = []
        for m in msgs:
            role = "user" if m.get("from") in ("human", "user") else "assistant"
            formatted.append({"role": role, "content": m.get("value", m.get("content", ""))})
        if not formatted:
            return None
        if formatted[0]["role"] != "user":
            formatted = [{"role": "user", "content": "Hello"}] + formatted
        return formatted[:4]  # Limit context depth

    def _send_one(msgs: list[dict], max_tokens: int = 256) -> bool:
        try:
            _port = int(os.environ.get("SGLANG_PORT", "30000"))
            resp = requests.post(
                f"http://localhost:{_port}/v1/chat/completions",
                json={
                    "model": "model",
                    "messages": msgs,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=300,
            )
            return resp.status_code == 200
        except Exception:
            return False

    all_msgs = [_format_conv(c) for c in conversations]
    all_msgs = [m for m in all_msgs if m]

    # Build prompts of varying lengths so prefill sees different token counts.
    # Repeat the last user message to create longer inputs.
    def _pad_to_tokens(msgs: list[dict], approx_tokens: int) -> list[dict]:
        """Repeat the user content until it reaches ~approx_tokens words (proxy for tokens)."""
        if not msgs:
            return msgs
        base = msgs[-1]["content"]
        repeated = (base + " ") * max(1, approx_tokens // max(len(base.split()), 1))
        padded = list(msgs)
        padded[-1] = dict(padded[-1])
        padded[-1]["content"] = repeated[: approx_tokens * 6]  # ~6 chars/token
        return padded

    # Send all requests concurrently with varying prompt lengths.
    # As requests finish at different rates, the server's decode batch size
    # naturally decreases from N down to 1, yielding diverse batch_size samples.
    # Groups of requests share the same prompt length so prefill shapes also vary.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if paged_prefill:
        # Paged prefill collection strategy:
        # Send requests in sequential rounds where each round shares the same long system
        # prompt. This populates the radix cache with the shared prefix, so subsequent
        # requests within the same round have extend_prefix_lens > 0 and are routed through
        # BatchPrefillWithPagedKVCacheWrapper (paged extend path).
        #
        # Each round uses a different shared prefix length to exercise diverse
        # (num_extend_tokens, prefix_len) combinations. max_tokens=1 per request
        # since we only need the prefill dump — no decode steps required.
        _BASE_SENTENCE = "You are an expert in GPU kernel optimization, CUDA programming, FlashAttention, paged KV caches, transformer inference systems, tensor parallelism, and efficient attention algorithms. "
        # Rounds: (num_concurrent, prefix_tokens)
        # num_concurrent controls len_indptr (batch size at prefill time).
        # prefix_tokens controls num_kv_indices / num_pages.
        # Varying both gives diverse (len_indptr, total_q, num_kv_indices) combinations.
        # - Avoid too many num_concurrent=1 rounds (they all yield len_indptr=2 and crowd
        #   out larger-batch shapes after the 20-entry dedup cap).
        # - Include large batches (64, 128) for divergent problem shapes.
        _ROUNDS = [
            (1, 128),
            (1, 512),
            (2, 64),
            (2, 512),
            (2, 1024),
            (4, 64),
            (4, 256),
            (4, 768),
            (8, 32),
            (8, 256),
            (8, 1024),
            (16, 64),
            (16, 256),
            (16, 768),
            (32, 32),
            (32, 256),
            (32, 512),
            (64, 64),
            (64, 256),
            (64, 768),
            (128, 32),
            (128, 128),
            (128, 512),
        ]
        total_success = 0
        total_sent = 0
        conv_offset = 0
        for round_idx, (num_concurrent, prefix_tokens) in enumerate(_ROUNDS):
            shared_prefix = (
                _BASE_SENTENCE * max(1, prefix_tokens // max(len(_BASE_SENTENCE.split()), 1))
            )[: prefix_tokens * 6]
            system_msg = {"role": "user", "content": shared_prefix}
            msgs_batch = []
            for j in range(num_concurrent):
                base_msgs = all_msgs[(conv_offset + j) % len(all_msgs)]
                user_turn = (
                    base_msgs[-1]
                    if base_msgs
                    else {"role": "user", "content": "Please explain this concept in detail."}
                )
                # Vary query length per request for diverse total_q
                target_tokens = 20 + j * 30
                padded_user = _pad_to_tokens([user_turn], target_tokens)[-1]
                msgs_batch.append([system_msg, padded_user])
            conv_offset += num_concurrent

            print(
                f"  Round {round_idx + 1}/{len(_ROUNDS)}: {num_concurrent} concurrent requests (prefix ~{prefix_tokens} tokens)..."
            )
            with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
                # max_tokens=1: we only need the prefill dump, not decode steps
                futs = [pool.submit(_send_one, m, 1) for m in msgs_batch]
                success = sum(f.result() for f in as_completed(futs))
            total_success += success
            total_sent += num_concurrent
            print(f"    Completed: {success}/{num_concurrent} successful")

        print(f"  Total: {total_success}/{total_sent} successful requests")
    else:
        # Controlled batch-size strategy: send sequential rounds of exactly B requests,
        # one round at a time. Since all B requests in a round share the same prompt
        # length, they finish prefill together and enter decode simultaneously —
        # guaranteeing batch_size=B in the decode dumps for that round.
        #
        # We vary both batch_size and prompt_tokens across rounds to produce
        # diverse (batch_size, num_kv_indices) combinations:
        #   - batch_size  ∈ {1, 2, 4, 8, 16, 32, 64}
        #   - prompt_tokens control how many KV pages each sequence occupies
        #
        # max_tokens is kept short (128) so rounds complete quickly and the
        # decode phase is captured cleanly before the batch size drops.
        rounds = [
            # (batch_size, prompt_tokens, max_tokens)
            (1, 50, 128),
            (1, 300, 128),
            (1, 800, 128),
            (2, 50, 128),
            (2, 300, 128),
            (2, 800, 128),
            (4, 50, 128),
            (4, 300, 128),
            (4, 800, 128),
            (8, 50, 128),
            (8, 300, 128),
            (8, 800, 128),
            (16, 50, 128),
            (16, 300, 128),
            (16, 800, 128),
            (32, 50, 128),
            (32, 300, 128),
            (64, 50, 128),
        ]

        conv_idx = 0
        total_success = 0
        total_sent = 0

        for B, prompt_tokens, max_tokens in rounds:
            # All B requests use the same prompt length so they prefill together.
            msgs_group = []
            for _ in range(B):
                msgs = _pad_to_tokens(all_msgs[conv_idx % len(all_msgs)], prompt_tokens)
                msgs_group.append((msgs, max_tokens))
                conv_idx += 1

            print(
                f"  batch_size={B:2d}, prompt~{prompt_tokens:4d}t, max_tokens={max_tokens}: ",
                end="",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=B) as pool:
                futs = [pool.submit(lambda a: _send_one(a[0], a[1]), arg) for arg in msgs_group]
                success = sum(f.result() for f in as_completed(futs))
            total_success += success
            total_sent += B
            print(f"{success}/{B} ok")

        print(f"  Total: {total_success}/{total_sent} successful requests")


# ──────────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ──────────────────────────────────────────────────────────────────────────────


def run_baseline_eval(def_files: list[Path], trace_dir: Path) -> None:
    """Run the baseline (reference) solution against collected workloads.

    Uses ``flashinfer-bench run`` to evaluate the baseline solution for each
    definition and saves a per-definition trace JSONL to
    ``{trace_dir}/traces/<def_name>_baseline.jsonl``.

    Exits with a non-zero status if any workload evaluation returns a status
    other than PASSED.
    """
    import datetime

    traces_dir = trace_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True

    for def_file in def_files:
        def_name = def_file.stem
        workload_file = trace_dir / "workloads" / def_file.parent.name / f"{def_name}.jsonl"
        if not workload_file.exists():
            print(f"  WARNING: workload file not found for {def_name}, skipping eval")
            continue

        print(f"\nPhase 5: Baseline evaluation — {def_name}")
        out_trace = traces_dir / f"{def_name}_baseline.jsonl"

        cmd = [
            sys.executable,
            "-m",
            "flashinfer_bench",
            "run",
            "--local",
            str(trace_dir),
            "--definitions",
            def_name,
            "--solutions",
            "baseline",
            "--save-results",
            "--warmup-runs",
            "3",
            "--iterations",
            "20",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout + result.stderr

        # Collect trace lines from the standard trace output location
        # flashinfer-bench writes traces to {trace_dir}/traces/<def_name>_workloads.jsonl
        auto_trace = traces_dir / f"{def_name}_workloads.jsonl"

        failed = []
        passed = 0

        if auto_trace.exists():
            import json as _json

            lines = [l.strip() for l in auto_trace.read_text().splitlines() if l.strip()]
            for line in lines:
                try:
                    rec = _json.loads(line)
                    status = rec.get("evaluation", {}).get("status", "UNKNOWN")
                    if status != "PASSED":
                        failed.append({"workload": rec.get("workload", {}).get("uuid", "?"), "status": status})
                    else:
                        passed += 1
                except Exception:
                    pass

            # Copy to per-def baseline trace name
            import shutil
            shutil.copy(auto_trace, out_trace)
            print(f"  Results: {passed} PASSED, {len(failed)} FAILED")
            if failed:
                print(f"  FAILED workloads:")
                for f in failed[:10]:
                    print(f"    uuid={f['workload']} status={f['status']}")
                all_passed = False
            else:
                print(f"  All {passed} workloads PASSED ✓")
                print(f"  Trace written to: {out_trace}")
        else:
            print(f"  WARNING: No trace output found at {auto_trace}")
            print(f"  flashinfer-bench stdout:\n{stdout[:2000]}")
            if result.returncode != 0:
                all_passed = False

    if not all_passed:
        print("\nERROR: Some baseline evaluations FAILED. Check trace output above.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBaseline evaluation complete. All workloads PASSED.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _install_latest_packages() -> None:
    """
    Install the latest SGLang and FlashInfer from source.

    Pulls the latest commits from the cloned repos under tmp/ (relative to this
    script's parent directory) and installs both packages in editable mode.
    If the repos are not present, falls back to pip install -U from PyPI.
    """
    print("Phase 0: Installing latest SGLang and FlashInfer from source...")
    repo_root = Path(__file__).parent.parent
    fi_repo = repo_root / "tmp" / "flashinfer"
    sg_repo = repo_root / "tmp" / "sglang"

    if fi_repo.exists() and sg_repo.exists():
        # Pull latest commits
        for repo, name in [(fi_repo, "flashinfer"), (sg_repo, "sglang")]:
            print(f"  git pull {name}...")
            subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], capture_output=False)

        # Install FlashInfer from source (pyproject.toml is at repo root)
        print("  Installing FlashInfer from source...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(fi_repo), "--no-build-isolation"],
            capture_output=False,
        )
        if r.returncode != 0:
            print("WARNING: FlashInfer source install failed")

        # Install SGLang from source (no extras to avoid outlines_core which needs Rust)
        print("  Installing SGLang from source...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", f"{sg_repo}/python"],
            capture_output=False,
        )
        if r.returncode != 0:
            print("WARNING: SGLang source install failed")
    else:
        print("  tmp/flashinfer or tmp/sglang not found — falling back to pip install -U")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "sglang", "flashinfer-python"],
            capture_output=False,
        )

    # Print installed versions
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sglang, flashinfer; "
            "print(f'  SGLang: {sglang.__version__}, FlashInfer: {flashinfer.__version__}')",
        ],
        capture_output=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect FlashInfer workloads")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Direct mode
    direct_p = sub.add_parser("direct", help="Call FlashInfer APIs directly (no model needed)")
    direct_p.add_argument("--definitions", nargs="+", help="Definition names")
    direct_p.add_argument("--op-type", help="Op type to collect all definitions for")
    direct_p.add_argument(
        "--flashinfer-trace-dir", required=True, help="Path to flashinfer_trace directory"
    )
    direct_p.add_argument("--dump-dir", help="Override dump directory path")
    direct_p.add_argument("--replace", action="store_true", help="Replace existing workloads")

    # SGLang mode
    sglang_p = sub.add_parser("sglang", help="Run SGLang inference to collect workloads")
    sglang_p.add_argument("--model-path", required=True, help="Model path or HuggingFace repo ID")
    sglang_p.add_argument("--definitions", nargs="+", help="Definition names")
    sglang_p.add_argument("--op-type", help="Op type to collect all definitions for")
    sglang_p.add_argument(
        "--flashinfer-trace-dir", required=True, help="Path to flashinfer_trace directory"
    )
    sglang_p.add_argument("--num-samples", type=int, default=100)
    sglang_p.add_argument("--dataset", help="Path to ShareGPT JSONL dataset")
    sglang_p.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel degree (default: auto-detected from definition tags)",
    )
    sglang_p.add_argument("--quantization", help="Quantization method (e.g. fp8, awq, gptq)")
    sglang_p.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="GB of model weights to offload to CPU (e.g. 32)",
    )
    sglang_p.add_argument(
        "--page-size",
        type=int,
        default=None,
        help="KV cache page size (default: auto from definition tags, fallback 1)",
    )
    sglang_p.add_argument("--dump-dir", help="Override dump directory path")
    sglang_p.add_argument("--replace", action="store_true", help="Replace existing workloads")
    sglang_p.add_argument(
        "--skip-const-axis-check",
        action="store_true",
        help=(
            "Pass --skip-const-axis-check to sanitize_dumps.py. Use when collecting "
            "from TP=1 SGLang for a TP=2 definition (e.g. h20 from Qwen3-14B TP=1)."
        ),
    )
    sglang_p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip Phase 0 package install (use when env already has correct versions)",
    )
    direct_p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip Phase 0 package install (use when env already has correct versions)",
    )
    sglang_p.add_argument(
        "--eval-baseline",
        action="store_true",
        default=True,
        help=(
            "After collecting workloads, run the baseline solution against them using "
            "'flashinfer-bench run' and write per-workload trace JSONL to "
            "{trace_dir}/traces/<def_name>_baseline.jsonl. All workloads must PASS. "
            "Enabled by default; use --no-eval-baseline to skip."
        ),
    )
    sglang_p.add_argument(
        "--no-eval-baseline",
        dest="eval_baseline",
        action="store_false",
        help="Skip baseline evaluation after workload collection.",
    )
    direct_p.add_argument(
        "--eval-baseline",
        action="store_true",
        default=True,
        help="After collecting workloads, run baseline evaluation (enabled by default).",
    )
    direct_p.add_argument(
        "--no-eval-baseline",
        dest="eval_baseline",
        action="store_false",
        help="Skip baseline evaluation after workload collection.",
    )

    args = parser.parse_args()

    if not getattr(args, "skip_install", False):
        _install_latest_packages()
    else:
        print("Phase 0: Skipping package install (--skip-install)")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sglang, flashinfer; "
                "print(f'  SGLang: {sglang.__version__}, FlashInfer: {flashinfer.__version__}')",
            ],
            capture_output=False,
        )

    trace_dir = Path(args.flashinfer_trace_dir).expanduser().resolve()

    if not trace_dir.exists():
        print(f"ERROR: flashinfer-trace directory not found: {trace_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve definition files
    def_names = getattr(args, "definitions", None)
    op_type = getattr(args, "op_type", None)
    def_files = _find_definitions(trace_dir, def_names, op_type)
    if not def_files:
        print("ERROR: no definition files found", file=sys.stderr)
        sys.exit(1)

    print(f"Definitions to collect: {[f.stem for f in def_files]}")

    # Determine dump directory
    if args.dump_dir:
        dump_dir = Path(args.dump_dir).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_dir = Path(f"workload_dumps_{ts}").resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FlashInfer Workload Collection")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Dump dir: {dump_dir}")

    if args.mode == "direct":
        run_direct_mode(def_files, trace_dir, dump_dir, args.replace)
    elif args.mode == "sglang":
        tp = getattr(args, "tp", None)
        ep = 1
        if tp is None:
            tp, ep = _tp_ep_from_definitions(def_files)
            print(f"Auto-detected from definition tags: tp={tp}, ep={ep}")
        # Page size: use explicit arg, or infer from definition constant axes
        page_size = getattr(args, "page_size", None) or 1
        if getattr(args, "page_size", None) is None:
            for f in def_files:
                defn = json.loads(f.read_text())
                ps = defn.get("axes", {}).get("page_size", {}).get("value")
                if ps is not None:
                    page_size = int(ps)
                    break
            print(f"Auto-detected page_size={page_size} from definition axes")
        run_sglang_mode(
            args.model_path,
            def_files,
            trace_dir,
            dump_dir,
            args.num_samples,
            getattr(args, "dataset", None),
            tp,
            args.replace,
            quantization=getattr(args, "quantization", None),
            cpu_offload_gb=getattr(args, "cpu_offload_gb", 0.0),
            ep=ep,
            page_size=page_size,
            skip_const_axis_check=getattr(args, "skip_const_axis_check", False),
        )

    print(f"\n{'='*60}")
    print(f"Collection complete! Dump dir: {dump_dir}")
    print(f"{'='*60}")

    if getattr(args, "eval_baseline", True):
        run_baseline_eval(def_files, trace_dir)


if __name__ == "__main__":
    main()
