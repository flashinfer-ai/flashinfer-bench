#!/usr/bin/env python3
"""
Headless workload collection agent using Anthropic tool-calling API.

Wraps collect_workloads.py with an autonomous retry/recovery loop that can
run in CI without any human interaction or Claude Code terminal session.

Usage:
    python scripts/collect_workloads_agent.py \
        --model-path /path/to/model \
        --definitions mla_ragged_prefill_causal_h16_qk192_vo128 \
        --flashinfer-trace-dir tmp/flashinfer-trace

    # With explicit TP and quantization:
    python scripts/collect_workloads_agent.py \
        --model-path /path/to/deepseek-v3 \
        --definitions mla_paged_decode_h16_ckv512_kpe64_ps1 \
        --tp 8 --quantization fp8 \
        --max-attempts 3

Environment:
    ANTHROPIC_API_KEY  — required
    CONDA_ENV          — conda env to use (default: flashinfer_bench)
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import anthropic

# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
CONDA_ENV = os.environ.get("CONDA_ENV", "flashinfer_bench")


def _run(cmd: str, timeout: int = 60) -> dict:
    """Run a shell command, return {stdout, stderr, returncode}."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        return {
            "stdout": result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def tool_run_shell(command: str, timeout_seconds: int = 120) -> str:
    result = _run(command, timeout=timeout_seconds)
    output = []
    if result["stdout"]:
        output.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        output.append(f"STDERR:\n{result['stderr']}")
    output.append(f"Exit code: {result['returncode']}")
    return "\n".join(output)


def tool_check_gpus() -> str:
    result = _run(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits",
        timeout=15,
    )
    if result["returncode"] != 0:
        return f"nvidia-smi failed: {result['stderr']}"
    lines = result["stdout"].strip().split("\n")
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            idx, used, total, util = parts
            rows.append(f"  GPU {idx}: {used}/{total} MiB used, {util}% util")
    return "GPU status:\n" + "\n".join(rows)


def tool_kill_sglang() -> str:
    result = _run("pkill -f 'sglang.launch_server' || true", timeout=10)
    time.sleep(2)
    return f"Sent SIGTERM to sglang processes. {result['stdout']}"


def tool_run_collection(
    model_path: str,
    definitions: list[str],
    flashinfer_trace_dir: str,
    tp: int = 8,
    quantization: str | None = None,
    ep: int = 1,
    extra_args: list[str] | None = None,
) -> str:
    """Launch collect_workloads.py under the correct conda env."""
    defs_str = " ".join(definitions)
    cmd_parts = [
        f"conda run -n {CONDA_ENV} python scripts/collect_workloads.py sglang",
        f"--model-path {model_path}",
        f"--definitions {defs_str}",
        f"--flashinfer-trace-dir {flashinfer_trace_dir}",
        "--replace",
        "--skip-install",
    ]
    if tp != 8:
        cmd_parts.append(f"--tp {tp}")
    if quantization:
        cmd_parts.append(f"--quantization {quantization}")
    if ep > 1:
        cmd_parts.append(f"--ep {ep}")
    if extra_args:
        cmd_parts.extend(extra_args)

    cmd = " ".join(cmd_parts)
    print(f"\n[agent] Running: {cmd}\n", flush=True)
    # Long timeout: model load + warmup + inference + sanitize can take ~30min
    return tool_run_shell(cmd, timeout_seconds=3600)


def tool_check_workloads(flashinfer_trace_dir: str, definition: str) -> str:
    trace_dir = Path(flashinfer_trace_dir)
    results = {}

    # Find op_type from definition file
    found = list(trace_dir.glob(f"definitions/**/{definition}.json"))
    op_type = found[0].parent.name if found else "unknown"

    workload_file = trace_dir / "workloads" / op_type / f"{definition}.jsonl"
    results["workload_file"] = str(workload_file)
    results["workload_exists"] = workload_file.exists()
    if workload_file.exists():
        lines = [l for l in workload_file.read_text().splitlines() if l.strip()]
        results["workload_count"] = len(lines)

    blob_dir = trace_dir / "blob" / "workloads" / op_type / definition
    blobs = list(blob_dir.glob("*.safetensors")) if blob_dir.exists() else []
    results["blob_count"] = len(blobs)

    trace_file = trace_dir / "traces" / op_type / f"{definition}.jsonl"
    results["trace_exists"] = trace_file.exists()
    if trace_file.exists():
        records = [json.loads(l) for l in trace_file.read_text().splitlines() if l.strip()]
        statuses = [r.get("evaluation", {}).get("status") for r in records]
        results["trace_statuses"] = dict(
            passed=statuses.count("PASSED"), failed=statuses.count("FAILED"), total=len(statuses)
        )

    return json.dumps(results, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Tool schema for Anthropic API
# ──────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_shell",
        "description": (
            "Run an arbitrary shell command in the repo root. Use for diagnostics, "
            "environment checks, git operations, log inspection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout_seconds": {
                    "type": "integer",
                    "default": 120,
                    "description": "Max seconds to wait",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "check_gpus",
        "description": "Check current GPU memory usage and utilization via nvidia-smi.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "kill_sglang",
        "description": "Kill any running sglang.launch_server processes to free GPU memory.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_collection",
        "description": (
            "Run collect_workloads.py sglang under the correct conda env. "
            "This is the primary collection command — use after verifying GPUs are free."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "definitions": {"type": "array", "items": {"type": "string"}},
                "flashinfer_trace_dir": {"type": "string"},
                "tp": {"type": "integer", "default": 8},
                "quantization": {"type": "string"},
                "ep": {"type": "integer", "default": 1},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra args passed verbatim to collect_workloads.py",
                },
            },
            "required": ["model_path", "definitions", "flashinfer_trace_dir"],
        },
    },
    {
        "name": "check_workloads",
        "description": (
            "Check whether workload files, blobs, and eval traces exist and are valid "
            "for a given definition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flashinfer_trace_dir": {"type": "string"},
                "definition": {"type": "string"},
            },
            "required": ["flashinfer_trace_dir", "definition"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "run_shell":
        return tool_run_shell(**inputs)
    elif name == "check_gpus":
        return tool_check_gpus()
    elif name == "kill_sglang":
        return tool_kill_sglang()
    elif name == "run_collection":
        return tool_run_collection(**inputs)
    elif name == "check_workloads":
        return tool_check_workloads(**inputs)
    else:
        return f"Unknown tool: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a headless workload collection agent for FlashInfer-Bench running in CI.
    Your job: collect workloads for the given kernel definitions using SGLang inference,
    then verify the results. No human is available — you must handle all failures autonomously.

    Workflow:
    1. check_gpus — verify all GPUs are free (< 1000 MiB used). If not, kill_sglang and recheck.
    2. run_collection — launch collect_workloads.py with the given parameters.
    3. On success: check_workloads to verify workload_count > 0, blobs exist, trace all PASSED.
    4. On failure: diagnose from stderr, fix the issue, retry (up to max_attempts).

    Common failure modes and fixes:
    - "CUDA error: out of memory" → kill_sglang, wait, retry
    - "No such file or directory: DtypeDecl.h" → run_shell to create symlink:
      CUBIN=$(conda run -n flashinfer_bench python -c "import flashinfer_cubin; print(flashinfer_cubin.__path__[0])")/cubins
      BMM=$(ls -d $CUBIN/*/batched_gemm-*/include 2>/dev/null | head -1)
      mkdir -p $CUBIN/flashinfer/trtllm/batched_gemm
      ln -sfn $BMM/trtllmGen_bmm_export $CUBIN/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export
      rm -rf ~/.cache/flashinfer/*/100a/cached_ops/fused_moe_trtllm_sm100
      Then retry.
    - "workload_count == 0 after success" → likely inference completed but no tensors dumped;
      check FLASHINFER_DUMP_INCLUDE filter matches definition tags. Try --skip-const-axis-check.
    - SGLang process exits immediately → check log for import errors; wrong conda env is common.
      Always use conda run -n {conda_env} python ..., never bare python.

    When collection succeeds and all checks pass, output exactly:
      COLLECTION_COMPLETE: <definition_name> <workload_count> workloads <blob_count> blobs
    If you exhaust all retries, output:
      COLLECTION_FAILED: <reason>
""").format(conda_env=CONDA_ENV)


def run_agent(
    model_path: str,
    definitions: list[str],
    flashinfer_trace_dir: str,
    tp: int,
    quantization: str | None,
    ep: int,
    max_attempts: int,
    claude_model: str,
    extra_args: list[str],
) -> bool:
    client = anthropic.Anthropic()

    task = textwrap.dedent(f"""\
        Collect workloads for the following kernel definitions:
          Definitions: {definitions}
          Model path: {model_path}
          Flashinfer trace dir: {flashinfer_trace_dir}
          TP: {tp}, EP: {ep}, Quantization: {quantization or 'auto'}
          Max attempts: {max_attempts}
          Extra collect_workloads args: {extra_args or []}
          Conda env: {CONDA_ENV}

        Start by checking GPU availability, then run collection.
    """)

    messages = [{"role": "user", "content": task}]
    attempt = 0

    print(f"[agent] Starting collection agent (model={claude_model}, max_attempts={max_attempts})")
    print(f"[agent] Definitions: {definitions}")
    print(f"[agent] Trace dir: {flashinfer_trace_dir}\n", flush=True)

    while True:
        response = client.messages.create(
            model=claude_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant response
        messages.append({"role": "assistant", "content": response.content})

        # Print any text blocks
        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"[agent] {block.text}", flush=True)

        # Check for terminal outputs
        for block in response.content:
            if hasattr(block, "text") and block.text:
                if "COLLECTION_COMPLETE:" in block.text:
                    print("\n[agent] ✅ Collection complete.", flush=True)
                    return True
                if "COLLECTION_FAILED:" in block.text:
                    print("\n[agent] ❌ Collection failed.", flush=True)
                    return False

        # Handle stop conditions
        if response.stop_reason == "end_turn":
            print("[agent] Model stopped without completion signal — treating as failure.")
            return False

        if response.stop_reason != "tool_use":
            print(f"[agent] Unexpected stop_reason: {response.stop_reason}")
            return False

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            attempt_inc = block.name == "run_collection"
            if attempt_inc:
                attempt += 1
                print(f"\n[agent] Collection attempt {attempt}/{max_attempts}", flush=True)
                if attempt > max_attempts:
                    print(f"[agent] Max attempts ({max_attempts}) reached.")
                    return False

            print(f"[agent] → {block.name}({json.dumps(block.input)[:120]})", flush=True)
            result = dispatch_tool(block.name, block.input)
            print(f"[agent] ← {result[:500]}\n", flush=True)

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Headless workload collection agent (Anthropic tool-calling)"
    )
    parser.add_argument("--model-path", required=True, help="Path to the HuggingFace model")
    parser.add_argument(
        "--definitions", nargs="+", required=True, help="Definition name(s) to collect"
    )
    parser.add_argument(
        "--flashinfer-trace-dir",
        default="tmp/flashinfer-trace",
        help="Path to flashinfer-trace repo (default: tmp/flashinfer-trace)",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size (default: 8)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (default: 1)")
    parser.add_argument("--quantization", default=None, help="Quantization (e.g. fp8)")
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="Max collection attempts (default: 3)"
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-6",
        help="Anthropic model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed verbatim to collect_workloads.py",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    success = run_agent(
        model_path=args.model_path,
        definitions=args.definitions,
        flashinfer_trace_dir=args.flashinfer_trace_dir,
        tp=args.tp,
        quantization=args.quantization,
        ep=args.ep,
        max_attempts=args.max_attempts,
        claude_model=args.claude_model,
        extra_args=args.extra_args,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
