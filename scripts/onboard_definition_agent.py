#!/usr/bin/env python3
"""
Headless definition onboarding agent using Anthropic tool-calling API.

Generates PR1 (definition JSON + reference test + coverage) and PR2
(baseline solution + workloads + blobs + eval trace) for a new kernel
definition, without any human interaction or Claude Code terminal session.

Three usage modes:

  Mode 1 — Explicit (fastest, recommended for CI):
    All parameters provided directly; no discovery phase.

    python scripts/onboard_definition_agent.py \
        --definition gqa_paged_prefill_causal_h40_kv10_d128_ps1 \
        --model-path /path/to/model \
        --model-name llama3-70b \
        --hf-repo-id meta-llama/Llama-3.1-70B-Instruct \
        --tp 4

  Mode 2 — Prompt-only (agent resolves all parameters):
    Agent reads existing definitions for schema patterns, searches HF cache
    for the model, and infers all parameters from the prompt.

    python scripts/onboard_definition_agent.py \
        --prompt "onboard gqa_paged_prefill_causal_h40_kv10_d128_ps1 for Llama 3.1 70B at TP=4"

  Mode 3 — Mixed (agent fills in only what's missing):
    Provide some args explicitly; agent resolves the rest from the prompt.

    python scripts/onboard_definition_agent.py \
        --prompt "onboard gqa_paged_prefill_causal_h40_kv10_d128_ps1 for Llama 3.1 70B" \
        --model-path /path/to/llama-70b \
        --tp 4

Environment:
    ANTHROPIC_API_KEY  — required (native Anthropic key starts with sk-ant-)
                         OR an NVIDIA Inference Hub key (starts with sk-) used
                         with --api-base-url https://inference-api.nvidia.com/v1
    CONDA_ENV          — conda env to use (default: flashinfer_bench)
    GITHUB_BRANCH      — branch to push PR1 to (default: auto-generated)

NVIDIA Inference Hub example:
    ANTHROPIC_API_KEY=sk-... python scripts/onboard_definition_agent.py \\
        --api-base-url https://inference-api.nvidia.com/v1 \\
        --claude-model aws/anthropic/bedrock-claude-sonnet-4-6 \\
        --definition gqa_paged_prefill_causal_h40_kv10_d128_ps1 ...
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONDA_ENV = os.environ.get("CONDA_ENV", "flashinfer_bench")
TRACE_DIR_DEFAULT = "tmp/flashinfer-trace"


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────


def _run(cmd: str, timeout: int = 60, cwd: str | None = None) -> dict:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or str(REPO_ROOT),
        )
        stdout = result.stdout
        stderr = result.stderr
        # Truncate very long outputs
        if len(stdout) > 6000:
            stdout = stdout[:3000] + "\n...[truncated]...\n" + stdout[-3000:]
        if len(stderr) > 3000:
            stderr = stderr[-3000:]
        return {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def _fmt(result: dict) -> str:
    parts = []
    if result["stdout"]:
        parts.append(f"STDOUT:\n{result['stdout']}")
    if result["stderr"]:
        parts.append(f"STDERR:\n{result['stderr']}")
    parts.append(f"Exit code: {result['returncode']}")
    return "\n".join(parts)


def tool_run_shell(command: str, timeout_seconds: int = 120) -> str:
    return _fmt(_run(command, timeout=timeout_seconds))


def tool_read_file(path: str) -> str:
    p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    if not p.exists():
        return f"File not found: {p}"
    try:
        content = p.read_text()
        if len(content) > 8000:
            return content[:4000] + "\n...[truncated — file too large]...\n" + content[-4000:]
        return content
    except Exception as e:
        return f"Error reading {p}: {e}"


def tool_write_file(path: str, content: str) -> str:
    p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error writing {p}: {e}"


def tool_list_files(directory: str, pattern: str = "*") -> str:
    d = Path(directory) if Path(directory).is_absolute() else REPO_ROOT / directory
    if not d.exists():
        return f"Directory not found: {d}"
    files = sorted(d.glob(pattern))
    return "\n".join(str(f.relative_to(REPO_ROOT)) for f in files[:100])


def tool_find_model_path(model_name: str) -> str:
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    results = []
    if hf_cache.exists():
        query = model_name.lower().replace("-", "").replace("_", "").replace("/", "")
        for model_dir in sorted(hf_cache.glob("models--*/")):
            dir_key = model_dir.name.lower().replace("-", "").replace("_", "")
            if query in dir_key:
                for snapshot in sorted((model_dir / "snapshots").glob("*/")):
                    results.append({"path": str(snapshot), "model_dir": model_dir.name})
    if not results:
        return f"No cached model found matching '{model_name}'. Provide --model-path explicitly."
    return json.dumps(results, indent=2)


def tool_git_op(operation: str, worktree: str | None = None) -> str:
    """Run a git operation in the repo or a specified worktree."""
    cwd = worktree if worktree else str(REPO_ROOT)
    return _fmt(_run(f"git {operation}", timeout=60, cwd=cwd))


def tool_run_tests(test_file: str) -> str:
    cmd = f"conda run -n {CONDA_ENV} python -m pytest {test_file} -v 2>&1"
    return _fmt(_run(cmd, timeout=180))


def tool_run_collection(
    model_path: str,
    definitions: list[str],
    flashinfer_trace_dir: str,
    tp: int = 1,
    quantization: str | None = None,
    ep: int = 1,
    extra_args: list[str] | None = None,
) -> str:
    defs_str = " ".join(definitions)
    cmd_parts = [
        f"conda run -n {CONDA_ENV} python scripts/collect_workloads.py sglang",
        f"--model-path {model_path}",
        f"--definitions {defs_str}",
        f"--flashinfer-trace-dir {flashinfer_trace_dir}",
        "--replace --skip-install",
    ]
    if tp and tp != 1:
        cmd_parts.append(f"--tp {tp}")
    if quantization:
        cmd_parts.append(f"--quantization {quantization}")
    if ep > 1:
        cmd_parts.append(f"--ep {ep}")
    if extra_args:
        cmd_parts.extend(extra_args)
    cmd = " ".join(cmd_parts)
    print(f"\n[agent] Running collection: {cmd}\n", flush=True)
    return _fmt(_run(cmd, timeout=3600))


def tool_run_baseline_eval(definition: str, flashinfer_trace_dir: str) -> str:
    cmd = (
        f"conda run -n {CONDA_ENV} python -m flashinfer_bench run "
        f"--local {flashinfer_trace_dir} "
        f"--definitions {definition} "
        f"--solutions baseline --save-results --warmup-runs 3 --iterations 20"
    )
    return _fmt(_run(cmd, timeout=300))


def tool_create_github_pr(title: str, body: str, branch: str, base: str = "main") -> str:
    # Ensure we're on the right branch
    body_escaped = body.replace("'", "'\\''")
    cmd = f"gh pr create --title '{title}' --body $'{body_escaped}' --base {base} --head {branch}"
    return _fmt(_run(cmd, timeout=60))


def tool_create_hf_pr(
    repo_id: str, title: str, description: str, branch: str, worktree: str
) -> str:
    # Step 1: create the PR (empty) to get the PR number
    script_create = textwrap.dedent(
        f"""\
        from huggingface_hub import HfApi
        api = HfApi()
        pr = api.create_pull_request(
            repo_id='{repo_id}',
            repo_type='dataset',
            title={repr(title)},
            description={repr(description)},
        )
        print('PR_URL:', pr.url)
        print('PR_NUM:', pr.num)
    """
    )
    cmd = f"conda run -n {CONDA_ENV} python -c {repr(script_create)}"
    result = _run(cmd, timeout=60)
    out = result.get("stdout", "")
    pr_num = None
    for line in out.splitlines():
        if line.startswith("PR_NUM:"):
            pr_num = line.split(":", 1)[1].strip()
    if not pr_num:
        return _fmt(result) + "\nERROR: could not parse PR number"

    # Step 2: push branch content directly to refs/pr/<num> on origin
    push_result = _run(f"git push origin HEAD:refs/pr/{pr_num} --force", timeout=300, cwd=worktree)
    return _fmt(result) + "\n" + _fmt(push_result)


def tool_check_workloads(flashinfer_trace_dir: str, definition: str) -> str:
    trace_dir = Path(flashinfer_trace_dir)
    if not (trace_dir / "definitions").exists():
        trace_dir = REPO_ROOT / flashinfer_trace_dir
    results = {}
    found = list(trace_dir.glob(f"definitions/**/{definition}.json"))
    op_type = found[0].parent.name if found else "unknown"
    wf = trace_dir / "workloads" / op_type / f"{definition}.jsonl"
    results["workload_exists"] = wf.exists()
    results["workload_count"] = len(wf.read_text().splitlines()) if wf.exists() else 0
    blobs = list((trace_dir / "blob" / "workloads" / op_type / definition).glob("*.safetensors"))
    results["blob_count"] = len(blobs)
    tf = trace_dir / "traces" / op_type / f"{definition}.jsonl"
    if tf.exists():
        records = [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
        statuses = [r.get("evaluation", {}).get("status") for r in records]
        results["trace_passed"] = statuses.count("PASSED")
        results["trace_failed"] = statuses.count("FAILED")
    return json.dumps(results, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Tool schema
# ──────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_shell",
        "description": "Run a shell command in the repo root. Use for diagnostics and ad-hoc operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read any file from the repo. Use to understand existing definition JSON, reference test, baseline solution, or coverage patterns.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative or absolute path"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates parent dirs). Use to create the definition JSON, reference test, baseline solution, or update coverage.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory, optionally filtered by glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "pattern": {"type": "string", "default": "*"},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "find_model_path",
        "description": "Search HuggingFace hub cache for a model by name. Returns local snapshot paths.",
        "input_schema": {
            "type": "object",
            "properties": {"model_name": {"type": "string"}},
            "required": ["model_name"],
        },
    },
    {
        "name": "git_op",
        "description": "Run a git command (e.g. 'add -A', 'commit -m \"msg\"', 'push', 'checkout -b branch'). Use worktree param for trace repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "description": "git sub-command and args"},
                "worktree": {
                    "type": "string",
                    "description": "Path to worktree (optional, defaults to repo root)",
                },
            },
            "required": ["operation"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest on a reference test file and return full output.",
        "input_schema": {
            "type": "object",
            "properties": {"test_file": {"type": "string"}},
            "required": ["test_file"],
        },
    },
    {
        "name": "run_collection",
        "description": "Run collect_workloads.py sglang to collect workloads via SGLang inference. Use after GPUs are verified free.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "definitions": {"type": "array", "items": {"type": "string"}},
                "flashinfer_trace_dir": {"type": "string"},
                "tp": {"type": "integer", "default": 1},
                "quantization": {"type": "string"},
                "ep": {"type": "integer", "default": 1},
                "extra_args": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["model_path", "definitions", "flashinfer_trace_dir"],
        },
    },
    {
        "name": "run_baseline_eval",
        "description": "Run flashinfer-bench eval for a definition against collected workloads. Produces traces JSONL with PASSED/FAILED status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "definition": {"type": "string"},
                "flashinfer_trace_dir": {"type": "string"},
            },
            "required": ["definition", "flashinfer_trace_dir"],
        },
    },
    {
        "name": "check_workloads",
        "description": "Verify that workload files, blobs, and eval traces exist and are valid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "flashinfer_trace_dir": {"type": "string"},
                "definition": {"type": "string"},
            },
            "required": ["flashinfer_trace_dir", "definition"],
        },
    },
    {
        "name": "create_github_pr",
        "description": "Open a GitHub pull request using 'gh pr create'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "branch": {"type": "string"},
                "base": {"type": "string", "default": "main"},
            },
            "required": ["title", "body", "branch"],
        },
    },
    {
        "name": "create_hf_pr",
        "description": "Open a HuggingFace dataset pull request via huggingface_hub API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "branch": {"type": "string"},
                "worktree": {"type": "string", "description": "Path to the trace git worktree"},
            },
            "required": ["repo_id", "title", "description", "branch", "worktree"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    dispatch = {
        "run_shell": tool_run_shell,
        "read_file": tool_read_file,
        "write_file": tool_write_file,
        "list_files": tool_list_files,
        "find_model_path": tool_find_model_path,
        "git_op": tool_git_op,
        "run_tests": tool_run_tests,
        "run_collection": tool_run_collection,
        "run_baseline_eval": tool_run_baseline_eval,
        "check_workloads": tool_check_workloads,
        "create_github_pr": tool_create_github_pr,
        "create_hf_pr": tool_create_hf_pr,
    }
    fn = dispatch.get(name)
    return fn(**inputs) if fn else f"Unknown tool: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a headless kernel definition onboarding agent for FlashInfer-Bench running in CI.
    Your job: submit PR1 (GitHub flashinfer-bench) and PR2 (HuggingFace flashinfer-trace)
    for a new kernel definition. No human is available — handle all steps autonomously.

    ── Phase 0: Discovery (skip for explicitly provided parameters) ──
    - Use read_file + list_files to understand existing definitions with the same op_type.
      Read 1–2 similar definitions (same op_type) to learn the schema, axes, constraints,
      reference implementation pattern, and baseline solution pattern.
    - Use find_model_path to locate the model snapshot in the HF cache.
    - Infer TP from the definition name suffix (e.g. h40_kv10 at TP=4 → 40 heads per device).
    - Use list_files on docs/model_coverage.mdx to understand coverage table format.

    ── Phase 1: Create definition JSON ──
    File: flashinfer_trace/definitions/{{op_type}}/{{definition_name}}.json
    - Copy the schema from a similar definition of the same op_type.
    - Update: name, description, const axis values (num_heads, head_dim, page_size, tp),
      tags (status:reference, fi_api:*, model:*, tp:*), and reference implementation.
    - The reference implementation is pure Python/PyTorch (no FlashInfer), used for correctness.
    - status:reference means the definition was manually authored (not captured from a real run).
      Use status:verified only if a real SGLang run confirmed the kernel fires.

    ── Phase 2: Create reference test ──
    File: flashinfer_trace/tests/references/test_{{definition_name}}.py
    - Read an existing test of the same op_type for the pattern.
    - The test imports the reference impl from the definition JSON and runs it against
      FlashInfer's API with random inputs, asserting allclose.
    - Run it with run_tests to verify all tests pass before committing.

    ── Phase 3: Update model coverage ──
    File: docs/model_coverage.mdx
    - Find the row for the model and update ❌ → 🟡 for this definition.
      Use 🟡 (not ✅) since workloads are not yet collected.

    ── Phase 4: Commit and open PR1 ──
    - IMPORTANT: Always branch from main, not from whatever is currently checked out:
        git checkout main && git checkout -b feat/def-{{definition_name}}
    - git add + commit all three files (definition JSON, reference test, coverage).
    - Push and open a GitHub PR with:
        Title: "feat: add {{definition_name}} definition"
        Body:  summary, kernel details table, reference test stdout, link to PR2 (add after PR2 opens)

    ── Phase 5: Collect workloads ──
    - Verify GPUs are free (run_shell nvidia-smi). Kill stale sglang if needed.
    - Run run_collection with the model path, definition name, TP, and trace dir.
    - Common failure recovery:
        OOM → run_shell pkill -f sglang.launch_server, wait 5s, retry
        DtypeDecl.h missing → create symlink (see below), clear fused_moe cache, retry
        Wrong env → always use conda run -n {conda_env}

      Symlink fix for DtypeDecl.h:
        CUBIN=$(conda run -n {conda_env} python -c "import flashinfer_cubin; print(flashinfer_cubin.__path__[0])")/cubins
        BMM=$(ls -d $CUBIN/*/batched_gemm-*/include 2>/dev/null | head -1)
        mkdir -p $CUBIN/flashinfer/trtllm/batched_gemm
        ln -sfn $BMM/trtllmGen_bmm_export $CUBIN/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export
        rm -rf ~/.cache/flashinfer/*/100a/cached_ops/fused_moe_trtllm_sm100

    ── Phase 6: Create baseline solution + run eval ──
    - Read an existing baseline solution of the same op_type for the pattern.
    - Write: {trace_dir}/solutions/baseline/{{op_type}}/{{definition_name}}/flashinfer_wrapper_<hash>.json
      The solution calls flashinfer.BatchPrefillWithPagedKVCacheWrapper (or equivalent).
      The hash is the first 6 chars of sha256 of the main.py content.
    - Run run_baseline_eval. All workloads must PASSED.

    ── Phase 7: Assemble and open PR2 ──
    - In the trace worktree, ALWAYS branch from origin/main:
        cd <trace_dir> && git checkout origin/main -b add-{{definition_name}}
      Never branch from an existing local branch (they may contain unrelated workloads).
    - Copy definition JSON and reference test into the trace worktree.
    - Commit all artifacts to the trace worktree branch.
    - Open HuggingFace PR on flashinfer-ai/flashinfer-trace with:
        Title: "Add {{definition_name}}: solution + workloads + blobs + eval trace"
        Body:  summary, workload diversity table, SGLang collection log
    - Update PR1 body to include the PR2 link.

    ── Completion ──
    When PR1 and PR2 are both open and all eval traces PASSED, output exactly:
      ONBOARD_COMPLETE: pr1=<url> pr2=<url>
    On unrecoverable failure after max_attempts:
      ONBOARD_FAILED: <reason>
"""
).format(conda_env=CONDA_ENV, trace_dir=TRACE_DIR_DEFAULT)


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────


def run_agent(
    flashinfer_trace_dir: str,
    max_attempts: int,
    claude_model: str,
    prompt: str | None = None,
    definition: str | None = None,
    model_path: str | None = None,
    model_name: str | None = None,
    hf_repo_id: str | None = None,
    tp: int | None = None,
    quantization: str | None = None,
    ep: int = 1,
    api_base_url: str | None = None,
) -> bool:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    use_openai_compat = bool(api_base_url)

    if use_openai_compat:
        from openai import OpenAI

        oa_client = OpenAI(base_url=api_base_url, api_key=api_key)
    else:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

    # Build task message
    known, unknown = [], []
    if definition:
        known.append(f"Definition name: {definition}")
    else:
        unknown.append("definition name (parse from prompt)")
    if model_path:
        known.append(f"Model path: {model_path}")
    else:
        unknown.append("model path (use find_model_path)")
    if model_name:
        known.append(f"Model name: {model_name}")
    if hf_repo_id:
        known.append(f"HuggingFace repo: {hf_repo_id}")
    if tp is not None:
        known.append(f"TP: {tp}")
    else:
        unknown.append("TP (infer from definition name or tags)")
    known += [
        f"EP: {ep}",
        f"Quantization: {quantization or 'auto'}",
        f"Flashinfer trace dir: {flashinfer_trace_dir}",
        f"Max collection attempts: {max_attempts}",
        f"Conda env: {CONDA_ENV}",
        f"Repo root: {REPO_ROOT}",
        f"Trace worktree: {REPO_ROOT / ('tmp/worktrees/trace-' + (definition or 'new'))} (create if missing)",
    ]

    parts = []
    if prompt:
        parts.append(f"User request: {prompt}")
    parts.append("Known parameters:\n" + "\n".join(f"  {p}" for p in known))
    if unknown:
        parts.append("Must resolve:\n" + "\n".join(f"  - {p}" for p in unknown))
    parts.append(
        "Start with Phase 0 discovery for any missing parameters, then proceed through all phases."
    )
    task = "\n\n".join(parts)

    messages = [{"role": "user", "content": task}]
    collection_attempts = 0

    print(f"[agent] Starting onboarding agent (model={claude_model})", flush=True)
    if use_openai_compat:
        print(f"[agent] API:        OpenAI-compat ({api_base_url})", flush=True)
    print(f"[agent] Definition: {definition or '(from prompt)'}", flush=True)
    print(f"[agent] Trace dir:  {flashinfer_trace_dir}\n", flush=True)

    # Convert TOOLS (Anthropic format) to OpenAI function format once
    if use_openai_compat:
        oa_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOLS
        ]

    while True:
        if use_openai_compat:
            # OpenAI-compatible path (NVIDIA Inference Hub, etc.)
            oa_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            oa_resp = oa_client.chat.completions.create(
                model=claude_model, max_tokens=8192, tools=oa_tools, messages=oa_messages
            )
            choice = oa_resp.choices[0]
            msg = choice.message

            # Extract text and tool calls from OpenAI response
            text_content = msg.content or ""
            tool_calls = msg.tool_calls or []

            if text_content:
                print(f"[agent] {text_content}", flush=True)
                if "ONBOARD_COMPLETE:" in text_content:
                    print("\n[agent] ✅ Onboarding complete.", flush=True)
                    return True
                if "ONBOARD_FAILED:" in text_content:
                    print("\n[agent] ❌ Onboarding failed.", flush=True)
                    return False

            if choice.finish_reason == "stop":
                print("[agent] Model stopped without completion signal.", flush=True)
                return False

            if choice.finish_reason != "tool_calls":
                print(f"[agent] Unexpected finish_reason: {choice.finish_reason}", flush=True)
                return False

            # Append assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if fn_name == "run_collection":
                    collection_attempts += 1
                    print(
                        f"\n[agent] Collection attempt {collection_attempts}/{max_attempts}",
                        flush=True,
                    )
                    if collection_attempts > max_attempts:
                        print("[agent] Max collection attempts reached.", flush=True)
                        return False
                print(f"[agent] → {fn_name}({json.dumps(fn_args)[:120]})", flush=True)
                result = dispatch_tool(fn_name, fn_args)
                print(f"[agent] ← {result[:400]}\n", flush=True)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # Native Anthropic SDK path
        response = client.messages.create(
            model=claude_model,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"[agent] {block.text}", flush=True)
                if "ONBOARD_COMPLETE:" in block.text:
                    print("\n[agent] ✅ Onboarding complete.", flush=True)
                    return True
                if "ONBOARD_FAILED:" in block.text:
                    print("\n[agent] ❌ Onboarding failed.", flush=True)
                    return False

        if response.stop_reason == "end_turn":
            print("[agent] Model stopped without completion signal.", flush=True)
            return False
        if response.stop_reason != "tool_use":
            print(f"[agent] Unexpected stop_reason: {response.stop_reason}", flush=True)
            return False

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "run_collection":
                collection_attempts += 1
                print(
                    f"\n[agent] Collection attempt {collection_attempts}/{max_attempts}", flush=True
                )
                if collection_attempts > max_attempts:
                    print(f"[agent] Max collection attempts reached.", flush=True)
                    return False

            print(f"[agent] → {block.name}({json.dumps(block.input)[:120]})", flush=True)
            result = dispatch_tool(block.name, block.input)
            print(f"[agent] ← {result[:400]}\n", flush=True)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Headless definition onboarding agent — generates PR1 and PR2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # Prompt-only:
              python scripts/onboard_definition_agent.py \\
                  --prompt "onboard gqa_paged_prefill_causal_h40_kv10_d128_ps1 for Llama 3.1 70B at TP=4"

              # Explicit:
              python scripts/onboard_definition_agent.py \\
                  --definition gqa_paged_prefill_causal_h40_kv10_d128_ps1 \\
                  --model-path /path/to/llama-70b \\
                  --model-name llama3-70b \\
                  --hf-repo-id meta-llama/Llama-3.1-70B-Instruct \\
                  --tp 4

              # Mixed:
              python scripts/onboard_definition_agent.py \\
                  --prompt "onboard gqa_paged_prefill_causal_h40_kv10_d128_ps1 for Llama 3.1 70B" \\
                  --model-path /path/to/llama-70b \\
                  --tp 4
        """
        ),
    )
    parser.add_argument("--prompt", default=None, help="Natural language onboarding request")
    parser.add_argument(
        "--definition",
        default=None,
        help="Definition name (e.g. gqa_paged_prefill_causal_h40_kv10_d128_ps1)",
    )
    parser.add_argument("--model-path", default=None, help="Local model snapshot path")
    parser.add_argument(
        "--model-name", default=None, help="Short model name for tags (e.g. llama3-70b)"
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="HuggingFace repo ID (e.g. meta-llama/Llama-3.1-70B-Instruct)",
    )
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (default: 1)")
    parser.add_argument("--quantization", default=None, help="Quantization (e.g. fp8)")
    parser.add_argument(
        "--flashinfer-trace-dir",
        default=TRACE_DIR_DEFAULT,
        help=f"Path to flashinfer-trace repo (default: {TRACE_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="Max collection attempts (default: 3)"
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-6",
        help=(
            "Model ID. For native Anthropic: claude-sonnet-4-6 (default). "
            "For NVIDIA Inference Hub: aws/anthropic/bedrock-claude-sonnet-4-6"
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL for non-Anthropic endpoints. "
            "Example: https://inference-api.nvidia.com/v1"
        ),
    )
    args = parser.parse_args()

    if not args.prompt and not args.definition:
        parser.error("At least one of --prompt or --definition is required.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # When using NVIDIA Inference Hub, default to the Hub model name
    claude_model = args.claude_model
    if args.api_base_url and claude_model == "claude-sonnet-4-6":
        claude_model = "aws/anthropic/bedrock-claude-sonnet-4-6"

    success = run_agent(
        prompt=args.prompt,
        definition=args.definition,
        model_path=args.model_path,
        model_name=args.model_name,
        hf_repo_id=args.hf_repo_id,
        flashinfer_trace_dir=args.flashinfer_trace_dir,
        tp=args.tp,
        quantization=args.quantization,
        ep=args.ep,
        max_attempts=args.max_attempts,
        claude_model=claude_model,
        api_base_url=args.api_base_url,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
