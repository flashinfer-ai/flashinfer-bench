"""Data collection for the FlashInfer-Bench dashboard."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
WORKTREES_DIR = os.path.join(REPO_ROOT, "tmp", "worktrees")
TRACE_REPO = os.path.join(REPO_ROOT, "tmp", "flashinfer-trace")
AGENTS_FILE = os.path.join(REPO_ROOT, ".claude", "agents.json")
DEFINITIONS_DIR = os.path.join(REPO_ROOT, "flashinfer_trace", "definitions")
MODEL_COVERAGE = os.path.join(REPO_ROOT, "docs", "model_coverage.mdx")
GPU_LOCKFILE = "/tmp/flashinfer-bench-gpu.lock"
GPU_INFOFILE = "/tmp/flashinfer-bench-gpu.info"


def _run_git(args, cwd=None):
    try:
        result = subprocess.run(
            ["git"] + args, cwd=cwd or REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _time_ago(timestamp):
    diff = int(time.time()) - int(timestamp)
    if diff < 60:
        return f"{diff}s ago"
    elif diff < 3600:
        return f"{diff // 60}m ago"
    elif diff < 86400:
        return f"{diff // 3600}h ago"
    else:
        return f"{diff // 86400}d ago"


def _load_agents():
    try:
        with open(AGENTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_progress(wt_path):
    p = os.path.join(wt_path, ".agent-progress.md")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return None


def _extract_status(progress_text):
    if not progress_text:
        return None
    for line in progress_text.splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return None


def _extract_pr_urls(progress_text):
    if not progress_text:
        return {}
    prs = {}
    for line in progress_text.splitlines():
        line = line.strip()
        for i in [1, 2, 3]:
            if line.startswith(f"- PR {i}"):
                parts = line.split(":", 1)
                url = parts[1].strip() if len(parts) > 1 else ""
                prs[f"pr{i}"] = url if url.startswith("http") else None
    return prs


def _is_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, TypeError, ValueError):
        return False


def get_definitions():
    """Scan flashinfer_trace/definitions/ and return all definition names + metadata."""
    defs = []
    if not os.path.isdir(DEFINITIONS_DIR):
        return defs
    for op_type_dir in sorted(os.listdir(DEFINITIONS_DIR)):
        op_path = os.path.join(DEFINITIONS_DIR, op_type_dir)
        if not os.path.isdir(op_path):
            continue
        for fname in sorted(os.listdir(op_path)):
            if not fname.endswith(".json"):
                continue
            def_name = fname[:-5]
            fpath = os.path.join(op_path, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
            defs.append(
                {
                    "name": def_name,
                    "op_type": op_type_dir,
                    "path": fpath,
                    "tags": data.get("tags", []),
                    "status_tag": next(
                        (t.split(":")[1] for t in data.get("tags", []) if t.startswith("status:")),
                        "unknown",
                    ),
                    "fi_api": next(
                        (
                            t.split(":", 1)[1]
                            for t in data.get("tags", [])
                            if t.startswith("fi_api:")
                        ),
                        None,
                    ),
                    "description": data.get("description", ""),
                }
            )
    return defs


def get_worktree_tasks():
    """Return all bench- worktrees with task status, agent info, and PR status."""
    if not os.path.isdir(WORKTREES_DIR):
        return []

    agents = _load_agents()
    tasks = []

    for entry in sorted(os.listdir(WORKTREES_DIR)):
        if not entry.startswith("bench-"):
            continue
        def_name = entry[len("bench-") :]
        wt_path = os.path.join(WORKTREES_DIR, entry)
        trace_wt = os.path.join(WORKTREES_DIR, f"trace-{def_name}")

        branch = _run_git(["branch", "--show-current"], cwd=wt_path)
        last_commit = _run_git(["log", "--oneline", "-1"], cwd=wt_path)
        dirty_out = _run_git(["status", "--short", "--ignore-submodules"], cwd=wt_path)
        dirty_count = len([l for l in dirty_out.splitlines() if l.strip()])
        commits_since_main_raw = _run_git(["log", "--oneline", "main..HEAD"], cwd=wt_path)
        commits_since_main = len([l for l in commits_since_main_raw.splitlines() if l.strip()])

        ts_raw = _run_git(["log", "-1", "--format=%ct"], cwd=wt_path)
        last_activity = int(ts_raw) if ts_raw.isdigit() else 0

        progress = _read_progress(wt_path)
        status = _extract_status(progress)
        prs = _extract_pr_urls(progress)

        # Agent status from agents.json
        agent_info = agents.get(def_name, {})
        pid = agent_info.get("pid")
        agent_running = bool(pid) and _is_pid_alive(pid)

        tasks.append(
            {
                "name": def_name,
                "bench_wt": wt_path,
                "trace_wt": trace_wt,
                "trace_wt_exists": os.path.isdir(trace_wt),
                "branch": branch,
                "last_commit": last_commit,
                "dirty_count": dirty_count,
                "commits_since_main": commits_since_main,
                "last_activity": last_activity,
                "last_activity_ago": _time_ago(last_activity) if last_activity else "unknown",
                "progress": progress,
                "progress_status": status or ("running" if agent_running else "idle"),
                "prs": prs,
                "all_prs_open": bool(prs.get("pr1") and prs.get("pr3")),
                "agent_running": agent_running,
                "agent_pid": pid,
                "agent_started": agent_info.get("started"),
                "agent_session_id": agent_info.get("session_id"),
                "has_task_spec": os.path.exists(os.path.join(wt_path, ".claude", "TASK.md")),
                "has_agent_log": os.path.exists(os.path.join(wt_path, ".agent.log")),
            }
        )

    return tasks


def get_gpu_status():
    """Check GPU lock status."""
    import subprocess

    try:
        result = subprocess.run(
            ["flock", "-n", GPU_LOCKFILE, "true"], capture_output=True, timeout=2
        )
        locked = result.returncode != 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        locked = False

    info = None
    if locked and os.path.exists(GPU_INFOFILE):
        with open(GPU_INFOFILE) as f:
            info = f.read().strip()

    return {"locked": locked, "info": info}


def get_workload_coverage():
    """Check which definitions have workloads collected in tmp/flashinfer-trace."""
    workloads_dir = os.path.join(TRACE_REPO, "workloads")
    covered = set()
    if os.path.isdir(workloads_dir):
        for op_dir in os.listdir(workloads_dir):
            op_path = os.path.join(workloads_dir, op_dir)
            if os.path.isdir(op_path):
                for fname in os.listdir(op_path):
                    if fname.endswith(".jsonl"):
                        covered.add(fname[:-6])
    return covered


def get_model_coverage_summary():
    """Parse docs/model_coverage.mdx for the summary table."""
    if not os.path.exists(MODEL_COVERAGE):
        return []
    with open(MODEL_COVERAGE) as f:
        content = f.read()
    rows = []
    in_table = False
    for line in content.splitlines():
        if line.startswith("| ") and "Architecture" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 3:
                rows.append({"model": cols[0], "architecture": cols[1], "coverage": cols[2]})
        elif in_table and not line.startswith("|"):
            in_table = False
    return rows
