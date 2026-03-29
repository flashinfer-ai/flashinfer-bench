#!/usr/bin/env python3
"""FlashInfer-Bench Dashboard — monitor parallel definition onboarding."""

import json
import os
import re
import subprocess
import sys

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

# Add dashboard dir to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from data_sources import (
    get_definitions,
    get_gpu_status,
    get_model_coverage_summary,
    get_workload_coverage,
    get_worktree_tasks,
)
from log_parser import get_log_summary, parse_agent_log

app = Flask(__name__)

REPO_ROOT = os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ARCHITECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "architect")

_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _valid_name(name):
    return bool(name and _NAME_RE.match(name))


def _run_architect(*args):
    """Run tools/architect with the given args, return (stdout, stderr, returncode)."""
    cmd = [ARCHITECT] + list(args)
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


# ── HTML routes ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    tasks = get_worktree_tasks()
    defs = get_definitions()
    workload_covered = get_workload_coverage()
    gpu = get_gpu_status()
    model_coverage = get_model_coverage_summary()

    # Annotate definitions with workload and task status
    task_by_name = {t["name"]: t for t in tasks}
    for d in defs:
        d["has_workload"] = d["name"] in workload_covered
        d["task"] = task_by_name.get(d["name"])

    # Summary stats
    stats = {
        "total_defs": len(defs),
        "verified": sum(1 for d in defs if d["status_tag"] == "verified"),
        "with_workload": len(workload_covered),
        "active_tasks": len(tasks),
        "running_agents": sum(1 for t in tasks if t["agent_running"]),
        "prs_open": sum(1 for t in tasks if t["all_prs_open"]),
    }

    return render_template(
        "index.html", tasks=tasks, defs=defs, stats=stats, gpu=gpu, model_coverage=model_coverage
    )


@app.route("/task/<name>")
def task_detail(name):
    tasks = get_worktree_tasks()
    task = next((t for t in tasks if t["name"] == name), None)
    if not task:
        return f"Task '{name}' not found", 404

    log_path = os.path.join(task["bench_wt"], ".agent.log")
    messages = parse_agent_log(log_path) if os.path.exists(log_path) else []
    summary = get_log_summary(messages) if messages else {}

    return render_template("task_detail.html", task=task, messages=messages, summary=summary)


@app.route("/definitions")
def definitions_page():
    defs = get_definitions()
    workload_covered = get_workload_coverage()
    tasks = get_worktree_tasks()
    task_by_name = {t["name"]: t for t in tasks}
    for d in defs:
        d["has_workload"] = d["name"] in workload_covered
        d["task"] = task_by_name.get(d["name"])
    return render_template("definitions.html", defs=defs)


# ── JSON API ───────────────────────────────────────────────────────────────


@app.route("/api/tasks")
def api_tasks():
    return jsonify(get_worktree_tasks())


@app.route("/api/tasks/<name>")
def api_task(name):
    tasks = get_worktree_tasks()
    task = next((t for t in tasks if t["name"] == name), None)
    if not task:
        return jsonify({"error": "not found"}), 404
    return jsonify(task)


@app.route("/api/definitions")
def api_definitions():
    return jsonify(get_definitions())


@app.route("/api/gpu")
def api_gpu():
    return jsonify(get_gpu_status())


@app.route("/api/workloads")
def api_workloads():
    return jsonify(sorted(get_workload_coverage()))


@app.route("/api/coverage")
def api_coverage():
    return jsonify(get_model_coverage_summary())


@app.route("/api/agent-log/<name>")
def api_agent_log(name):
    tasks = get_worktree_tasks()
    task = next((t for t in tasks if t["name"] == name), None)
    if not task:
        return jsonify({"error": "not found"}), 404
    log_path = os.path.join(task["bench_wt"], ".agent.log")
    messages = parse_agent_log(log_path) if os.path.exists(log_path) else []
    return jsonify({"messages": messages, "summary": get_log_summary(messages)})


# ── Action API (POST) ──────────────────────────────────────────────────────


@app.route("/api/tasks", methods=["POST"])
def api_create_task():
    """Create bench+trace worktrees for a definition name."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    op_type = data.get("op_type", "").strip()
    if not _valid_name(name):
        return jsonify({"error": "invalid or missing name"}), 400
    args = ["create", name]
    if op_type:
        args += ["--op-type", op_type]
    stdout, stderr, code = _run_architect(*args)
    return jsonify({"ok": code == 0, "output": stdout or stderr}), (200 if code == 0 else 500)


@app.route("/api/tasks/<name>/spawn", methods=["POST"])
def api_spawn(name):
    """Spawn a Claude agent for a definition."""
    if not _valid_name(name):
        return jsonify({"error": "invalid name"}), 400
    data = request.get_json(silent=True) or {}
    args = ["spawn", name]
    if data.get("model"):
        args += ["--model", data["model"]]
    stdout, stderr, code = _run_architect(*args)
    return jsonify({"ok": code == 0, "output": stdout or stderr}), (200 if code == 0 else 500)


@app.route("/api/tasks/<name>/kill", methods=["POST"])
def api_kill(name):
    """Kill a running agent."""
    if not _valid_name(name):
        return jsonify({"error": "invalid name"}), 400
    stdout, stderr, code = _run_architect("kill", name)
    return jsonify({"ok": code == 0, "output": stdout or stderr}), (200 if code == 0 else 500)


@app.route("/api/tasks/<name>/rescue", methods=["POST"])
def api_rescue(name):
    """Commit dirty work in the bench worktree."""
    if not _valid_name(name):
        return jsonify({"error": "invalid name"}), 400
    data = request.get_json(silent=True) or {}
    msg = data.get("message", f"WIP: rescue via dashboard for {name}")
    stdout, stderr, code = _run_architect("rescue", name, "--action", "commit", "-m", msg)
    return jsonify({"ok": code == 0, "output": stdout or stderr}), (200 if code == 0 else 500)


@app.route("/api/tasks/<name>/remove", methods=["POST"])
def api_remove(name):
    """Remove worktrees for a definition (requires all PRs open or force)."""
    if not _valid_name(name):
        return jsonify({"error": "invalid name"}), 400
    data = request.get_json(silent=True) or {}
    args = ["remove", name, "-y"]
    if data.get("force"):
        args.append("--force")
    stdout, stderr, code = _run_architect(*args)
    return jsonify({"ok": code == 0, "output": stdout or stderr}), (200 if code == 0 else 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    print(f"FlashInfer-Bench Dashboard: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
