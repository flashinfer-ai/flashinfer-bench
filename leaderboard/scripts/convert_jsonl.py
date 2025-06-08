import json
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path("/YOUR PATH TO/llm-kernel-agent-results/jsonl_format")
OUT = Path("/YOUR PATH TO/CodeGen/flashinfer-bench/kernelboard/public/data")
OUT.mkdir(parents=True, exist_ok=True)

all_tasks = []

for task_dir in ROOT.iterdir():
    if not task_dir.is_dir():
        continue
    task_id = task_dir.name
    all_tasks.append(task_id)
    task_out = []

    for run_file in task_dir.glob("*.jsonl"):
        with open(run_file) as f:
            submission = {
                "task_id": task_id,
                "submission_id": run_file.stem,
                "model": None,
                "code": None,
                "analysis": None,
                "nsys": None
            }
            for line in f:
                obj = json.loads(line)
                t = obj.get("type")
                d = obj.get("data")
                if t == "models":
                    submission["model"] = d.get("codegen_model")
                elif t == "round":
                    submission["code"] = d.get("model_output")
                    submission["analysis"] = d.get("analysis")
                    # crude nsys parser: get average latency if tool_output is structured
                    if isinstance(d.get("tool_output"), str) and "avg_ns" in d["tool_output"]:
                        try:
                            submission["nsys"] = json.loads(d["tool_output"])  # optional enhancement
                        except Exception:
                            pass
            task_out.append(submission)

    with open(OUT / f"{task_id}.json", "w") as f:
        json.dump(task_out, f, indent=2)

# Write tasks.json
with open(OUT / "tasks.json", "w") as f:
    json.dump(all_tasks, f, indent=2)
