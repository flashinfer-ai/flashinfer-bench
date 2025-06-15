import json
import re
from pathlib import Path

INPUT_ROOT = Path("/ssd1/yiyanz/llm-kernel-agent-results/jsonl_format")
OUTPUT_DIR = Path("/ssd1/yiyanz/CodeGen/flashinfer-bench/leaderboard/public/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_runtime_and_hw(tool_output: str) -> tuple[float | None, str | None]:
    try:
        match = re.search(r'evaluate_kernel:\s*\[TextContent\(type=\'text\', text=\'(.*?)\'', tool_output, re.DOTALL)
        if not match:
            return None, None

        raw_json_str = match.group(1)
        json_str = raw_json_str.encode('utf-8').decode('unicode_escape')
        parsed = json.loads(json_str)

        runtime = parsed.get("runtime")
        hardware = parsed.get("metadata", {}).get("hardware")
        return runtime, hardware
    except Exception as e:
        print("Failed to extract runtime/hardware:", e)
        return None, None

def extract_runtime_from_iteration(iteration: dict) -> float:
    tool_output = iteration.get("tool_output", "")
    try:
        match = re.search(r'evaluate_kernel:\s*\[TextContent\(type=\'text\', text=\'(.*?)\'', tool_output, re.DOTALL)
        if not match:
            return float("inf")

        raw_json_str = match.group(1)
        json_str = raw_json_str.encode('utf-8').decode('unicode_escape')
        parsed = json.loads(json_str)

        return parsed.get("runtime", float("inf"))
    except Exception:
        return float("inf")

import ast

def extract_latency_and_kernel_name(tool_output: str) -> tuple[float | None, str | None]:
    try:
        match = re.search(
            r'nsys_profiler:\s*\[TextContent\(type=\'text\', text=\'(.*?)(?<!\\)\'',
            tool_output,
            re.DOTALL
        )
        if not match:
            return None, None

        raw_json_str = match.group(1)
        fixed = raw_json_str.replace("\\\\", "\\")
        parsed = json.loads(fixed)

        kernel_stats = parsed.get("stdout", [])
        kernel_entries = [entry for entry in kernel_stats if isinstance(entry, dict)]

        # Prefer cudaLaunchKernel
        for k in kernel_entries:
            if "cudaLaunchKernel" in k.get("name", ""):
                return k.get("avg_ns"), k.get("name")

        # Fallback: kernel with max avg_ns
        if kernel_entries:
            best = max(kernel_entries, key=lambda x: x.get("avg_ns", 0))
            return best.get("avg_ns"), best.get("name")

        return None, None
    except Exception as e:
        print("Failed to extract latency + name:", e)
        return None, None


def extract_latency_ns(tool_output: str) -> float | None:
    try:
        match = re.search(r'nsys_profiler:\s*\[TextContent\(type=\'text\', text=\'(.*?)(?<!\\)\'', tool_output, re.DOTALL)
        if not match:
            return None

        raw_json_literal = match.group(1)
        # Safely interpret the escaped string literal
        clean_json = ast.literal_eval(f"'{raw_json_literal}'")
        parsed = json.loads(clean_json)

        kernel_stats = parsed.get("stdout", [])
        kernel_entries = [entry for entry in kernel_stats if isinstance(entry, dict)]

        # Prefer cudaLaunchKernel, fallback to largest avg_ns
        kernels = [k for k in kernel_entries if "cudaLaunchKernel" in k.get("name", "")]
        if kernels:
            return kernels[0].get("avg_ns")

        if kernel_entries:
            return max(kernel_entries, key=lambda x: x.get("avg_ns", 0)).get("avg_ns")

        return None
    except Exception as e:
        print("Failed to extract latency_ns:", e)
        return None

def parse_jsonl_file(filepath):
    task_id = filepath.parent.name
    submission_id = filepath.stem

    result = {
        "task_id": task_id,
        "submission_id": submission_id,
        "model": None,
        "code": None,
        "tool_name": None,
        "tool_output": None,
        "analysis": None,
        "latency_ns": None
    }

    iterations = []

    with open(filepath) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue

            t = entry.get("type")
            d = entry.get("data")

            if t == "model_used":
                result["model"] = d.get("codegen_model")

            elif t == "iteration":
                iterations.append(d)

    if iterations:
        best = min(iterations, key=extract_runtime_from_iteration)
        result["model_output_code"] = best.get("model_output")
        result["tool_name"] = best.get("tool_name")
        result["tool_output"] = best.get("tool_output")
        result["analysis"] = best.get("analysis")
        result["latency_ns"] = extract_latency_ns(best.get("tool_output", ""))
        runtime, hardware = extract_runtime_and_hw(best.get("tool_output", ""))
        result["runtime_ms"] = runtime
        result["hardware"] = hardware

    return result

def main():
    all_tasks = []

    for task_dir in INPUT_ROOT.iterdir():
        if not task_dir.is_dir():
            continue

        task_id = task_dir.name
        all_tasks.append(task_id)

        submissions = []
        for jsonl_file in task_dir.glob("*.jsonl"):
            submission = parse_jsonl_file(jsonl_file)
            submissions.append(submission)

        with open(OUTPUT_DIR / f"{task_id}.json", "w") as f:
            json.dump(submissions, f, indent=2)

    with open(OUTPUT_DIR / "generation_tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2)

if __name__ == "__main__":
    main()