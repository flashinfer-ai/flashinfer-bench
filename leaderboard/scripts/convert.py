import os
import json
import glob
import argparse

def parse_benchmark_result(tool_output):
    """Parse benchmark result JSON from tool_output string."""
    try:
        result = json.loads(tool_output)
        compiled = result.get("compiled", False)
        device_name = result.get("device", {}).get("name")
        if compiled:
            time = result.get("time")
            return time, device_name, True
        else:
            return None, device_name, False
    except Exception as e:
        print(f"Failed to parse tool_output: {e}")
        return None, None, False

def load_submissions(task_path):
    """Process all .jsonl files under a task directory and return best results per file."""
    submissions = []
    for file in glob.glob(os.path.join(task_path, "*.jsonl")):
        with open(file) as f:
            logs = [json.loads(line) for line in f if line.strip()]

        # Extract source name from file name, e.g., "gemm__gpt-4o.jsonl"
        base = os.path.basename(file)
        if "__" in base:
            _, source = base.replace(".jsonl", "").split("__", 1)
        else:
            source = "unknown"

        best_runtime = float("inf")
        best_hw = None
        found_valid = False

        for entry in logs:
            if entry.get("compiled") is True:
                runtime = entry.get("time")
                hardware = entry.get("device", {}).get("name")
                if runtime is not None and runtime < best_runtime:
                    best_runtime = runtime
                    best_hw = hardware
                    found_valid = True

        if found_valid:
            submissions.append({
                "user_name": source,
                "score": best_runtime,
                "hardware": best_hw
            })

    return submissions

def generate_leaderboard(task_name, task_id, submissions):
    """Rank submissions by best runtime and generate leaderboard entry."""
    submissions.sort(key=lambda x: x["score"])
    top_users = []
    seen_sources = set()

    for sub in submissions:
        if sub["user_name"] not in seen_sources:
            top_users.append({
                "rank": len(top_users) + 1,
                "score": round(sub["score"], 3),
                "user_name": sub["user_name"]
            })
            seen_sources.add(sub["user_name"])
        if len(top_users) >= 3:
            break

    hw_list = list({s["hardware"] for s in submissions if s["hardware"]})

    return {
        "id": task_id,
        "name": task_name,
        "gpu_types": hw_list,
        "priority_gpu_type": hw_list[0] if hw_list else None,
        "top_users": top_users
    }

def write_output(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to base directory with task folders")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")

    args = parser.parse_args()

    task_dirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
    print(f"Found {len(task_dirs)} task directories: {task_dirs}")
        

    leaderboard = []
    for i, task_name in enumerate(task_dirs):
        task_path = os.path.join(args.input, task_name)
        submissions = load_submissions(task_path)
        if submissions:
            entry = generate_leaderboard(task_name, i + 1, submissions)
            leaderboard.append(entry)

    write_output(leaderboard, args.output)
    print(f"✅ Leaderboard written to {args.output}")
