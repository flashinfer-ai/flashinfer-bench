import os
import json
import glob
import re
import argparse

def parse_tool_output(tool_output):
    """Extract runtime and hardware from tool_output string."""
    try:
        eval_match = re.search(
            r"evaluate_kernel: \[TextContent\(type='text', text='(.*?)'", tool_output, re.DOTALL
        )
        if eval_match:
            raw = eval_match.group(1).replace("\\\\", "\\")
            parsed = json.loads(raw)
            return parsed.get("runtime"), parsed.get("metadata", {}).get("hardware")
    except Exception as e:
        print(f"Failed to parse tool_output: {e}")
    return None, None

def load_submissions(task_path):
    """Process all .jsonl files under a task directory and return best results per file."""
    submissions = []
    for file in glob.glob(os.path.join(task_path, "*.jsonl")):
        with open(file) as f:
            logs = [json.loads(line) for line in f if line.strip()]

        print(f"Processing {file} with {len(logs)} log entries")
        model = None
        best_runtime = float("inf")
        best_hw = None

        for entry in logs:
            if entry["type"] == "model_used":
                model = entry["data"].get("codegen_model")
            elif entry["type"] == "iteration":
                rt, hw = parse_tool_output(entry["data"].get("tool_output", ""))
                if rt is not None and rt < best_runtime:
                    best_runtime = rt
                    best_hw = hw

        if model and best_runtime != float("inf"):
            submissions.append({
                "user_name": model,
                "score": best_runtime,
                "hardware": best_hw
            })

    return submissions

def generate_leaderboard(task_name, task_id, submissions):
    """Rank submissions by score and generate the leaderboard entry."""
    submissions.sort(key=lambda x: x["score"])
    top_users = []
    seen_models = set()

    for i, sub in enumerate(submissions):
        # if sub["user_name"] not in seen_models:
        if True:
            top_users.append({
                "rank": len(top_users) + 1,
                "score": round(sub["score"], 3),
                "user_name": sub["user_name"]
            })
            seen_models.add(sub["user_name"])
        if len(top_users) >= 3:
            break

    hw_list = list({s["hardware"] for s in submissions if s["hardware"]})

    return {
        "id": task_id,
        "name": task_name,
        "deadline": "2025-12-31T23:59:59Z",
        "gpu_types": hw_list,
        "priority_gpu_type": hw_list[0] if hw_list else None,
        "top_users": top_users
    }

def write_output(data, output_path, as_typescript=False):
    with open(output_path, "w") as f:
        if as_typescript:
            f.write("export const leaderboardData = ")
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to base directory with task folders")
    parser.add_argument("--output", type=str, required=True, help="Output file (JSON or TS)")
    parser.add_argument("--ts", action="store_true", help="Output as TypeScript module")

    args = parser.parse_args()

    task_dirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]

    leaderboard = []
    for i, task_name in enumerate(task_dirs):
        task_path = os.path.join(args.input, task_name)
        submissions = load_submissions(task_path)
        if submissions:
            entry = generate_leaderboard(task_name, i + 1, submissions)
            leaderboard.append(entry)

    write_output(leaderboard, args.output, as_typescript=args.ts)
    print(f"✅ Leaderboard written to {args.output}")
