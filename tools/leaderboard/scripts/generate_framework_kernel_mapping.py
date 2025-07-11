import os
import json
import argparse

def load_kernel_to_id_map(summary_path):
    """Build kernel_name -> leaderboard_id mapping from leaderboard_summary.json"""
    with open(summary_path, "r") as f:
        leaderboard_entries = json.load(f)
    return {entry["name"]: entry["id"] for entry in leaderboard_entries}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of Kernel Calling Record JSON files")
    parser.add_argument("--summary", required=True, help="Path to leaderboard_summary.json")
    parser.add_argument("--output", default="framework_kernels.jsonl", help="Path to output .jsonl file")
    args = parser.parse_args()

    # Load kernel_name -> leaderboard_id mapping
    kernel_to_id = load_kernel_to_id_map(args.summary)

    results = []

    # Iterate through all calling record JSON files
    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(args.input_dir, filename)
        with open(path, "r") as f:
            record = json.load(f)

        model = record["model"]
        for kernel in record.get("kernels", []):
            kernel_name = kernel["id"]
            leaderboard_id = kernel_to_id.get(kernel_name)
            if leaderboard_id is None:
                print(f"⚠️ Warning: Kernel '{kernel_name}' not found in leaderboard_summary.json")
                continue
            results.append({
                "model": model,
                "kernel_name": kernel_name,
                "leaderboard_id": leaderboard_id
            })

    # Write to output .jsonl file
    with open(args.output, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"✅ Wrote {len(results)} records to {args.output}")

if __name__ == "__main__":
    main()
