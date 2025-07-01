import subprocess
import os
import sys

def run_script(script, args):
    print(f"▶️ Running: {script} {' '.join(args)}")
    result = subprocess.run(["python", script] + args)
    if result.returncode != 0:
        print(f"❌ Script failed: {script}")
        sys.exit(1)
    print(f"✅ Done: {script}\n")

def run_module(module_name):
    print(f"▶️ Running module: python -m {module_name}")
    result = subprocess.run(["python", "-m", module_name])
    if result.returncode != 0:
        print(f"❌ Module {module_name} failed")
        sys.exit(1)
    print(f"✅ Module {module_name} completed\n")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    input_dir = os.path.join(root_dir, "data", "benchmark_logs")
    kernel_record_dir = os.path.join(root_dir, "data", "kernel_record")
    output_dir = os.path.join(root_dir, "leaderboard", "static")
    summary_path = os.path.join(output_dir, "leaderboard.json")
    framework_output = os.path.join(output_dir, "framework_kernels.jsonl")

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: convert.py
    run_script("scripts/convert.py", [
        "--input", input_dir,
        "--output", summary_path
    ])

    # Step 2: generate_detailed_leaderboard.py
    run_script("scripts/generate_detailed_leaderboard.py", [
        "--input", summary_path,
        "--output_dir", output_dir
    ])

    # Step 3: generate_framework_kernel_mapping.py
    run_script("scripts/generate_framework_kernel_mapping.py", [
        "--input_dir", kernel_record_dir,
        "--summary", summary_path,
        "--output", framework_output
    ])

    # Step 4: Run leaderboard
    run_module("leaderboard.run")

if __name__ == "__main__":
    main()
