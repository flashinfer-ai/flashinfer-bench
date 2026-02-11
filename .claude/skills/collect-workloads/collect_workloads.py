#!/usr/bin/env python3
"""
Auto-collect workloads from SGLang inference runs using FlashInfer logging API.

This script implements the complete workflow:
1. Update SGLang from main branch
2. Create workload collection branch
3. Locate FlashInfer API calls in SGLang
4. Set FlashInfer logging environment
5. Launch SGLang server with recommended config
6. Run ShareGPT inference benchmark
7. Process and sanitize dumped tensors (TODO)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class WorkloadCollector:
    """Orchestrates the complete workload collection pipeline."""

    def __init__(
        self,
        definition_name: str,
        sglang_path: Path,
        conda_env: str = "flashinfer",
        model_name: Optional[str] = None,
        launch_command: Optional[str] = None,
        num_prompts: int = 1000,
        tp: int = 1,
    ):
        self.definition_name = definition_name
        self.sglang_path = Path(sglang_path)
        self.conda_env = conda_env
        self.model_name = model_name
        self.launch_command = launch_command
        self.num_prompts = num_prompts
        self.tp = tp

        # Setup paths
        self.repo_root = Path(__file__).parent.parent.parent.parent
        self.definition_path = self._find_definition()
        self.dump_dir = self.sglang_path / "flashinfer_dumps"

        # Load definition
        with open(self.definition_path) as f:
            self.definition = json.load(f)

        self.op_type = self.definition["op_type"]

    def _find_definition(self) -> Path:
        """Find the definition JSON file for the given definition name."""
        definitions_dir = self.repo_root / "flashinfer_trace" / "definitions"

        # Search all op_type directories
        for op_type_dir in definitions_dir.iterdir():
            if not op_type_dir.is_dir():
                continue

            def_path = op_type_dir / f"{self.definition_name}.json"
            if def_path.exists():
                return def_path

        raise FileNotFoundError(
            f"Definition '{self.definition_name}' not found in {definitions_dir}"
        )

    def run_command(self, cmd: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with conda environment activated."""
        # Wrap command in conda activation
        full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env} && {cmd}"

        print(f"$ {cmd}")
        result = subprocess.run(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
        )

        if check and result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        return result

    def step1_update_sglang(self):
        """Step 1: Get updated main branch from SGLang and install."""
        print("\n" + "="*60)
        print("Step 1: Update SGLang from main branch")
        print("="*60)

        # Check if sglang path exists
        if not self.sglang_path.exists():
            print(f"❌ SGLang path not found: {self.sglang_path}")
            print("Please clone SGLang first using /clone-repos")
            sys.exit(1)

        # Fetch and pull latest main
        print("Fetching latest changes from main...")
        self.run_command("git fetch origin", cwd=self.sglang_path)
        self.run_command("git checkout main", cwd=self.sglang_path)
        self.run_command("git pull origin main", cwd=self.sglang_path)

        # Install SGLang
        print("Installing SGLang...")
        self.run_command("pip install -e '.[all]'", cwd=self.sglang_path)

        # Verify installation
        result = self.run_command("python -c 'import sglang; print(sglang.__version__)'")
        sglang_version = result.stdout.strip()
        print(f"✓ SGLang installed: version {sglang_version}")

    def step2_create_branch(self):
        """Step 2: Checkout new branch for workload collection."""
        print("\n" + "="*60)
        print(f"Step 2: Create workload collection branch")
        print("="*60)

        branch_name = f"collect_workload_for_{self.definition_name}"

        # Check if branch already exists
        result = self.run_command(
            f"git rev-parse --verify {branch_name}",
            cwd=self.sglang_path,
            check=False
        )

        if result.returncode == 0:
            print(f"Branch '{branch_name}' already exists, checking out...")
            self.run_command(f"git checkout {branch_name}", cwd=self.sglang_path)
        else:
            print(f"Creating new branch '{branch_name}'...")
            self.run_command(f"git checkout -b {branch_name}", cwd=self.sglang_path)

        print(f"✓ On branch: {branch_name}")

    def _generate_dump_filter(self) -> str:
        """Generate FLASHINFER_DUMP_INCLUDE filter pattern from definition tags.

        Reads FlashInfer API names from tags field (e.g., "api:batch_decode_with_paged_kv_cache").
        Falls back to inferring from definition name if no api tags present.

        Examples with api tags:
        - tags: ["api:batch_decode_with_paged_kv_cache"] → "*batch_decode_with_paged_kv_cache*"
        - tags: ["api:mla_decode", "api:mla_prefill"] → "*mla_decode*,*mla_prefill*"

        Fallback examples (without api tags):
        - gqa_paged_decode_h32_kv8_d128_ps1 → "*gqa_paged*,*decode*"
        - rmsnorm_h7168 → "*rmsnorm*"
        """
        # First, check for api:* tags in the definition
        api_tags = [tag.split(":", 1)[1] for tag in self.definition.get("tags", [])
                    if tag.startswith("api:")]

        if api_tags:
            # Use API names from tags
            filters = [f"*{api_name}*" for api_name in api_tags]
            return ",".join(filters)

        # Fallback: infer from definition name and op_type
        name_lower = self.definition_name.lower()
        filters = []

        # Check for op_type
        if self.op_type in name_lower:
            filters.append(f"*{self.op_type}*")

        # Check for decode/prefill/etc
        if "decode" in name_lower:
            filters.append("*decode*")
        elif "prefill" in name_lower:
            filters.append("*prefill*")

        # If we have filters, join them; otherwise use a generic pattern
        if filters:
            return ",".join(filters)
        else:
            # Last resort: use op_type
            return f"*{self.op_type}*"

    def step3_setup_flashinfer_logging(self):
        """Step 3: Set FlashInfer logging environment variables with API filtering."""
        print("\n" + "="*60)
        print("Step 3: Configure FlashInfer logging")
        print("="*60)

        # Create dump directory
        self.dump_dir.mkdir(parents=True, exist_ok=True)

        # Generate dump filter based on definition name
        dump_filter = self._generate_dump_filter()

        # Set environment variables
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_DUMP_DIR"] = str(self.dump_dir)
        os.environ["FLASHINFER_DUMP_SAFETENSORS"] = "1"
        os.environ["FLASHINFER_DUMP_MAX_COUNT"] = "10000"
        os.environ["FLASHINFER_DUMP_INCLUDE"] = dump_filter

        print(f"✓ FLASHINFER_LOGLEVEL=10")
        print(f"✓ FLASHINFER_DUMP_DIR={self.dump_dir}")
        print(f"✓ FLASHINFER_DUMP_SAFETENSORS=1")
        print(f"✓ FLASHINFER_DUMP_MAX_COUNT=10000")
        print(f"✓ FLASHINFER_DUMP_INCLUDE={dump_filter}")
        print(f"  (Only dumps APIs matching: {dump_filter})")

    def step4_launch_sglang_server(self) -> subprocess.Popen:
        """Step 4: Launch SGLang server with recommended config."""
        print("\n" + "="*60)
        print("Step 4: Launch SGLang server")
        print("="*60)

        if self.launch_command:
            # Use custom launch command
            cmd = self.launch_command
            print(f"Using custom launch command: {cmd}")
        else:
            # Use recommended config or default
            if not self.model_name:
                # Infer model from definition name or use default
                self.model_name = self._infer_model_from_definition()

            cmd = (
                f"python3 -m sglang.launch_server "
                f"--model {self.model_name} "
                f"--tp {self.tp} "
                f"--host 0.0.0.0 "
                f"--port 30000 "
                f"--attention-backend flashinfer "
                f"--disable-cuda-graph "
                f"--log-level info"
            )
            print(f"Using default launch command:")
            print(f"  {cmd}")

        # Prepare environment with FlashInfer logging (inherit from step3)
        env = os.environ.copy()

        # Launch server in background
        full_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self.conda_env} && {cmd}"

        print("Starting SGLang server (this may take a few minutes)...")
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready
        print("Waiting for server to be ready...")
        time.sleep(60)  # Initial wait

        # Check if server is responding
        max_retries = 10
        for i in range(max_retries):
            try:
                import requests
                response = requests.get("http://localhost:30000/health", timeout=5)
                if response.status_code == 200:
                    print("✓ SGLang server is ready")
                    return process
            except Exception as e:
                if i < max_retries - 1:
                    print(f"  Waiting for server... ({i+1}/{max_retries})")
                    time.sleep(10)
                else:
                    print(f"❌ Server failed to start after {max_retries} retries")
                    process.kill()
                    raise

        return process

    def step5_run_inference_benchmark(self):
        """Step 5: Run SGLang bench_serving with ShareGPT dataset."""
        print("\n" + "="*60)
        print("Step 5: Run inference benchmark")
        print("="*60)

        cmd = (
            f"python3 -m sglang.bench_serving "
            f"--backend sglang "
            f"--dataset-name sharegpt "
            f"--num-prompts {self.num_prompts}"
        )

        print(f"Running benchmark with {self.num_prompts} prompts...")
        print(f"  {cmd}")

        # Run benchmark
        result = self.run_command(cmd, cwd=self.sglang_path)

        print("✓ Benchmark complete")
        print(f"\nBenchmark output:\n{result.stdout}")

    def _infer_model_from_definition(self) -> str:
        """Infer appropriate model from definition name."""
        name_lower = self.definition_name.lower()

        # Map common patterns to models
        if "mla" in name_lower or "dsa" in name_lower:
            return "deepseek-ai/DeepSeek-V3"
        elif "gdn" in name_lower:
            return "Qwen/Qwen3-Next-80B-A3B-Instruct"
        elif "gqa" in name_lower:
            return "meta-llama/Llama-3.1-8B-Instruct"
        else:
            # Default to a common model
            return "meta-llama/Llama-3.1-8B-Instruct"

    def cleanup_server(self, process: subprocess.Popen):
        """Shutdown SGLang server gracefully."""
        print("\n" + "="*60)
        print("Cleanup: Stopping SGLang server")
        print("="*60)

        process.terminate()
        try:
            process.wait(timeout=10)
            print("✓ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            process.kill()
            process.wait()
            print("✓ Server killed")

    def run(self):
        """Execute the complete workflow."""
        print("\n" + "="*70)
        print(f"FlashInfer Workload Collection: {self.definition_name}")
        print("="*70)
        print(f"Definition: {self.definition_name}")
        print(f"Op Type: {self.op_type}")
        print(f"SGLang Path: {self.sglang_path}")
        print(f"Conda Env: {self.conda_env}")
        print(f"Dump Dir: {self.dump_dir}")
        print("="*70)

        server_process = None

        try:
            # Steps 1-3: Setup
            self.step1_update_sglang()
            self.step2_create_branch()
            self.step3_setup_flashinfer_logging()

            # Steps 4-5: Run inference
            server_process = self.step4_launch_sglang_server()
            self.step5_run_inference_benchmark()

            # TODO: Step 6+: Sanitize and process dumps (to be implemented)
            print("\n" + "="*60)
            print("TODO: Steps 6+ (sanitization, workload generation, PR submission)")
            print("="*60)
            print(f"Raw dumps available at: {self.dump_dir}")

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if server_process:
                self.cleanup_server(server_process)

        print("\n" + "="*70)
        print("Workload Collection Complete (Steps 1-5)")
        print("="*70)
        print(f"Next steps:")
        print(f"  1. Review dumps in: {self.dump_dir}")
        print(f"  2. Implement sanitization (Step 6)")
        print(f"  3. Generate workload JSONL")
        print(f"  4. Submit PR to flashinfer-trace")


def main():
    parser = argparse.ArgumentParser(
        description="Collect workloads from SGLang inference using FlashInfer logging"
    )
    parser.add_argument(
        "--definition-name",
        required=True,
        help="Definition name to collect workloads for (e.g., gqa_paged_decode_h32_kv8_d128_ps1)"
    )
    parser.add_argument(
        "--sglang-path",
        default="./tmp/sglang",
        help="Path to SGLang repository (default: ./tmp/sglang)"
    )
    parser.add_argument(
        "--conda-env",
        default="flashinfer",
        help="Conda environment name (default: flashinfer)"
    )
    parser.add_argument(
        "--model-name",
        help="Model to use for inference (auto-inferred if not provided)"
    )
    parser.add_argument(
        "--launch-command",
        help="Custom SGLang launch command (overrides default config)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process (default: 1000)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree (default: 1)"
    )

    args = parser.parse_args()

    collector = WorkloadCollector(
        definition_name=args.definition_name,
        sglang_path=Path(args.sglang_path),
        conda_env=args.conda_env,
        model_name=args.model_name,
        launch_command=args.launch_command,
        num_prompts=args.num_prompts,
        tp=args.tp,
    )

    collector.run()


if __name__ == "__main__":
    main()
