"""
Benchmark SGLang serving throughput using the ShareGPT dataset.

Launches an SGLang server, sends batched requests from ShareGPT, and reports
throughput and latency statistics across a sweep of batch sizes. Model server
flags are loaded from model_configs.json, keyed by GPU type and model name.

Usage:
    # Auto-detect GPU type from nvidia-smi
    python3 bench_sharegpt.py --model llama-3.1-8b  --model-path /path/to/Llama-3.1-8B-Instruct
    python3 bench_sharegpt.py --model deepseek-v3   --model-path /path/to/DeepSeek-V3
    python3 bench_sharegpt.py --model qwen3-235b-a22b --model-path /path/to/Qwen3-235B-A22B

    # Explicitly specify GPU type (b200, h200, h100, mi300x)
    python3 bench_sharegpt.py --gpu h100 --model llama-3.1-70b --model-path /path/to/Llama-3.1-70B-Instruct

    # TP size is auto-detected from CUDA_VISIBLE_DEVICES (set by gpu-lock):
    #   tools/gpu-lock --gpus 4 -- python3 bench_sharegpt.py --model qwen3-235b-a22b ...
    #   → CUDA_VISIBLE_DEVICES=0,1,2,3 → TP=4 used automatically

    # Tracing flags
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model --disable-radix-cache
    python3 bench_sharegpt.py --model qwen3-235b-a22b --model-path /path/to/model --enable-deterministic-inference

    # Custom batch sizes and number of batches
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model --batch-sizes 32 128 --num-batches 8

Environment variables (optional, for flashinfer-bench tracing):
    FIB_ENABLE_APPLY=1        Enable the flashinfer-bench apply hook
    FIB_DATASET_PATH=<dir>    Directory to write flashinfer trace data

    Example:
        FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=/path/to/traces/ python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/model
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_CONFIG_FILE = Path(__file__).parent / "model_configs.json"

# Map nvidia-smi GPU name substrings → sgl-cookbook hardware IDs
_GPU_NAME_MAP = [
    ("B200", "b200"),
    ("H200", "h200"),
    ("H100", "h100"),
    ("A100", "a100"),
    ("MI355", "mi355x"),
    ("MI325", "mi325x"),
    ("MI300", "mi300x"),
]


def detect_gpu_type() -> str:
    """Auto-detect GPU type from nvidia-smi or rocm-smi, returning a sgl-cookbook hardware ID."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0].upper()
            for substr, gpu_id in _GPU_NAME_MAP:
                if substr in name:
                    return gpu_id
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            name = result.stdout.upper()
            for substr, gpu_id in _GPU_NAME_MAP:
                if substr in name:
                    return gpu_id
    except Exception:
        pass
    return "b200"  # default fallback


def detect_tp_size() -> Optional[int]:
    """Detect available GPU count from CUDA_VISIBLE_DEVICES (set by gpu-lock) to use as TP size."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "-1":
        return len(cvd.split(","))
    # Fall back to total GPU count from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            count = len(result.stdout.strip().splitlines())
            return count if count > 0 else None
    except Exception:
        pass
    return None


from datasets import load_dataset
from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


@dataclass
class TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    extra_request_body: Dict[str, Any] = field(default_factory=dict)
    routing_key: Optional[str] = None


def load_model_configs() -> dict:
    """Load model configurations from model_configs.json."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Model config file not found: {_CONFIG_FILE}")
    with _CONFIG_FILE.open() as f:
        data = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_available_models(gpu_type: str) -> List[str]:
    """Return model keys available for a given GPU type."""
    configs = load_model_configs()
    gpu_key = gpu_type.lower()
    if gpu_key not in configs:
        # Fall back to first GPU type defined
        gpu_key = next(iter(configs))
    return list(configs[gpu_key].keys())


def get_model_config(model_type: str, gpu_type: str) -> dict:
    """Return server flags for a given model + GPU type from model_configs.json."""
    configs = load_model_configs()
    gpu_key = gpu_type.lower()
    if gpu_key not in configs:
        available_gpus = list(configs.keys())
        raise ValueError(f"Unknown GPU type {gpu_type!r}. Available: {available_gpus}")
    model_key = model_type.lower()
    gpu_configs = configs[gpu_key]
    if model_key not in gpu_configs:
        available_models = list(gpu_configs.keys())
        raise ValueError(
            f"Model {model_type!r} not configured for GPU {gpu_type!r}. "
            f"Available models for {gpu_type}: {available_models}"
        )
    return gpu_configs[model_key]


def log(msg: str) -> None:
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def load_prompts_from_sharegpt(n: int) -> List[str]:
    """Load n prompts from the ShareGPT dataset."""
    import json as _json
    import os as _os

    _local = "/tmp/sharegpt_synthetic.jsonl"
    if _os.path.exists(_local):
        prompts = []
        with open(_local) as _f:
            for _line in _f:
                _d = _json.loads(_line)
                prompts.append(_d.get("prompt", _d.get("conversations", [{}])[0].get("value", "")))
                if len(prompts) >= n:
                    break
    else:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train",
            streaming=True,
        )
        prompts = []
        for example in ds:
            conv = example.get("conversations", [])
            if conv and conv[0]["from"].lower() == "human":
                prompts.append(conv[0]["value"])
            if len(prompts) >= n:
                break

    log(f"Loaded {len(prompts)} prompts from ShareGPT dataset")
    prompt_lengths = [len(p) for p in prompts]
    if prompt_lengths:
        log(
            f"Prompt length statistics — "
            f"Min: {min(prompt_lengths)}, "
            f"Max: {max(prompt_lengths)}, "
            f"Avg: {sum(prompt_lengths) / len(prompt_lengths):.1f}"
        )
    return prompts


class DummyTokenizer:
    """Minimal tokenizer shim required by the benchmark function."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def build_bench_args() -> SimpleNamespace:
    """Build the SimpleNamespace expected by bench_serving.benchmark."""
    return SimpleNamespace(
        backend="sglang",
        dataset_name="custom",
        disable_ignore_eos=False,
        disable_stream=True,
        return_logprob=False,
        return_routed_experts=False,
        output_file=None,
        output_details=False,
        warmup_requests=0,
        plot_throughput=False,
        header=None,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        profile_activities=["CPU", "GPU"],
        profile_num_steps=None,
        profile_by_stage=False,
        profile_stages=None,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
    )


def run_benchmark(base_url: str, prompts: List[str], batch_size: int) -> list:
    """Run the benchmark over prompts in batches and return all results."""
    tokenizer = DummyTokenizer()
    set_global_args(build_bench_args())

    num_batches = math.ceil(len(prompts) / batch_size)
    all_results = []

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        current_time = time.time()
        input_requests = [
            TestRequest(
                prompt=p,
                prompt_len=0,
                output_len=0,
                timestamp=current_time,
                text_prompt_len=0,
                vision_prompt_len=0,
            )
            for p in batch_prompts
        ]

        log(f"Running batch {i + 1}/{num_batches} with {len(input_requests)} prompts")
        results = asyncio.run(
            benchmark(
                backend="sglang",
                api_url=f"{base_url}/generate",
                base_url=base_url,
                model_id="default",
                tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=float("inf"),
                max_concurrency=batch_size,
                disable_tqdm=False,
                lora_names=None,
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={"sampling_params": {"temperature": 0}},
                profile=False,
            )
        )
        all_results.append(results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ShareGPT dataset with SGLang using different models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type (b200, h200, h100, mi300x). Auto-detected from nvidia-smi if not specified.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to use from model_configs.json (e.g. llama-3.1-8b, qwen3-235b-a22b).",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model weights directory"
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Add --disable-radix-cache to server args (for ragged prefill collection)",
    )
    parser.add_argument(
        "--enable-deterministic-inference",
        action="store_true",
        help="Add --enable-deterministic-inference to server args (for paged prefill collection)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:20000",
        help="Base URL of the SGLang server (default: http://127.0.0.1:20000)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 64],
        help="List of batch sizes to benchmark (default: 1 16 64)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of batches per batch-size sweep (default: 4)",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Pass --disable-cuda-graph to the SGLang server",
    )
    args = parser.parse_args()

    # Resolve GPU type
    gpu_type = args.gpu if args.gpu else detect_gpu_type()
    log(f"GPU type:       {gpu_type}" + (" (auto-detected)" if not args.gpu else ""))

    model_config = get_model_config(args.model, gpu_type)
    num_prompts = args.batch_sizes[-1] * args.num_batches + 1

    log(f"Model type:     {args.model}")
    log(f"Model path:     {args.model_path}")
    log(f"Batch sizes:    {args.batch_sizes}")
    log(f"Total prompts:  {num_prompts} ({args.num_batches} batches per sweep)")

    prompts = load_prompts_from_sharegpt(num_prompts)

    server_args = []

    if args.disable_cuda_graph:
        server_args.append("--disable-cuda-graph")

    if args.disable_radix_cache:
        server_args.append("--disable-radix-cache")
        log("Added --disable-radix-cache for ragged prefill collection")

    if args.enable_deterministic_inference:
        server_args.append("--enable-deterministic-inference")
        log("Added --enable-deterministic-inference for paged prefill collection")

    server_args.extend(model_config["server_flags"])

    # Auto-detect TP size from CUDA_VISIBLE_DEVICES (set by gpu-lock) and override if needed
    detected_tp = detect_tp_size()
    if detected_tp is not None:
        # Extract TP size from current server_args
        config_tp = None
        for i, arg in enumerate(server_args):
            if arg in ("--tp-size", "--tp") and i + 1 < len(server_args):
                try:
                    config_tp = int(server_args[i + 1])
                except ValueError:
                    pass
                break
        if config_tp != detected_tp:
            # Remove existing --tp-size / --tp flags and replace
            filtered = []
            skip_next = False
            for arg in server_args:
                if skip_next:
                    skip_next = False
                    continue
                if arg in ("--tp-size", "--tp"):
                    skip_next = True
                    continue
                filtered.append(arg)
            server_args = filtered + ["--tp-size", str(detected_tp)]
            log(
                f"TP size: {detected_tp} (auto-detected from CUDA_VISIBLE_DEVICES, overrides config TP={config_tp})"
            )
        else:
            log(f"TP size: {detected_tp} (matches config)")
    log(f"Server args:    {server_args}")

    process = popen_launch_server(
        args.model_path,
        args.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args,
        env={"SGLANG_RECORD_STEP_TIME": "1", "SGLANG_TEST_REQUEST_TIME_STATS": "1", **os.environ},
    )

    try:
        for batch_size in args.batch_sizes:
            log(f"Running benchmark with batch size {batch_size}")
            run_benchmark(args.base_url, prompts[: batch_size * args.num_batches], batch_size)
    except Exception as e:
        log(f"Benchmark failed: {e}")
        raise
    finally:
        log("Shutting down server...")
        kill_process_tree(process.pid)
        time.sleep(3)
        log("Server shutdown complete")


if __name__ == "__main__":
    main()
