"""
Benchmark SGLang serving throughput using the ShareGPT dataset.

Launches an SGLang server, sends batched requests from ShareGPT, and reports
throughput and latency statistics across a sweep of batch sizes. Model server
flags are loaded from model_configs.json.

Usage:
    python3 bench_sharegpt.py --model llama-3.1-8b   --model-path /path/to/Llama-3.1-8B-Instruct    --disable-cuda-graph
    python3 bench_sharegpt.py --model deepseek-v3    --model-path /path/to/DeepSeek-V3              --disable-cuda-graph
    python3 bench_sharegpt.py --model deepseek-v3.2  --model-path /path/to/DeepSeek-V3.2            --disable-cuda-graph
    python3 bench_sharegpt.py --model qwen3-30b      --model-path /path/to/Qwen3-30B-A3B            --disable-cuda-graph

    # Disable radix cache (e.g. for ragged prefill traces)
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B-Instruct      --disable-radix-cache --disable-cuda-graph

    # Custom batch sizes and number of batches
    python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B-Instruct --batch-sizes 32 128 --num-batches 8 --disable-cuda-graph

Environment variables (optional, for flashinfer-bench tracing):
    FIB_ENABLE_APPLY=1        Enable the flashinfer-bench apply hook
    FIB_DATASET_PATH=<dir>    Directory to write flashinfer trace data

    Example:
        FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=/path/to/traces/ python3 bench_sharegpt.py --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B-Instruct --disable-cuda-graph
"""

import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_CONFIG_FILE = Path(__file__).parent / "model_configs.json"

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
        return json.load(f)


def get_model_config(model_type: str) -> dict:
    """Return server flags for a given model type loaded from model_configs.json."""
    configs = load_model_configs()
    key = model_type.lower()
    if key not in configs:
        raise ValueError(
            f"Unsupported model type: {model_type!r}. Available in {_CONFIG_FILE.name}: {list(configs.keys())}"
        )
    return configs[key]


def log(msg: str) -> None:
    print(f"[BENCHMARK] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def load_prompts_from_sharegpt(n: int) -> List[str]:
    """Load n prompts from the ShareGPT dataset."""
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True,
    )
    prompts = []
    for example in ds:
        conv = example.get("conversations", [])
        if conv:
            item = conv[0]
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    item = {}
            if isinstance(item, dict) and item.get("from", "").lower() == "human":
                prompts.append(item["value"])
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
        warmup_requests=3,
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
    available_models = list(load_model_configs().keys())
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model configuration to use. Available: {available_models}",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model weights directory"
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Enable ragged prefill collection (adds --disable-radix-cache to server args)",
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

    model_config = get_model_config(args.model)
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

    server_args.extend(model_config["server_flags"])
    log(f"Server arguments: {server_args}")

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
