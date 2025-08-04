"""
Usage:
    python3 bench_sharegpt.py \
        --model-path /raid/catalyst/models/Llama-3.1-8B-Instruct/ \
        --num-prompts 64 >& bench_sharegpt.log

Arguments:
--model-path: Path to the model directory that can be served by SGLang.
--num-prompts: Batch size & Number of first-round human prompts to load from the ShareGPT dataset.
"""

import os
import json
import time
import asyncio
import argparse
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from types import SimpleNamespace

from sglang.bench_serving import benchmark, set_global_args
from sglang.test.test_utils import (
    popen_launch_server,
    kill_process_tree,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
)


@dataclass
class TestRequest:
    prompt: str
    prompt_len: int
    output_len: int
    image_data: Optional[str] = None


def log(msg):
    print(f"[VIVIAN] {time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def load_prompts_from_sharegpt(n: int):
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        streaming=True
    )
    prompts = []

    for example in ds:
        conv = example.get("conversations", [])
        if conv and conv[0]["from"].lower() == "human":
            prompts.append(conv[0]["value"])
        if len(prompts) >= n:
            break

    log(f"Loaded {len(prompts)} prompts from ShareGPT.")
    return prompts


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []

import math

def run_benchmark(base_url, prompts, batch_size, label=""):
    tokenizer = DummyTokenizer()
    set_global_args(SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=True,
        return_logprob=False,
        backend="sglang",
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        output_details=False,
    ))

    num_batches = math.ceil(len(prompts) / batch_size)
    all_results = []

    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        input_requests = [
            TestRequest(prompt=p, prompt_len=0, output_len=512)
            for p in batch_prompts
        ]

        print(f"[{label}] Running batch {i + 1}/{num_batches} with {len(input_requests)} prompts...")
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
                extra_request_body={},
                profile=None,
            )
        )
        all_results.append(results)

    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to use from ShareGPT")
    args = parser.parse_args()

    batch_size = args.num_prompts
    base_url = "http://127.0.0.1:20000"

    # Load ShareGPT prompts
    prompts = load_prompts_from_sharegpt(args.num_prompts)

    # Launch the server
    process = popen_launch_server(
        args.model_path,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[
            # "--cuda-graph-max-bs", batch_size,  # uncommented to enable CUDA graph
            "--chunked-prefill-size", "-1",
            "--max-prefill-tokens", "1024",
            "--disable-cuda-graph",
            "--max-running-requests", batch_size,
            "--tp-size", 1,
            "--quantization", "fp8",
        ],
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            **os.environ,
        },
    )

    try:
        # log("Warming up model with one batch...")
        # run_benchmark(base_url, prompts[:batch_size], batch_size, label="warmup")

        # log("Running main benchmark...")
        results = run_benchmark(base_url, prompts, batch_size, label="main")

        log("If enabled, kernel trace should be available in your log directory.")

    finally:
        kill_process_tree(process.pid)
        time.sleep(3)
        log("Server shutdown complete.")


if __name__ == "__main__":
    main()
