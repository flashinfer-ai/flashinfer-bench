"""InferenceX fixed-sequence request generation and dispatch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


INFERENCEX_REVISION = "b1f05e9ac71859b83ff316e904289becf8dcd670"
INFERENCEX_GENERATOR = "utils/bench_serving/benchmark_serving.py::sample_random_requests"
INFERENCEX_RECIPE = "benchmarks/benchmark_lib.sh::run_benchmark_serving"


def inferencex_fixed_seq_request_contract() -> dict[str, Any]:
    """Describe the pinned InferenceX request semantics used by onboarding."""
    return {
        "project": "InferenceX",
        "revision": INFERENCEX_REVISION,
        "generator": INFERENCEX_GENERATOR,
        "recipe": INFERENCEX_RECIPE,
        "dataset": "random",
        "request_rate": "inf",
        "ignore_eos": True,
        "use_chat_template": False,
        "generator_workers": 1,
        "transport": "in_process_sglang_engine",
        "performance_metrics": False,
    }


@dataclass(frozen=True)
class SyntheticRequest:
    """One reproducible synthetic request after tokenizer round-tripping."""

    prompt: str
    input_ids: tuple[int, ...]
    output_len: int


def request_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the digest of a request manifest, excluding its digest field."""
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_request_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable digest to a JSON-compatible request manifest."""
    finalized = dict(manifest)
    finalized["manifest_sha256"] = request_manifest_digest(finalized)
    return finalized


def sample_random_token_requests(
    tokenizer: Any,
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    seed: int | None = None,
    prefix_len: int = 0,
) -> list[SyntheticRequest]:
    """Generate requests with InferenceX's serial random-dataset algorithm."""
    import numpy as np

    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")
    rng = np.random.RandomState(seed)
    vocab_size = int(tokenizer.vocab_size)
    prefix_token_ids = rng.randint(0, vocab_size, size=prefix_len).tolist()

    def sample_uniform(seq_len: int) -> list[int]:
        lower = int(seq_len * range_ratio)
        return rng.randint(lower, seq_len + 1, size=num_prompts).tolist()

    input_lens = sample_uniform(input_len)
    output_lens = sample_uniform(output_len)
    offsets = rng.randint(0, vocab_size, size=num_prompts)
    local_rng = np.random.RandomState(rng.get_state()[1][:4].tolist())

    requests: list[SyntheticRequest] = []
    for index, target_len in enumerate(input_lens):
        target_prompt_len = prefix_len + target_len
        token_ids = prefix_token_ids + [
            (int(offsets[index]) + index + position) % vocab_size
            for position in range(target_len)
        ]
        prompt = tokenizer.decode(token_ids)

        # Match InferenceX's text round-trip before storing canonical token IDs.
        for _ in range(10):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) < target_prompt_len:
                missing = target_prompt_len - len(token_ids)
                token_ids.extend(local_rng.randint(0, vocab_size, size=missing).tolist())
            elif len(token_ids) > target_prompt_len:
                token_ids = token_ids[:target_prompt_len]
            else:
                break
            prompt = tokenizer.decode(token_ids)

        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        requests.append(
            SyntheticRequest(
                prompt=prompt,
                input_ids=tuple(int(token_id) for token_id in token_ids),
                output_len=int(output_lens[index]),
            )
        )
    return requests


def run_engine_requests(
    engine: Any,
    requests: list[SyntheticRequest],
    sampling_params: list[dict[str, Any]],
    *,
    max_concurrency: int,
) -> list[Any]:
    """Dispatch requests in bounded batches through an in-process SGLang Engine."""
    if len(requests) != len(sampling_params):
        raise ValueError("requests and sampling_params must have equal length")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")

    results: list[Any] = []
    for start in range(0, len(requests), max_concurrency):
        request_batch = requests[start : start + max_concurrency]
        parameter_batch = sampling_params[start : start + max_concurrency]
        result = engine.generate(
            input_ids=[list(request.input_ids) for request in request_batch],
            sampling_params=parameter_batch,
        )
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
    return results
