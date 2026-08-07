"""Run reviewed prompt scenarios against an SGLang Engine."""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from flashinfer_bench.serve.inferencex_requests import (
    SyntheticRequest,
    finalize_request_manifest,
    inferencex_fixed_seq_request_contract,
    request_manifest_digest,
    run_engine_requests,
    sample_random_token_requests,
)

_HF_CONFIG_OVERRIDE: dict[str, Any] | None = None
_HF_CONFIG_PATCHED = False


@dataclass
class _RequestBatch:
    run_name: str
    scenario_name: str
    max_concurrency: int
    requests: list[SyntheticRequest]
    sampling_params: list[dict[str, Any]]


def run_sglang_model(plan: dict[str, Any]) -> dict[str, Any]:
    """Run all SGLang passes and return the exact request manifest used."""
    request_batches = _request_batches_from_manifest(plan)
    for paged, page_size in _execution_passes(plan):
        request_batches = _run_sglang_pass(
            plan,
            paged=paged,
            page_size=page_size,
            request_batches=request_batches,
        )
    if request_batches is None:
        raise RuntimeError("SGLang execution produced no request batches")
    manifest = _build_request_manifest(plan, request_batches)
    expected = plan.get("request_manifest")
    if isinstance(expected, dict) and manifest["manifest_sha256"] != expected.get(
        "manifest_sha256"
    ):
        raise RuntimeError("SGLang request manifest changed before execution")
    return manifest


def _execution_passes(plan: dict[str, Any]) -> list[tuple[bool, int | None]]:
    modes = set(plan.get("pass_modes") or ["default"])
    page_sizes = plan.get("page_sizes")
    if not isinstance(page_sizes, list) or not page_sizes:
        page_sizes = [None]
    paged = [(True, int(size) if size is not None else None) for size in page_sizes]
    if "both" in modes:
        return [(False, None), *paged]
    passes = []
    if "default" in modes:
        passes.append((False, None))
    if "paged" in modes:
        passes.extend(paged)
    return passes or [(False, None)]


def _run_sglang_pass(
    plan: dict[str, Any],
    *,
    paged: bool,
    page_size: int | None,
    request_batches: list[_RequestBatch] | None,
) -> list[_RequestBatch]:
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("stage plan missing runtime")
    sglang_config = plan.get("sglang") if isinstance(plan.get("sglang"), dict) else {}
    reviewed_kwargs = sglang_config.get("engine_kwargs") or {}
    if not isinstance(reviewed_kwargs, dict):
        raise ValueError("sglang.engine_kwargs must be an object")
    reviewed_kwargs = dict(reviewed_kwargs)
    hf_config_override = _pop_decrypted_config_json(reviewed_kwargs)

    enable_piecewise = sglang_config.get("enable_piecewise_cuda_graph")
    enable_piecewise = bool(paged if enable_piecewise is None else enable_piecewise)
    engine_kwargs: dict[str, Any] = {
        "model_path": str(runtime.get("model_name") or plan.get("model_name")),
        "tp_size": int(runtime.get("tp_size") or 1),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "attention_backend": "flashinfer",
        "disable_cuda_graph": bool(sglang_config.get("disable_cuda_graph", True)),
        "disable_piecewise_cuda_graph": not enable_piecewise,
        "log_level": "info",
    }
    if bool(sglang_config.get("force_flashinfer_backends", True)):
        engine_kwargs.update(
            {
                "prefill_attention_backend": "flashinfer",
                "decode_attention_backend": "flashinfer",
                "sampling_backend": "flashinfer",
            }
        )
    if enable_piecewise:
        engine_kwargs["enable_deterministic_inference"] = True
    if paged and page_size is not None:
        engine_kwargs["page_size"] = page_size
    if type(sglang_config.get("cuda_graph_max_bs")) is int:
        engine_kwargs["cuda_graph_max_bs"] = int(sglang_config["cuda_graph_max_bs"])
    if isinstance(sglang_config.get("mem_fraction_static"), (int, float)):
        engine_kwargs["mem_fraction_static"] = float(sglang_config["mem_fraction_static"])
    protected = {
        "model_path",
        "tp_size",
        "trust_remote_code",
        "disable_cuda_graph",
        "disable_piecewise_cuda_graph",
        "page_size",
    }
    overlap = sorted(protected & set(reviewed_kwargs))
    if overlap:
        raise ValueError(f"engine_kwargs cannot override managed fields: {overlap}")
    engine_kwargs.update(reviewed_kwargs)
    engine_kwargs = _filter_supported_engine_kwargs(
        engine_kwargs,
        optional={
            "disable_piecewise_cuda_graph",
            "enable_deterministic_inference",
            "cuda_graph_max_bs",
            "mem_fraction_static",
        },
    )

    old_paged = os.environ.get("SGLANG_FLASHINFER_USE_PAGED")
    old_mode = os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE")
    if paged:
        os.environ["SGLANG_FLASHINFER_USE_PAGED"] = "1"
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = (
            f"paged_ps{page_size}" if page_size else "paged"
        )
    else:
        os.environ.pop("SGLANG_FLASHINFER_USE_PAGED", None)
        os.environ["FLASHINFER_TRACE_ACTIVE_PROBE_MODE"] = "default"
    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    executed_batches: list[_RequestBatch] | None = None
    try:
        print(
            "[flashinfer_bench.onboarding] sglang pass: "
            + json.dumps(
                {"paged": paged, "page_size": page_size, "engine_kwargs": engine_kwargs},
                sort_keys=True,
            ),
            flush=True,
        )
        with _sglang_dumper_environment(sglang_config):
            import sglang as sgl

            _install_hf_config_override(hf_config_override)
            engine = sgl.Engine(**engine_kwargs)
            try:
                executed_batches = _run_generation_requests(
                    engine, plan, request_batches=request_batches
                )
            finally:
                _shutdown_engine(engine)
    finally:
        _restore_env("SGLANG_FLASHINFER_USE_PAGED", old_paged)
        _restore_env("FLASHINFER_TRACE_ACTIVE_PROBE_MODE", old_mode)
    if executed_batches is None:
        raise RuntimeError("SGLang pass completed without request batches")
    return executed_batches


@contextmanager
def _sglang_dumper_environment(config: dict[str, Any]) -> Iterator[None]:
    """Enable SGLang's generic module input/output dumper for one bounded pass."""
    output_dir = config.get("dumper_output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        yield
        return

    filter_expression = config.get("dumper_filter")
    if not isinstance(filter_expression, str) or not filter_expression:
        layers = config.get("dumper_layers")
        if not isinstance(layers, list) or not layers or not all(
            type(item) is int and item >= 0 for item in layers
        ):
            raise ValueError(
                "SGLang dumper requires dumper_filter or non-negative dumper_layers"
            )
        filter_expression = f"layer_id in {list(dict.fromkeys(layers))!r}"
    active_mode = os.environ.get("FLASHINFER_TRACE_ACTIVE_PROBE_MODE", "default")
    experiment = "onboarding_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", active_mode)
    updates = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": output_dir,
        "DUMPER_EXP_NAME": experiment,
        "DUMPER_NON_INTRUSIVE_MODE": "all",
        "DUMPER_FILTER": filter_expression,
        "DUMPER_ENABLE_OUTPUT_FILE": "1",
        "DUMPER_ENABLE_OUTPUT_CONSOLE": "0",
        "DUMPER_ENABLE_VALUE": "1",
        "DUMPER_ENABLE_GRAD": "0",
        "DUMPER_ENABLE_MODEL_VALUE": "0",
        "DUMPER_ENABLE_MODEL_GRAD": "0",
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    _refresh_loaded_dumper()
    try:
        yield
    finally:
        for name, value in previous.items():
            _restore_env(name, value)
        _refresh_loaded_dumper()


def _refresh_loaded_dumper() -> None:
    module = sys.modules.get("sglang.srt.debug_utils.dumper")
    if module is None:
        return
    module.dumper.reset()
    module.dumper.configure(**asdict(module.DumperConfig.from_env()))


def _run_generation_requests(
    engine: Any,
    plan: dict[str, Any],
    *,
    request_batches: list[_RequestBatch] | None = None,
) -> list[_RequestBatch]:
    if request_batches is None:
        request_batches = _build_request_batches(engine, plan)
    for batch in request_batches:
        print(
            f"[flashinfer_bench.onboarding] request: "
            f"{batch.run_name}/{batch.scenario_name}",
            flush=True,
        )
        run_engine_requests(
            engine,
            batch.requests,
            batch.sampling_params,
            max_concurrency=batch.max_concurrency,
        )
    return request_batches


def _build_request_batches(engine: Any, plan: dict[str, Any]) -> list[_RequestBatch]:
    scenarios = _request_scenarios(plan)
    base = plan.get("sampling")
    if not isinstance(base, dict) or type(base.get("max_new_tokens")) is not int:
        raise ValueError("stage plan missing sampling.max_new_tokens")
    runs = [("base", dict(base), True)]
    runs.extend(
        (item["name"], item["sampling_params"], item["use_scenario_tokens"])
        for item in _supplemental_runs(plan)
    )
    batches = []
    for name, parameters, use_scenario_tokens in runs:
        for scenario in scenarios:
            requests, sampling_params = _synthetic_token_batch(
                engine,
                scenario,
                dict(parameters),
                use_scenario_tokens=use_scenario_tokens,
            )
            batches.append(
                _RequestBatch(
                    run_name=name,
                    scenario_name=str(scenario["name"]),
                    max_concurrency=int(scenario["max_concurrency"]),
                    requests=requests,
                    sampling_params=sampling_params,
                )
            )
    return batches


def _build_request_manifest(
    plan: dict[str, Any], batches: list[_RequestBatch]
) -> dict[str, Any]:
    groups = []
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    for batch in batches:
        requests = []
        for index, (request, sampling_params) in enumerate(
            zip(batch.requests, batch.sampling_params, strict=True)
        ):
            request_count += 1
            input_tokens += len(request.input_ids)
            output_tokens += request.output_len
            requests.append(
                {
                    "request_id": (
                        f"{batch.run_name}/{batch.scenario_name}/{index:04d}"
                    ),
                    "input_ids": list(request.input_ids),
                    "input_len": len(request.input_ids),
                    "output_len": request.output_len,
                    "sampling_params": sampling_params,
                }
            )
        groups.append(
            {
                "run_name": batch.run_name,
                "scenario_name": batch.scenario_name,
                "max_concurrency": batch.max_concurrency,
                "requests": requests,
            }
        )
    return finalize_request_manifest(
        {
            "schema_version": 2,
            "run_id": str(plan.get("run_id") or ""),
            "model_name": str(plan.get("model_name") or ""),
            "request_contract": inferencex_fixed_seq_request_contract(),
            "execution_contract": _execution_contract(plan),
            "summary": {
                "groups": len(groups),
                "requests": request_count,
                "input_tokens": input_tokens,
                "requested_output_tokens": output_tokens,
            },
            "request_groups": groups,
        }
    )


def _request_batches_from_manifest(
    plan: dict[str, Any],
) -> list[_RequestBatch] | None:
    manifest = plan.get("request_manifest")
    if manifest is None:
        return None
    if not isinstance(manifest, dict):
        raise ValueError("request_manifest must be an object")
    if manifest.get("schema_version") != 2:
        raise ValueError("request_manifest.schema_version must be 2")
    if manifest.get("request_contract") != inferencex_fixed_seq_request_contract():
        raise ValueError("request_manifest.request_contract does not match InferenceX")
    expected_digest = manifest.get("manifest_sha256")
    if not isinstance(expected_digest, str) or request_manifest_digest(manifest) != expected_digest:
        raise ValueError("request_manifest.manifest_sha256 is invalid")
    if manifest.get("run_id") != plan.get("run_id"):
        raise ValueError("request_manifest.run_id does not match the stage plan")
    if manifest.get("model_name") != plan.get("model_name"):
        raise ValueError("request_manifest.model_name does not match the stage plan")
    if manifest.get("execution_contract") != _execution_contract(plan):
        raise ValueError(
            "request_manifest.execution_contract does not match the stage plan"
        )
    groups = manifest.get("request_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("request_manifest.request_groups must be a non-empty list")

    batches = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"request_manifest group #{group_index} must be an object")
        raw_requests = group.get("requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError(
                f"request_manifest group #{group_index} must contain requests"
            )
        requests = []
        sampling_params = []
        for request_index, item in enumerate(raw_requests):
            if not isinstance(item, dict):
                raise ValueError(
                    f"request_manifest request #{group_index}/{request_index} must be an object"
                )
            input_ids = item.get("input_ids")
            parameters = item.get("sampling_params")
            output_len = item.get("output_len")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or not all(type(token_id) is int and token_id >= 0 for token_id in input_ids)
                or type(output_len) is not int
                or output_len < 1
                or not isinstance(parameters, dict)
            ):
                raise ValueError(
                    f"request_manifest request #{group_index}/{request_index} is invalid"
                )
            requests.append(
                SyntheticRequest(
                    prompt="",
                    input_ids=tuple(input_ids),
                    output_len=output_len,
                )
            )
            sampling_params.append(dict(parameters))
        batches.append(
            _RequestBatch(
                run_name=str(group.get("run_name") or "base"),
                scenario_name=str(group.get("scenario_name") or f"group_{group_index}"),
                max_concurrency=_positive_int(
                    group.get("max_concurrency"),
                    f"request_manifest group #{group_index}.max_concurrency",
                ),
                requests=requests,
                sampling_params=sampling_params,
            )
        )
    return batches


def _execution_contract(plan: dict[str, Any]) -> dict[str, Any]:
    """Return stable runtime settings that must match across capture stages."""
    runtime = plan.get("runtime")
    sglang = plan.get("sglang")
    if not isinstance(runtime, dict) or not isinstance(sglang, dict):
        raise ValueError("stage plan requires runtime and sglang objects")
    stable_sglang_fields = (
        "disable_cuda_graph",
        "enable_piecewise_cuda_graph",
        "force_flashinfer_backends",
        "mem_fraction_static",
        "cuda_graph_max_bs",
        "engine_kwargs",
    )
    return {
        "runtime": dict(runtime),
        "sglang": {
            name: sglang.get(name)
            for name in stable_sglang_fields
            if name in sglang
        },
    }


def _request_scenarios(plan: dict[str, Any]) -> list[dict[str, Any]]:
    value = plan.get("request_scenarios")
    if not isinstance(value, list) or not value:
        raise ValueError("stage plan missing request_scenarios")
    scenarios = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"request_scenarios[{index}] must be an object")
        source = str(item.get("source") or "")
        if source != "inferencex_fixed_seq":
            raise ValueError(f"request_scenarios[{index}].source is unsupported: {source}")
        scenario = {
            "name": str(item.get("name") or f"scenario_{index + 1}"),
            "source": source,
            "random_input_len": _positive_int(
                item.get("random_input_len"),
                f"request_scenarios[{index}].random_input_len",
            ),
            "random_output_len": _positive_int(
                item.get("random_output_len"),
                f"request_scenarios[{index}].random_output_len",
            ),
            "random_range_ratio": _range_ratio(
                item.get("random_range_ratio"),
                f"request_scenarios[{index}].random_range_ratio",
            ),
            "random_prefix_len": _non_negative_int(
                item.get("random_prefix_len"),
                f"request_scenarios[{index}].random_prefix_len",
            ),
            "num_prompts": _positive_int(
                item.get("num_prompts"), f"request_scenarios[{index}].num_prompts"
            ),
            "max_concurrency": _positive_int(
                item.get("max_concurrency"),
                f"request_scenarios[{index}].max_concurrency",
            ),
            "seed": _integer(item.get("seed"), f"request_scenarios[{index}].seed"),
        }
        context_fraction = item.get("context_fraction")
        if context_fraction is not None:
            scenario["context_fraction"] = _range_ratio(
                context_fraction, f"request_scenarios[{index}].context_fraction"
            )
        scenarios.append(scenario)
    return scenarios


def _synthetic_token_batch(
    engine: Any, scenario: dict[str, Any], parameters: dict[str, Any], *, use_scenario_tokens: bool
) -> tuple[list[SyntheticRequest], list[dict[str, Any]]]:
    input_len = _effective_input_len(engine, scenario)
    requests = sample_random_token_requests(
        _engine_tokenizer(engine),
        num_prompts=int(scenario["num_prompts"]),
        input_len=input_len,
        output_len=int(scenario["random_output_len"]),
        range_ratio=float(scenario["random_range_ratio"]),
        seed=int(scenario["seed"]),
        prefix_len=int(scenario["random_prefix_len"]),
    )
    return requests, _sampling_params(
        parameters, requests, use_scenario_tokens=use_scenario_tokens
    )


def _sampling_params(
    parameters: dict[str, Any], requests: list[Any], *, use_scenario_tokens: bool
) -> list[dict[str, Any]]:
    sampling_params = []
    for request in requests:
        item = dict(parameters)
        if use_scenario_tokens:
            item["max_new_tokens"] = request.output_len
        else:
            item.setdefault("max_new_tokens", request.output_len)
        item["ignore_eos"] = True
        sampling_params.append(item)
    return sampling_params


def _effective_input_len(engine: Any, scenario: dict[str, Any]) -> int:
    target = int(scenario["random_input_len"])
    prefix_len = int(scenario["random_prefix_len"])
    context_len = _engine_context_length(engine)
    if context_len is not None:
        available = context_len - int(scenario["random_output_len"]) - prefix_len - 1
        if available < 1:
            raise ValueError("request scenario does not fit the model context length")
        target = min(target, available)
    fraction = scenario.get("context_fraction")
    if fraction is None:
        return target
    if context_len is None:
        return target
    available = int(context_len * float(fraction)) - prefix_len
    return max(min(target, available), 1)


def _engine_context_length(engine: Any) -> int | None:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    model_config = getattr(tokenizer_manager, "model_config", None)
    candidates = [model_config, getattr(model_config, "hf_config", None)]
    for candidate in candidates:
        for name in ("context_len", "context_length", "max_position_embeddings"):
            value = getattr(candidate, name, None)
            if type(value) is int and value > 0:
                return value
    return None


def _engine_tokenizer(engine: Any) -> Any:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    tokenizer = getattr(tokenizer_manager, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("SGLang Engine did not expose its tokenizer")
    return tokenizer


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _range_ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise ValueError(f"{field} must be greater than 0 and at most 1")
    return ratio


def _supplemental_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    value = plan.get("supplemental_runs")
    if not isinstance(value, list):
        raise ValueError("stage plan missing supplemental_runs")
    return [item for item in value if isinstance(item, dict)]


def _supported_engine_kwarg_names() -> set[str]:
    from sglang.srt.server_args import ServerArgs

    return set(inspect.signature(ServerArgs).parameters)


def _filter_supported_engine_kwargs(
    values: dict[str, Any], *, optional: set[str]
) -> dict[str, Any]:
    supported = _supported_engine_kwarg_names()
    unsupported = sorted(set(values) - supported)
    required = sorted(set(unsupported) - optional)
    if required:
        raise RuntimeError(f"SGLang Engine does not support kwargs: {required}")
    return {key: value for key, value in values.items() if key not in unsupported}


def _pop_decrypted_config_json(engine_kwargs: dict[str, Any]) -> dict[str, Any] | None:
    raw = engine_kwargs.pop("decrypted_config_json", None)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("engine_kwargs.decrypted_config_json must be a non-empty JSON string")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("engine_kwargs.decrypted_config_json must decode to an object")
    return value


def _install_hf_config_override(value: dict[str, Any] | None) -> None:
    if value is None:
        return
    global _HF_CONFIG_OVERRIDE, _HF_CONFIG_PATCHED
    _HF_CONFIG_OVERRIDE = value
    if _HF_CONFIG_PATCHED:
        return
    from transformers import PretrainedConfig

    original = PretrainedConfig.from_dict.__func__

    @classmethod
    def patched(cls, config_dict: dict[str, Any], **kwargs: Any) -> Any:
        return original(cls, dict(_HF_CONFIG_OVERRIDE or config_dict), **kwargs)

    PretrainedConfig.from_dict = patched
    _HF_CONFIG_PATCHED = True


def _shutdown_engine(engine: Any) -> None:
    for name in ("shutdown", "release", "close"):
        method = getattr(engine, name, None)
        if callable(method):
            method()
            return


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
