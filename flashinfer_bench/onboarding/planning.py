"""Build the small serializable plan shared by both onboarding stages."""

from __future__ import annotations

from typing import Any


def build_stage_plan(
    *,
    stage: str,
    config: dict[str, Any],
    reviewed_definitions: list[dict[str, Any]] | None = None,
    pass_modes: list[str] | None = None,
    page_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible SGLang stage plan."""
    if stage not in {"definitions", "workloads"}:
        raise ValueError("stage must be definitions or workloads")
    model_name = _non_empty_string(config.get("model_name"), "model_name")
    tp_size = _positive_int(config.get("tp_size"), "tp_size")
    max_new_tokens = _positive_int(config.get("max_new_tokens"), "max_new_tokens")
    batch_sizes = _positive_int_list(config.get("batch_sizes"), "batch_sizes")
    request_scenarios = _build_synthetic_scenarios(
        medium_input_len=_positive_int(config.get("isl", 1024), "isl"),
        output_len=_positive_int(config.get("osl", 8), "osl"),
        batch_sizes=batch_sizes,
        range_ratio=_ratio(
            config.get("random_range_ratio", 1.0),
            "random_range_ratio",
        ),
        requests_per_concurrency=_positive_int(
            config.get("requests_per_concurrency", 10),
            "requests_per_concurrency",
        ),
        seed=_integer(config.get("seed", 0), "seed"),
    )
    plan = {
        "stage": stage,
        "model_name": model_name,
        "runtime": {
            "model_name": model_name,
            "tp_size": tp_size,
        },
        "pass_modes": pass_modes or ["default"],
        "page_sizes": page_sizes or [],
        "request_scenarios": request_scenarios,
        "sampling": {"max_new_tokens": max_new_tokens},
        "supplemental_runs": _supplemental_runs(config.get("supplemental_runs")),
        "sglang": {
            "disable_cuda_graph": bool(config.get("disable_cuda_graph", True)),
            "enable_piecewise_cuda_graph": config.get("enable_piecewise_cuda_graph"),
            "force_flashinfer_backends": bool(config.get("force_flashinfer_backends", True)),
            "mem_fraction_static": float(config.get("mem_fraction_static", 0.7)),
            "cuda_graph_max_bs": config.get("cuda_graph_max_bs"),
            "engine_kwargs": _engine_kwargs(config.get("engine_kwargs")),
        },
    }
    if reviewed_definitions is not None:
        plan["reviewed_definitions"] = reviewed_definitions
        plan["max_new_workloads"] = _positive_int(
            config.get("max_new_workloads"), "max_new_workloads"
        )
    return plan


def _build_synthetic_scenarios(
    *,
    medium_input_len: int,
    output_len: int,
    batch_sizes: list[int],
    range_ratio: float,
    requests_per_concurrency: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return the bounded request matrix used by both pipeline stages."""
    scenarios: list[dict[str, Any]] = []
    input_lengths = list(dict.fromkeys((128, medium_input_len)))
    scenario_index = 0
    for input_len in input_lengths:
        for batch_size in batch_sizes:
            scenarios.append(
                _synthetic_scenario(
                    name=f"tokens_i{input_len}_o{output_len}_bs{batch_size}",
                    input_len=input_len,
                    output_len=output_len,
                    batch_size=batch_size,
                    range_ratio=range_ratio,
                    requests_per_concurrency=requests_per_concurrency,
                    seed=seed + scenario_index,
                )
            )
            scenario_index += 1

    scenarios.append(
        _synthetic_scenario(
            name=f"long_context_max8192_o{output_len}_bs1",
            input_len=8192,
            output_len=output_len,
            batch_size=1,
            range_ratio=range_ratio,
            requests_per_concurrency=requests_per_concurrency,
            seed=seed + scenario_index,
            context_fraction=0.25,
        )
    )
    scenario_index += 1

    shared_batch_size = max((size for size in batch_sizes if size <= 8), default=1)
    shared_input_len = max(medium_input_len, 128)
    scenarios.append(
        _synthetic_scenario(
            name=(
                f"shared_prefix_i{shared_input_len}_p{shared_input_len * 3 // 4}_"
                f"o{output_len}_bs{shared_batch_size}"
            ),
            input_len=shared_input_len // 4,
            output_len=output_len,
            batch_size=shared_batch_size,
            range_ratio=1.0,
            requests_per_concurrency=requests_per_concurrency,
            seed=seed + scenario_index,
            shared_prefix_len=shared_input_len * 3 // 4,
        )
    )
    return scenarios


def _synthetic_scenario(
    *,
    name: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    range_ratio: float,
    requests_per_concurrency: int,
    seed: int,
    context_fraction: float | None = None,
    shared_prefix_len: int | None = None,
) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "name": name,
        "source": "inferencex_fixed_seq",
        "random_input_len": input_len,
        "random_output_len": output_len,
        "random_range_ratio": range_ratio,
        "random_prefix_len": shared_prefix_len or 0,
        "num_prompts": batch_size * requests_per_concurrency,
        "max_concurrency": batch_size,
        "seed": seed,
    }
    if context_fraction is not None:
        scenario["context_fraction"] = context_fraction
    return scenario


def _supplemental_runs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("supplemental_runs must be a list")
    runs = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"supplemental_runs[{index}] must be an object")
        name = _non_empty_string(item.get("name"), f"supplemental_runs[{index}].name")
        params = item.get("sampling_params")
        if not isinstance(params, dict):
            raise ValueError(f"supplemental_runs[{index}].sampling_params must be an object")
        runs.append(
            {
                "name": name,
                "sampling_params": dict(params),
                "use_scenario_tokens": bool(item.get("use_scenario_tokens", False)),
            }
        )
    return runs


def _engine_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("engine_kwargs must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("engine_kwargs keys must be non-empty strings")
        if type(item) not in {str, int, float, bool} and item is not None:
            raise ValueError(f"engine_kwargs.{key} must be a JSON scalar")
    return dict(value)


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    ratio = float(value)
    if not 0 < ratio <= 1:
        raise ValueError(f"{field} must be greater than 0 and at most 1")
    return ratio


def _positive_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return [_positive_int(item, f"{field}[{index}]") for index, item in enumerate(value)]
