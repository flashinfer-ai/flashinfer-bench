"""Inspect a Transformers model for FlashInfer-Bench-replaceable operators.

This variant only checks whether a matching definition exists in ``definitions/``.
It does not require any solutions or traces to be present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_ENV_PYTHONS = (
    "/data/workspace/airulan/conda_envs/fib_e2e/bin/python",
    "/data/workspace/airulan/conda_envs/fib/bin/python",
)
DEFAULT_DEFINITIONS_ROOT = REPO_ROOT / "flashinfer_trace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Hugging Face model and report which GEMM/RMSNorm operators are "
            "replaceable by existing FlashInfer-Bench definitions."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-mode",
        choices=["empty", "pretrained"],
        default="empty",
        help="Use init_empty_weights skeleton or load full pretrained weights",
    )
    parser.add_argument(
        "--definitions-root",
        "--trace-set-path",
        dest="definitions_root",
        default=str(DEFAULT_DEFINITIONS_ROOT),
        help=(
            "Path to a trace-set root or a definitions directory. Only definitions/ are read. "
            f"Default: {DEFAULT_DEFINITIONS_ROOT}"
        ),
    )
    parser.add_argument(
        "--sample-module-names",
        type=int,
        default=8,
        help="How many module names to keep per matched definition",
    )
    parser.add_argument(
        "--include-modules",
        action="store_true",
        help="Include the full named module list in the JSON output",
    )
    parser.add_argument("--output-json", help="Optional path to write the inspection summary JSON")
    return parser


def _expected_env_text() -> str:
    return " or ".join(EXPECTED_ENV_PYTHONS)


def _require_module(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency '{name}' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {_expected_env_text()}"
        ) from exc


def _load_model_skeleton(args: argparse.Namespace):
    _require_module("torch")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers' in the current interpreter.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {_expected_env_text()}"
        ) from exc

    if args.load_mode == "pretrained":
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )

    try:
        from accelerate import init_empty_weights
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'accelerate' required for --load-mode empty.\n"
            f"Current python: {sys.executable}\n"
            f"Expected env python: {_expected_env_text()}"
        ) from exc

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)


def _resolve_definitions_dir(definitions_root: str) -> Path:
    root = Path(definitions_root).expanduser().resolve()
    if root.is_dir() and root.name == "definitions":
        return root

    definitions_dir = root / "definitions"
    if definitions_dir.is_dir():
        return definitions_dir

    raise FileNotFoundError(
        f"Could not find a definitions directory under '{root}'. "
        "Pass either a trace-set root or the definitions directory itself."
    )


def _load_definition_names(definitions_dir: Path) -> Set[str]:
    names: Set[str] = set()
    for json_path in sorted(definitions_dir.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        name = payload.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _module_type_name(module: Any) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__name__}"


def _linear_shape(module: Any) -> Optional[tuple[int, int]]:
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        return None

    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.out_features), int(module.in_features)

    cls_name = type(module).__name__
    if cls_name == "Conv1D":
        return int(weight.shape[1]), int(weight.shape[0])

    if "Linear" in cls_name:
        return int(weight.shape[0]), int(weight.shape[1])

    return None


def _rmsnorm_hidden_size(module: Any) -> Optional[int]:
    cls_name = type(module).__name__.lower()
    if "rmsnorm" not in cls_name and "rms_norm" not in cls_name:
        return None
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 1:
        return None
    return int(weight.shape[0])


def _summarize_definition(
    def_name: str,
    op_type: str,
    occurrences: int,
    module_names: List[str],
    known_definitions: Set[str],
) -> Dict[str, Any]:
    return {
        "definition": def_name,
        "op_type": op_type,
        "occurrences": occurrences,
        "module_names": module_names,
        "definition_present": def_name in known_definitions,
    }


def _coverage_stats(definitions: List[Dict[str, Any]]) -> Dict[str, int]:
    total_occurrences = sum(item["occurrences"] for item in definitions)
    present = [item for item in definitions if item["definition_present"]]
    present_occurrences = sum(item["occurrences"] for item in present)
    return {
        "unique_total": len(definitions),
        "unique_present": len(present),
        "unique_missing": len(definitions) - len(present),
        "occurrences_total": total_occurrences,
        "occurrences_present": present_occurrences,
        "occurrences_missing": total_occurrences - present_occurrences,
    }


def inspect_model(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model_skeleton(args)
    definitions_dir = _resolve_definitions_dir(args.definitions_root)
    known_definitions = _load_definition_names(definitions_dir)

    module_type_counts: Counter[str] = Counter()
    all_modules: List[Dict[str, str]] = []
    gemm_counts: Counter[str] = Counter()
    gemm_modules: Dict[str, List[str]] = defaultdict(list)
    rmsnorm_counts: Counter[str] = Counter()
    rmsnorm_modules: Dict[str, List[str]] = defaultdict(list)

    for name, module in model.named_modules():
        if not name:
            continue
        module_type = _module_type_name(module)
        module_type_counts[module_type] += 1
        if args.include_modules:
            all_modules.append({"name": name, "type": module_type})

        linear_shape = _linear_shape(module)
        if linear_shape is not None:
            out_features, in_features = linear_shape
            def_name = f"gemm_n{out_features}_k{in_features}"
            gemm_counts[def_name] += 1
            if len(gemm_modules[def_name]) < args.sample_module_names:
                gemm_modules[def_name].append(name)

        hidden_size = _rmsnorm_hidden_size(module)
        if hidden_size is not None:
            def_name = f"rmsnorm_h{hidden_size}"
            rmsnorm_counts[def_name] += 1
            if len(rmsnorm_modules[def_name]) < args.sample_module_names:
                rmsnorm_modules[def_name].append(name)

    gemm_summaries = [
        _summarize_definition(
            def_name,
            "gemm",
            gemm_counts[def_name],
            gemm_modules[def_name],
            known_definitions,
        )
        for def_name in sorted(gemm_modules)
    ]
    rmsnorm_summaries = [
        _summarize_definition(
            def_name,
            "rmsnorm",
            rmsnorm_counts[def_name],
            rmsnorm_modules[def_name],
            known_definitions,
        )
        for def_name in sorted(rmsnorm_modules)
    ]

    replaceable = {
        "gemm": [item for item in gemm_summaries if item["definition_present"]],
        "rmsnorm": [item for item in rmsnorm_summaries if item["definition_present"]],
    }
    missing = {
        "gemm": [item for item in gemm_summaries if not item["definition_present"]],
        "rmsnorm": [item for item in rmsnorm_summaries if not item["definition_present"]],
    }

    coverage = {
        "gemm": _coverage_stats(gemm_summaries),
        "rmsnorm": _coverage_stats(rmsnorm_summaries),
    }
    coverage["overall"] = {
        "unique_total": coverage["gemm"]["unique_total"] + coverage["rmsnorm"]["unique_total"],
        "unique_present": coverage["gemm"]["unique_present"] + coverage["rmsnorm"]["unique_present"],
        "unique_missing": coverage["gemm"]["unique_missing"] + coverage["rmsnorm"]["unique_missing"],
        "occurrences_total": coverage["gemm"]["occurrences_total"]
        + coverage["rmsnorm"]["occurrences_total"],
        "occurrences_present": coverage["gemm"]["occurrences_present"]
        + coverage["rmsnorm"]["occurrences_present"],
        "occurrences_missing": coverage["gemm"]["occurrences_missing"]
        + coverage["rmsnorm"]["occurrences_missing"],
    }

    summary: Dict[str, Any] = {
        "model": args.model,
        "load_mode": args.load_mode,
        "definitions_root": str(Path(args.definitions_root).expanduser().resolve()),
        "definitions_dir": str(definitions_dir),
        "definition_name_count": len(known_definitions),
        "module_type_counts": dict(module_type_counts.most_common()),
        "coverage": coverage,
        "replaceable": replaceable,
        "missing": missing,
    }
    if args.include_modules:
        summary["modules"] = all_modules
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = inspect_model(args)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[output] wrote inspection summary to {output_path}")


if __name__ == "__main__":
    main()
