"""Inspect the local E2E benchmark environment for Transformers, vLLM, and SGLang."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the current Python environment is ready for the "
            "Transformers / vLLM / SGLang E2E benchmark launchers."
        )
    )
    parser.add_argument("--output-json", help="Optional path to write the environment report JSON")
    return parser


def _probe_import(module_name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - surfacing exact import failure is the point
        return {
            "ok": False,
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "ok": True,
        "module": module_name,
        "version": getattr(module, "__version__", ""),
        "file": getattr(module, "__file__", ""),
    }


def _probe_torch() -> Dict[str, Any]:
    torch_info = _probe_import("torch")
    if not torch_info["ok"]:
        return torch_info

    import torch

    info: Dict[str, Any] = dict(torch_info)
    info.update(
        {
            "cuda_is_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", ""),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": f"{capability[0]}.{capability[1]}",
                }
            )
        info["devices"] = devices
    return info


def _probe_nvidia_smi() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found in PATH"}

    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _recommendations(report: Dict[str, Any]) -> list[str]:
    recommendations: list[str] = []

    torch_info = report.get("torch", {})
    if not torch_info.get("ok"):
        recommendations.append("Install a working PyTorch build before benchmarking.")
        return recommendations

    if not torch_info.get("cuda_is_available"):
        recommendations.append("PyTorch cannot see CUDA in this environment; check CUDA toolkit and drivers.")

    vllm_info = report.get("vllm_bench_latency", report.get("vllm", {}))
    if not vllm_info.get("ok"):
        error_text = str(vllm_info.get("error", "")).lower()
        if "undefined symbol" in error_text:
            recommendations.append(
                "vLLM import failed with an undefined symbol; reinstall vLLM against the current torch/CUDA stack."
            )
        else:
            recommendations.append("vLLM is not importable; reinstall or rebuild it in this environment.")

    sglang_info = report.get("sglang_bench_one_batch", report.get("sglang", {}))
    if not sglang_info.get("ok"):
        recommendations.append("SGLang is not importable; reinstall it in this environment.")

    if not recommendations:
        recommendations.append("Core framework imports look healthy for local E2E benchmarking.")
    return recommendations


def collect_report() -> Dict[str, Any]:
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "nvidia_smi": _probe_nvidia_smi(),
        "torch": _probe_torch(),
        "transformers": _probe_import("transformers"),
        "flashinfer_bench": _probe_import("flashinfer_bench"),
        "vllm": _probe_import("vllm"),
        "vllm_bench_latency": _probe_import("vllm.benchmarks.latency"),
        "vllm_bench_serve": _probe_import("vllm.benchmarks.serve"),
        "sglang": _probe_import("sglang"),
        "sglang_bench_one_batch": _probe_import("sglang.bench_one_batch"),
        "sglang_bench_serving": _probe_import("sglang.bench_serving"),
    }
    report["recommendations"] = _recommendations(report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = collect_report()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[output] wrote environment report to {output_path}")


if __name__ == "__main__":
    main()
