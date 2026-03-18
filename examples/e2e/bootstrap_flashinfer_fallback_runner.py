"""Launch a target after substituting SGLang fused_add_rmsnorm with FlashInfer baseline."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start a target Python module or script after replacing "
            "sgl_kernel.fused_add_rmsnorm with flashinfer.norm.fused_add_rmsnorm."
        )
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write a small JSON summary after the target exits",
    )
    parser.add_argument(
        "--flashinfer-workspace-base",
        help=(
            "Optional override for FLASHINFER_WORKSPACE_BASE. Use this to isolate FlashInfer "
            "JIT cache from other Python environments."
        ),
    )
    parser.add_argument("--chdir", help="Optional working directory before launching the target")
    parser.add_argument(
        "--prepend-pythonpath",
        action="append",
        default=[],
        help="Additional path prepended to sys.path before launching the target",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module", help="Python module executed like `python -m module`")
    target.add_argument("--script", help="Python script path executed like `python script.py`")

    parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target. Prefix with `--`.",
    )
    return parser


def _normalize_target_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def _prepare_environment(args: argparse.Namespace) -> None:
    if args.flashinfer_workspace_base:
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(Path(args.flashinfer_workspace_base).resolve())
    if args.chdir:
        os.chdir(Path(args.chdir).resolve())
    for path in reversed(args.prepend_pythonpath):
        sys.path.insert(0, str(Path(path).resolve()))


def _install_fallback_substitution() -> Callable[..., Any]:
    import flashinfer
    import sgl_kernel

    original = sgl_kernel.fused_add_rmsnorm
    sgl_kernel.fused_add_rmsnorm = flashinfer.norm.fused_add_rmsnorm
    return original


def _restore_fallback_substitution(original: Callable[..., Any]) -> None:
    import sgl_kernel

    sgl_kernel.fused_add_rmsnorm = original


def _run_target(args: argparse.Namespace, target_args: list[str]) -> None:
    if args.module:
        sys.argv = [args.module, *target_args]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        return

    script_path = str(Path(args.script).resolve())
    sys.argv = [script_path, *target_args]
    runpy.run_path(script_path, run_name="__main__")


def main() -> None:
    args = build_parser().parse_args()
    target_args = _normalize_target_args(args.target_args)
    _prepare_environment(args)

    original = _install_fallback_substitution()
    target_desc = args.module if args.module else args.script
    print("[fallback] installed FlashInfer fused_add_rmsnorm substitution")
    print(f"[fallback] launching target: {target_desc}")

    try:
        _run_target(args, target_args)
    finally:
        _restore_fallback_substitution(original)
        if args.summary_json:
            payload = {
                "requested": True,
                "active": True,
                "status": "enabled",
                "mode": "fallback_substitution",
                "patched_entrypoint": "sgl_kernel.fused_add_rmsnorm",
                "replacement_entrypoint": "flashinfer.norm.fused_add_rmsnorm",
            }
            output_path = Path(args.summary_json).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[fallback] wrote summary to {output_path}")


if __name__ == "__main__":
    main()
