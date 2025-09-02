from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List

from flashinfer_bench.compile.builder import (
    Builder,
    BuildError,
    create_pkg_name,
    write_sources_to_dir,
)
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.solution import Solution, SupportedLanguages

CUDA_ALLOWED_EXTS = [".cu", ".cpp", ".cc", ".cxx", ".c"]


class CUDABuilder(Builder):
    def __init__(self) -> None:
        super().__init__()
        self._build_dirs: Dict[str, str] = {}

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.CUDA

    def _make_key(self, solution: Solution) -> str:
        return f"cuda::{create_pkg_name(solution)}"

    def _make_closer(self):
        # We keep build dirs for torch extension caching. The temp dirs can be cleaned by calling `clear_cache` on program exit.
        return lambda: None

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        # CUDA solutions must provide a C/CUDA symbol as entry point.
        # If user prefer a Python wrapper, set language to `python` and ensure compilation and binding are properly handled.
        entry_file_extension = "." + sol.spec.entry_point.split("::")[0].split(".")[-1]
        if entry_file_extension not in CUDA_ALLOWED_EXTS:
            raise BuildError(
                f"Entry file type not recognized. Must be one of {CUDA_ALLOWED_EXTS}, got {entry_file_extension}."
            )

        try:
            import torch
            from torch.utils.cpp_extension import load
        except Exception as e:
            raise BuildError("PyTorch is not available in the current environment") from e

        symbol = sol.spec.entry_point.split("::")[-1]
        name = create_pkg_name(sol, "fib_cuda_")
        cache_root = os.environ.get(
            "FLASHINFER_BENCH_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "flashinfer_bench"),
        )
        build_dir = os.path.join(cache_root, "cuda", name)
        write_sources_to_dir(build_dir, sol.sources)
        self._build_dirs[name] = build_dir

        sources = [s for s in sol.sources if s.path.endswith(tuple(CUDA_ALLOWED_EXTS))]

        if not sources:
            raise BuildError("No CUDA/C++ sources provided for CUDA build")

        src_paths = [os.path.join(build_dir, s.path) for s in sources]

        # Hardcode cutlass headers for now
        extra_include_paths = [
            os.path.join(build_dir),
            os.path.abspath("./3rdparty/cutlass/include"),
        ]
        extra_ldflags = ["-lcublas", "-lcudnn"]

        closer = self._make_closer()

        try:
            ext = load(
                name=name,
                sources=src_paths,
                extra_include_paths=extra_include_paths,
                extra_ldflags=extra_ldflags,
                with_cuda=True,
                build_directory=build_dir,
                verbose=True,
            )
        except Exception as e:
            raise BuildError(f"CUDA build failed for solution '{sol.name}': {e}") from e

        try:
            fn = getattr(ext, symbol)
        except AttributeError as e:
            raise BuildError(f"Exported symbol '{symbol}' not found in built extension") from e

        arg_order = list(defn.inputs.keys())

        def _kw_adapter(**kwargs):
            args = [kwargs[name] for name in arg_order]
            return fn(*args)

        meta = {
            "definition": defn.name,
            "solution": sol.name,
            "language": "cuda",
            "name": name,
            "entry": sol.spec.entry_point,
            "symbol": symbol,
            "build_dir": build_dir,
            "binary": getattr(ext, "__file__", None),
            "extra_include_paths": extra_include_paths,
            "extra_ldflags": extra_ldflags,
        }
        return Runnable(fn=_kw_adapter, closer=closer, meta=meta)

    def clear_cache(self) -> None:
        super().clear_cache()
        for build_dir in self._build_dirs.values():
            shutil.rmtree(build_dir, ignore_errors=True)
        self._build_dirs.clear()
