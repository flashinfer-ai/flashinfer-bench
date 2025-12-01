from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional

from flashinfer_bench.compile.builder import (
    Builder,
    BuildError,
    create_pkg_name,
    write_sources_to_dir,
)
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data import Definition, Solution, SourceFile, SupportedLanguages
from flashinfer_bench.utils import is_cuda_available

CUDA_ALLOWED_EXTS = [".cu", ".cpp", ".cc", ".cxx", ".c"]

logger = logging.getLogger(__name__)


def _get_package_paths(pkg_name: str, lib_names: Optional[List[str]] = None):
    include_path = None
    ldflags = []

    try:
        include_dir = resources.files(pkg_name) / "include"
        if include_dir.exists():
            include_path = str(include_dir)

        if lib_names:
            lib_dir = resources.files(pkg_name) / "lib"
            if lib_dir.exists():
                lib_path = Path(lib_dir)

                if sys.platform.startswith("linux"):
                    ldflags = [f"-L{lib_path}", f"-Wl,-rpath,{lib_path}"]

                    for lib_name in lib_names:
                        # Look for unversioned .so first
                        lib_file = lib_path / f"lib{lib_name}.so"
                        if lib_file.exists():
                            ldflags.append(f"-l{lib_name}")
                        else:
                            # Find versioned .so files
                            versioned = sorted(lib_path.glob(f"lib{lib_name}.so.*"))
                            if versioned:
                                ldflags.append(f"-l:{versioned[-1].name}")
                            else:
                                ldflags.append(f"-l{lib_name}")  # Fallback

                elif sys.platform == "win32":
                    ldflags = [f"/LIBPATH:{lib_path}"] + lib_names

    except Exception:
        logger.warning(
            "Failed to discover resources for CUDA package '%s'; continuing without it.",
            pkg_name,
            exc_info=True,
        )

    return include_path, ldflags


CUDA_DEPS = {
    "cublas": ("nvidia.cublas", ["cublas", "cublasLt"]),
    "cudnn": ("nvidia.cudnn", ["cudnn"]),
    "cutlass": ("flashinfer_bench.thirdparty.cutlass", None),  # Header-only
}


def _discover_cuda_deps(extra_include_paths: Dict[str, str], extra_ldflags: Dict[str, List[str]]):
    for dep_name, (pkg_name, libs) in CUDA_DEPS.items():
        include_path, ldflags = _get_package_paths(pkg_name, libs)
        if include_path:
            extra_include_paths[dep_name] = include_path
        if ldflags:
            extra_ldflags[dep_name] = ldflags


CUDA_DEPS_INCLUDE_PATTERNS = {
    "cublas": re.compile(
        r'^\s*#\s*include\s*[<"]\s*(?:cublas|cublasLt)', re.MULTILINE | re.IGNORECASE
    ),
    "cudnn": re.compile(r'^\s*#\s*include\s*[<"]\s*cudnn', re.MULTILINE | re.IGNORECASE),
    "cutlass": re.compile(r'^\s*#\s*include\s*[<"]\s*cutlass/', re.MULTILINE),
}


def _check_dependency(sources: List[SourceFile], dep_name: str) -> bool:
    pattern = CUDA_DEPS_INCLUDE_PATTERNS.get(dep_name)
    if not pattern:
        return False

    for source in sources:
        if not isinstance(source.content, str):
            continue

        # Fast skip
        if dep_name not in source.content.lower():
            continue

        # Remove comments
        content = source.content
        content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        if pattern.search(content):
            return True

    return False


class CUDABuilder(Builder):
    _cuda_available: bool = None

    @classmethod
    def _get_cuda_available(cls) -> bool:
        if cls._cuda_available is None:
            cls._cuda_available = is_cuda_available()
        return cls._cuda_available

    def __init__(self) -> None:
        super().__init__()
        self._build_dirs: Dict[str, str] = {}
        self._extra_include_paths: Dict[str, str] = {}
        self._extra_ldflags: Dict[str, List[str]] = {}
        _discover_cuda_deps(self._extra_include_paths, self._extra_ldflags)

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.CUDA and self._get_cuda_available()

    def _make_key(self, solution: Solution) -> str:
        return f"cuda::{create_pkg_name(solution)}"

    def _make_closer(self):
        # We keep build dirs for torch extension caching. The temp dirs can be cleaned by
        # calling `clear_cache` on program exit.
        return lambda: None

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        # CUDA solutions must provide a C/CUDA symbol as entry point.
        # If user prefer a Python wrapper, set language to `python` and ensure compilation and
        # binding are properly handled.
        entry_file_extension = "." + sol.spec.entry_point.split("::")[0].split(".")[-1]
        if entry_file_extension not in CUDA_ALLOWED_EXTS:
            raise BuildError(
                f"Entry file type not recognized. Must be one of {CUDA_ALLOWED_EXTS}, "
                f"got {entry_file_extension}."
            )

        if not self._get_cuda_available():
            raise BuildError("torch.cuda is not available in the current environment")

        from torch.utils.cpp_extension import load

        symbol = sol.spec.entry_point.split("::")[-1]
        name = create_pkg_name(sol, "fib_cuda_")
        cache_root = os.environ.get(
            "FIB_CACHE_PATH", os.path.join(os.path.expanduser("~"), ".cache", "flashinfer_bench")
        )
        build_dir = os.path.join(cache_root, "cuda", name)
        write_sources_to_dir(build_dir, sol.sources)
        self._build_dirs[name] = build_dir

        sources = [s for s in sol.sources if s.path.endswith(tuple(CUDA_ALLOWED_EXTS))]

        has_cuda_sources = any(s.path.endswith(".cu") for s in sources)
        if not has_cuda_sources:
            raise BuildError("No CUDA sources provided for CUDA build")

        src_paths = [os.path.join(build_dir, s.path) for s in sources]

        extra_include_paths = [build_dir]
        extra_ldflags = []

        for dep in CUDA_DEPS.keys():
            if _check_dependency(sources, dep):
                inc_path = self._extra_include_paths.get(dep)
                if not inc_path:
                    raise BuildError(
                        f"{dep} is not available in the current environment but referenced "
                        f"by {sol.name}"
                    )
                extra_include_paths.append(inc_path)
                ldflags = self._extra_ldflags.get(dep)
                if ldflags:
                    extra_ldflags.extend(ldflags)

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
