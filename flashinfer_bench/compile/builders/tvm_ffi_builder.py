"""TVM-FFI based builder for CUDA kernels with automatic caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tvm_ffi

from flashinfer_bench.compile.builder import Builder, BuildError, create_pkg_name
from flashinfer_bench.compile.runnable import Runnable, TVMFFIRunnable
from flashinfer_bench.data import Definition, Solution, SupportedLanguages
from flashinfer_bench.env import get_fib_cache_path

logger = logging.getLogger(__name__)

CUDA_EXTENSIONS = [".cu"]
CPP_EXTENSIONS = [".cpp", ".cc", ".cxx", ".c"]


class TVMFFIBuilder(Builder):
    """Builder using TVM-FFI with automatic caching and multi-process sharing.

    Build strategy:
    1. Check if .so exists in cache (multi-process safe)
    2. If not, compile with tvm_ffi.cpp.build_inline() to cache
    3. Load with tvm_ffi.load_module()

    Benefits:
    - Multi-process benchmark: Only first process compiles, others load from cache
    - Cross-framework: Same .so works with PyTorch, JAX, CuPy (DLPack)
    - No JIT/AOT distinction: Smart caching handles both cases
    """

    def __init__(self) -> None:
        super().__init__()
        self._extra_include_paths: Dict[str, str] = {}
        self._extra_ldflags: Dict[str, List[str]] = {}

    def can_build(self, sol: Solution) -> bool:
        return sol.spec.language == SupportedLanguages.CUDA

    def _make_key(self, solution: Solution) -> str:
        return f"tvm_ffi_{create_pkg_name(solution)}"

    def _make_closer(self):
        return lambda: None

    def _get_build_path(self, key: str) -> Path:
        return get_fib_cache_path() / "tvm_ffi" / key

    def _write_sources(self, path: Path, sol: Solution) -> Tuple[List[str], List[str]]:
        """Extract and write all source files to the given path."""
        path.mkdir(parents=True, exist_ok=True)
        cpp_files: List[str] = []
        cuda_files: List[str] = []
        for src in sol.sources:
            src_path = path / src.path
            if src_path.is_dir():
                raise BuildError(f"Source path is a directory: {src_path}")

            src_path.write_text(src.content)

            if str(src_path).endswith(tuple(CPP_EXTENSIONS)):
                cpp_files.append(str(src_path))
            elif str(src_path).endswith(tuple(CUDA_EXTENSIONS)):
                cuda_files.append(str(src_path))

        if len(cpp_files) == 0 and len(cuda_files) == 0:
            raise BuildError("No sources found")
        return cpp_files, cuda_files

    def _get_language(self, cpp_files: List[str], cuda_files: List[str]) -> str:
        return "cuda" if len(cuda_files) > 0 else "cpp"

    def _get_entry_symbol(self, sol: Solution) -> str:
        """Extract function symbol from entry_point."""
        entry_point = sol.spec.entry_point
        if "::" not in entry_point:
            raise BuildError(
                f"Invalid entry_point format: {entry_point}. Expected 'file.cu::symbol'"
            )
        return entry_point.split("::")[-1]

    def _make_runnable(
        self, mod: tvm_ffi.Module, entry_symbol: str, defn: Definition, metadata: Dict[str, Any]
    ) -> Runnable:
        """Create Runnable from TVM-FFI module."""
        try:
            fn = getattr(mod, entry_symbol)
        except AttributeError as e:
            raise BuildError(f"Entry point '{entry_symbol}' not found in module") from e

        # Create keyword adapter to match definition interface
        arg_order = list(defn.inputs.keys()) + list(defn.outputs.keys())

        def _kw_adapter(**kwargs):
            args = [kwargs[name] for name in arg_order]
            return fn(*args)

        return TVMFFIRunnable(
            fn=_kw_adapter, closer=self._make_closer(), meta=metadata, definition=defn
        )

    def _build(self, defn: Definition, sol: Solution) -> Runnable:
        """Build with automatic caching - compile once, load from cache afterwards."""
        key = self._make_key(sol)
        build_path = self._get_build_path(key)
        entry_symbol = self._get_entry_symbol(sol)
        cpp_files, cuda_files = self._write_sources(build_path, sol)
        language = self._get_language(cpp_files, cuda_files)
        extra_include_paths = [str(build_path)]

        try:
            # Use build_inline instead of build to
            output_lib_path = tvm_ffi.cpp.build(
                name=key,
                cpp_files=cpp_files,
                cuda_files=cuda_files,
                extra_include_paths=extra_include_paths,
                build_directory=build_path,
            )
        except Exception as e:
            raise BuildError(f"TVM-FFI compilation failed for '{sol.name}': {e}") from e

        # Load the compiled module
        try:
            mod = tvm_ffi.load_module(output_lib_path)
        except Exception as e:
            raise BuildError(f"Failed to load compiled module: {e}") from e

        metadata = {
            "definition": defn.name,
            "solution": sol.name,
            "language": language,
            "binding": "tvm_ffi",
            "key": key,
            "symbol": entry_symbol,
            "binary": output_lib_path,
        }

        return self._make_runnable(mod, entry_symbol, defn, metadata)
