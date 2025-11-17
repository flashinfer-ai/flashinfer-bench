"""TVM-FFI based builder for CUDA kernels with automatic caching."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tvm_ffi
from tvm_ffi.utils import FileLock

from flashinfer_bench.compile.builder import Builder, BuildError, create_pkg_name
from flashinfer_bench.compile.runnable import Runnable, TVMFFIRunnable
from flashinfer_bench.data import Definition, Solution, SupportedLanguages
from flashinfer_bench.env import get_fib_cache_path

logger = logging.getLogger(__name__)

# File extension mappings for source file classification
CUDA_EXTENSIONS = [".cu"]  # CUDA source files
CPP_EXTENSIONS = [".cpp", ".cc", ".cxx", ".c"]  # C/C++ source files


class Language(Enum):
    """Enum representing source code languages supported by the builder."""

    CUDA = "cuda"
    """The solution's language is CUDA"""
    CPP = "cpp"
    """The solution's language is C/C++"""


class TVMFFIBuilder(Builder):
    """Builder using TVM-FFI with automatic caching and supports multi-process and multi-threaded
    compilation. The result is framework agnostic and supports DLPack interop with PyTorch, JAX,
    etc.

    Cache logic: If the builder is asked to build the same solution again, it will return the cached
    result. If another builder is asking to build the same solution, as long as the build directory
    exists, it will return the cached result.

    The solution to compile should be written in destination-passing style, i.e. the function
    should take the input tensors and the output tensors as arguments.

    Examples
    --------
    >>> builder = TVMFFIBuilder()
    >>> runnable = builder.build(definition, solution)
    >>> output = runnable(x=input_tensor)  # Allocates and returns output
    >>> runnable.call_dest(x=input_tensor, output=output_tensor)  # Destination-passing style
    """

    _BUILD_DIR_NAME = "tvm_ffi"
    """Subdirectory under FIB_CACHE_PATH where build artifacts are stored"""

    _LOCK_FILE_NAME = "flashinfer_bench_tvm_ffi_lock"
    """File lock name for multi-process synchronization during compilation"""

    _KEY_PREFIX = "tvm_ffi_"
    """Prefix for cache keys to avoid collisions with other builders"""

    def __init__(self) -> None:
        """Initialize the TVMFFIBuilder.

        Sets up empty dictionaries for future extensibility with extra
        include paths and linker flags (currently unused).
        """
        super().__init__()
        self._extra_include_paths: Dict[str, str] = {}
        self._extra_ldflags: Dict[str, List[str]] = {}

    def can_build(self, sol: Solution) -> bool:
        """Check if this builder can build the given solution.

        Parameters
        ----------
        sol : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is CUDA (includes both .cu and .cpp files)
        """
        return sol.spec.language == SupportedLanguages.CUDA

    def _make_key(self, solution: Solution) -> str:
        """Generate unique cache key for a solution.

        Parameters
        ----------
        solution : Solution
            Solution to generate key for

        Returns
        -------
        str
            Unique key combining builder name and solution package name
        """
        return self._KEY_PREFIX + create_pkg_name(solution)

    def _make_closer(self):
        """Create a closer function for resource cleanup.

        Returns
        -------
        callable
            No-op closer since TVM-FFI handles cleanup automatically
        """
        return lambda: None

    def _get_build_path(self, key: str) -> Path:
        """Get the build directory path for a given cache key.

        Parameters
        ----------
        key : str
            Unique cache key for the solution

        Returns
        -------
        Path
            Directory path where build artifacts will be stored
        """
        return get_fib_cache_path() / self._BUILD_DIR_NAME / key

    def _check_sources(self, path: Path, key: str, sol: Solution) -> bool:
        """Check if the source code is vaild, and if the cached .so can be used by comparing source
        files and .so existence.

        Returns True (can use cached .so) only if:
        1. The compiled .so file exists
        2. All source files exist with identical content

        Parameters
        ----------
        path : Path
            Build directory path
        key : str
            Unique key for this solution (used to find .so file)
        sol : Solution
            Solution containing source files

        Returns
        -------
        can_use_cached : bool
            True if the cached .so can be used, False if compilation is needed
        """
        # Check if build directory exists
        if not path.exists():
            return False
        elif not path.is_dir():
            raise BuildError(f"Build directory exists but is not a directory: {path}")

        # Check if .so exists
        so_path = path / f"{key}.so"
        if not so_path.is_file():
            return False

        # Check if all files exist and content is identical
        for src in sol.sources:
            # Defensive assertion: the path in the solution should be validated by the Solution
            # model validator, but we add this defensive assertion to be safe.
            src_path_obj = Path(src.path)
            assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
            assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

            src_path = path / src.path

            if not src_path.exists():
                return False
            elif not src_path.is_file():
                raise BuildError(f"Source path exists but is not a file: {src_path}")

            if src_path.read_text() != src.content:
                return False

        # All checks passed: can use cached .so
        return True

    def _detect_language(self, sol: Solution) -> Language:
        """Detect source language based on file extensions.

        Parameters
        ----------
        sol : Solution
            Solution containing source files

        Returns
        -------
        Language
            CUDA if any .cu files present, otherwise CPP

        Raises
        ------
        BuildError
            If no valid source files found
        """
        has_cuda = False
        has_cpp = False

        for src in sol.sources:
            path_str = str(src.path)
            if path_str.endswith(tuple(CUDA_EXTENSIONS)):
                has_cuda = True
            elif path_str.endswith(tuple(CPP_EXTENSIONS)):
                has_cpp = True

        if not has_cuda and not has_cpp:
            raise BuildError("No CUDA or C++ sources found")

        return Language.CUDA if has_cuda else Language.CPP

    def _write_sources(self, path: Path, sol: Solution) -> Tuple[List[str], List[str]]:
        """Write all source files to build directory and collect file paths.

        Creates parent directories as needed for files in subdirectories.
        Overwrites files unconditionally (caller already determined a full build is needed).

        Parameters
        ----------
        path : Path
            Build directory where source files will be written
        sol : Solution
            Solution containing source files to write

        Returns
        -------
        cpp_files : List[str]
            List of C++ source file paths
        cuda_files : List[str]
            List of CUDA source file paths
        """
        path.mkdir(parents=True, exist_ok=True)
        cpp_files: List[str] = []
        cuda_files: List[str] = []

        for src in sol.sources:
            # Defensive assertion: path should be validated at Solution creation time
            src_path_obj = Path(src.path)
            assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
            assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

            src_path = path / src.path

            # Ensure parent directories exist
            src_path.parent.mkdir(parents=True, exist_ok=True)

            # Write source file
            src_path.write_text(src.content)

            # Collect file path by extension
            path_str = str(src_path)
            if path_str.endswith(tuple(CPP_EXTENSIONS)):
                cpp_files.append(path_str)
            elif path_str.endswith(tuple(CUDA_EXTENSIONS)):
                cuda_files.append(path_str)

        return cpp_files, cuda_files

    def _get_entry_symbol(self, sol: Solution) -> str:
        """Extract function symbol from entry_point.

        Parameters
        ----------
        sol : Solution
            Solution with entry_point in format 'file.ext::symbol'

        Returns
        -------
        str
            The function symbol name to be loaded from the compiled module

        Raises
        ------
        BuildError
            If entry_point format is invalid (missing '::' separator)
        """
        entry_point = sol.spec.entry_point
        if "::" not in entry_point:
            raise BuildError(
                f"Invalid entry_point format: {entry_point}. Expected 'file.extension::symbol'"
            )
        return entry_point.split("::")[-1]

    def _make_runnable(
        self, mod: tvm_ffi.Module, entry_symbol: str, defn: Definition, metadata: Dict[str, Any]
    ) -> TVMFFIRunnable:
        """Create Runnable from TVM-FFI module.

        Wraps the compiled function with a keyword argument adapter that matches
        the definition's input/output interface.

        Parameters
        ----------
        mod : tvm_ffi.Module
            Loaded TVM-FFI module containing the compiled function
        entry_symbol : str
            Name of the function to extract from the module
        defn : Definition
            Definition specifying the function interface
        metadata : Dict[str, Any]
            Metadata about the build (language, paths, etc.)

        Returns
        -------
        TVMFFIRunnable
            Runnable wrapper that handles tensor allocation and keyword arguments

        Raises
        ------
        BuildError
            If the entry_symbol is not found in the module
        """
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
        """Build with automatic caching - compile once, load from cache afterwards.

        This method implements intelligent caching:
        1. Checks if a compiled .so file already exists
        2. If not, writes source files and compiles them
        3. Loads the compiled module (from cache or fresh build)
        4. Returns a runnable wrapper

        The caching is multi-process safe, enabling efficient parallel benchmarking.

        Parameters
        ----------
        defn : Definition
            Problem definition specifying inputs/outputs
        sol : Solution
            Solution containing source code and build specification

        Returns
        -------
        Runnable
            TVMFFIRunnable that can be called with input tensors

        Raises
        ------
        BuildError
            If compilation fails, module loading fails, or entry point is invalid
        """
        key = self._make_key(sol)
        build_path = self._get_build_path(key)
        entry_symbol = self._get_entry_symbol(sol)
        language = self._detect_language(sol)
        can_use_cached = self._check_sources(build_path, key, sol)

        # Check if cached .so can be used
        # This checking and rebuilding is thread-safe through the FileLock
        if can_use_cached:
            output_lib_path = str(build_path / f"{key}.so")
        else:
            # Ensure build directory exists before creating file lock
            build_path.mkdir(parents=True, exist_ok=True)
            with FileLock(build_path / self._LOCK_FILE_NAME):
                # Double-check after acquiring lock (another process may have built it)
                if self._check_sources(build_path, key, sol):
                    output_lib_path = str(build_path / f"{key}.so")
                else:
                    cpp_files, cuda_files = self._write_sources(build_path, sol)
                    extra_include_paths = [str(build_path)]
                    try:
                        # Compile sources to shared library
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

        # Create metadata for the runnable
        metadata = {
            "definition": defn.name,
            "solution": sol.name,
            "language": language.value,
            "binding": "tvm_ffi",
            "key": key,
            "symbol": entry_symbol,
            "binary": output_lib_path,
        }

        return self._make_runnable(mod, entry_symbol, defn, metadata)
