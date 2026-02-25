"""TVM-FFI based builder for CUDA kernels with automatic caching. This is the primary builder for
CUDA and C++ kernels."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, ClassVar, List, Tuple

from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata
from flashinfer_bench.compile.utils import write_sources_to_path
from flashinfer_bench.data import Definition, Solution, SupportedBindings, SupportedLanguages

logger = logging.getLogger(__name__)

# File extension mappings for source file classification
_CUDA_EXTENSIONS: List[str] = [".cu"]  # CUDA source files
_CPP_EXTENSIONS: List[str] = [".cpp", ".cc", ".cxx", ".c"]  # C/C++ source files


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

    _PACKAGE_PREFIX: ClassVar[str] = "tvm_ffi_"
    """Prefix for cache keys to avoid collisions with other builders"""

    _BUILD_DIR_NAME: ClassVar[str] = "tvm_ffi"
    """Subdirectory under FIB_CACHE_PATH where build artifacts are stored"""

    _LOCK_FILE_NAME: ClassVar[str] = "flashinfer_bench_tvm_ffi_lock"
    """File lock name for multi-process synchronization during compilation"""

    def __init__(self) -> None:
        """Initialize the TVMFFIBuilder."""
        super().__init__(self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check if TVM-FFI is available in the current environment."""
        try:
            import tvm_ffi  # noqa: F401
        except ImportError:
            return False
        return True

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can build the given solution. The solution should be CUDA or
        C++ source code with TVM-FFI binding (or no binding specified, which defaults to TVM-FFI).

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is CUDA or C++ and binding is TVM-FFI or None
        """
        is_cpp_or_cuda = (
            solution.spec.language == SupportedLanguages.CUDA
            or solution.spec.language == SupportedLanguages.CPP
        )
        is_tvm_ffi_binding = (
            solution.spec.binding is None or solution.spec.binding == SupportedBindings.TVM_FFI
        )
        return is_cpp_or_cuda and is_tvm_ffi_binding

    def _check_sources(self, path: Path, key: str, solution: Solution) -> bool:
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
        solution : Solution
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
        for src in solution.sources:
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

    def _filter_sources(self, source_paths: List[Path]) -> Tuple[List[str], List[str]]:
        """Filter source files by extension into C++ and CUDA source file paths.

        Parameters
        ----------
        source_paths : List[Path]
            List of source file paths.

        Returns
        -------
        cpp_files : List[str]
            List of C++ source file paths
        cuda_files : List[str]
            List of CUDA source file paths
        """
        cpp_files: List[str] = []
        cuda_files: List[str] = []
        for src_path in source_paths:
            if src_path.suffix in _CPP_EXTENSIONS:
                cpp_files.append(str(src_path))
            elif src_path.suffix in _CUDA_EXTENSIONS:
                cuda_files.append(str(src_path))

        return cpp_files, cuda_files

    def _get_entry_symbol(self, solution: Solution) -> str:
        """Extract function symbol from entry_point.

        Parameters
        ----------
        solution : Solution
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
        entry_point = solution.spec.entry_point
        if "::" not in entry_point:
            raise BuildError(
                f"Invalid entry_point format: {entry_point}. Expected 'file.extension::symbol'"
            )
        return entry_point.split("::")[-1]

    def _get_cleaner(self, build_path: Path) -> Callable[[], None]:
        """Get a cleaner function for the build directory. It will remove the build directory.

        Parameters
        ----------
        build_path : Path
            The path to the build directory

        Returns
        -------
        callable
            A function that cleans up the build directory.
        """

        def cleaner() -> None:
            shutil.rmtree(build_path, ignore_errors=True)

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build with automatic caching - compile once, load from cache afterwards.

        This method implements intelligent caching:
        1. Checks if a compiled .so file already exists
        2. If not, writes source files and compiles them
        3. Loads the compiled module (from cache or fresh build)
        4. Returns a runnable wrapper

        The caching is multi-process safe, enabling efficient parallel benchmarking.

        Parameters
        ----------
        definition : Definition
            Problem definition specifying inputs/outputs
        solution : Solution
            Solution containing source code and build specification

        Returns
        -------
        Runnable
            A runnable wrapper around the compiled TVM-FFI module that supports both
            value-returning style (via __call__) and destination-passing style (via call_dps)

        Raises
        ------
        BuildError
            If compilation fails, module loading fails, or entry point is invalid
        """
        import tvm_ffi
        from tvm_ffi.utils import FileLock

        package_name, build_path = self._get_package_name_and_build_path(solution)
        entry_symbol = self._get_entry_symbol(solution)
        can_use_cached = self._check_sources(build_path, package_name, solution)

        # Check if cached .so can be used. If not, build the solution.
        # This check and build are thread-safe through the FileLock
        if can_use_cached:
            output_lib_path = str(build_path / f"{package_name}.so")
        else:
            # Ensure build directory exists before creating file lock
            build_path.mkdir(parents=True, exist_ok=True)
            with FileLock(build_path / self._LOCK_FILE_NAME):
                # Double-check after acquiring lock (another process may have built it)
                if self._check_sources(build_path, package_name, solution):
                    output_lib_path = str(build_path / f"{package_name}.so")
                else:
                    src_paths = write_sources_to_path(build_path, solution.sources)
                    cpp_files, cuda_files = self._filter_sources(src_paths)
                    extra_include_paths = [str(build_path)]
                    try:
                        # Compile sources to shared library
                        output_lib_path = tvm_ffi.cpp.build(
                            name=package_name,
                            cpp_files=cpp_files,
                            cuda_files=cuda_files,
                            extra_include_paths=extra_include_paths,
                            build_directory=build_path,
                        )
                    except Exception as e:
                        raise BuildError(
                            f"TVM-FFI compilation failed for '{solution.name}': {e}"
                        ) from e

        # Load the compiled module
        try:
            mod = tvm_ffi.load_module(output_lib_path)
        except Exception as e:
            raise BuildError(f"Failed to load compiled module: {e}") from e

        # Create metadata for the runnable
        metadata = RunnableMetadata(
            build_type="tvm_ffi",
            definition_name=definition.name,
            solution_name=solution.name,
            destination_passing_style=solution.spec.destination_passing_style,
            definition=definition,
            misc={"entry_symbol": entry_symbol, "binary": output_lib_path},
        )

        try:
            callable = getattr(mod, entry_symbol)
        except AttributeError as e:
            raise BuildError(f"Entry point '{entry_symbol}' not found in module") from e

        self._try_validate_signature(callable, definition, solution)

        cleaner = self._get_cleaner(build_path)
        return Runnable(callable=callable, metadata=metadata, cleaner=cleaner)
