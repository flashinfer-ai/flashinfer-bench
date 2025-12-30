"""Builder for pure Python solutions."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, ClassVar

from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata
from flashinfer_bench.compile.utils import write_sources_to_path
from flashinfer_bench.data import Definition, Solution, SupportedLanguages


class PythonBuilder(Builder):
    """Builder for Python solutions.

    This builder loads Python source files into a temporary module and returns a callable
    that can be executed. The sources are written to a cache directory and imported as a
    Python package.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "fib_python_"
    """Prefix for cache keys to avoid collisions with other builders. fib_ prefix is added
    to avoid name collision in python imports."""

    _BUILD_DIR_NAME: ClassVar[str] = "python"
    """Subdirectory under FIB_CACHE_PATH where build results are stored."""

    def __init__(self) -> None:
        super().__init__(self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check if Python is available in the current environment."""
        return True

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution."""
        return solution.spec.language == SupportedLanguages.PYTHON

    def _get_cleaner(self, package: str, build_path: Path) -> Callable[[], None]:
        """Create a cleaner function that removes build artifacts.

        The cleaner unloads the imported module, removes it from sys.path, and
        deletes the build directory.

        Parameters
        ----------
        package : str
            The package name to unload from sys.modules.
        build_path : Path
            The directory to delete.

        Returns
        -------
        Callable[[], None]
            A function that performs the cleanup.
        """

        def cleaner() -> None:
            try:
                # Unload module and submodules
                to_delete = [m for m in sys.modules if m == package or m.startswith(package + ".")]
                for m in to_delete:
                    sys.modules.pop(m, None)
            except Exception:
                pass

            try:
                build_path_str = str(build_path)
                if build_path_str in sys.path:
                    sys.path.remove(build_path_str)
            finally:
                shutil.rmtree(build_path, ignore_errors=True)

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a Python solution into a runnable.

        This method writes the solution sources to a temporary directory, imports the
        module, and extracts the entry point function.

        Parameters
        ----------
        definition : Definition
            The problem definition.
        solution : Solution
            The Python solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the Python function.

        Raises
        ------
        BuildError
            If the entry file is not a Python file, the module import fails, or the
            entry symbol is not found or not callable.
        """
        entry_file = solution.get_entry_path()
        entry_symbol = solution.get_entry_symbol()

        if entry_file.suffix != ".py":
            raise BuildError(f"Entry file '{entry_file}' is not a Python file")

        package_name, build_path = self._get_package_name_and_build_path(solution)
        module_name = package_name + "." + ".".join(Path(entry_file).with_suffix("").parts)

        # Create package directory structure: build_path/<package_name>/...
        package_path = build_path / package_name
        write_sources_to_path(package_path, solution.sources)

        cleaner = self._get_cleaner(package_name, build_path)

        # Insert build_path into sys.path so we can import <package_name>.<module>
        sys.path.insert(0, str(build_path))

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            cleaner()
            raise BuildError(f"Failed importing module '{module_name}' from sources: {e}") from e

        try:
            fn: Any = getattr(mod, entry_symbol)
        except AttributeError as e:
            cleaner()
            raise BuildError(
                f"Entry symbol '{entry_symbol}' not found in module '{module_name}'"
            ) from e

        if not callable(fn):
            cleaner()
            raise BuildError(f"Entry symbol '{entry_symbol}' is not callable")

        metadata = RunnableMetadata(
            build_type="python",
            definition_name=definition.name,
            solution_name=solution.name,
            destination_passing_style=solution.spec.destination_passing_style,
            definition=definition,
            misc={"module_name": module_name, "entry_symbol": entry_symbol},
        )

        self._try_validate_signature(fn, definition, solution)

        return Runnable(callable=fn, metadata=metadata, cleaner=cleaner)
