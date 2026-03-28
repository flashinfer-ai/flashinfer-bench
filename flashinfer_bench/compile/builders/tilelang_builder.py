"""Builder for TileLang GPU kernels."""

from __future__ import annotations

import importlib.util
from typing import ClassVar

from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data import Definition, Solution, SupportedLanguages

from .python_builder import PythonBuilder


class TileLangBuilder(PythonBuilder):
    """Builder for TileLang solutions.

    This builder extends PythonBuilder to handle TileLang GPU kernels. TileLang code
    is Python-based, so the build process is similar to PythonBuilder, with the
    main difference being the language tag in metadata.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "fib_tilelang_"
    """Prefix for cache keys to distinguish TileLang solutions from pure Python ones."""

    _BUILD_DIR_NAME: ClassVar[str] = "tilelang"
    """Subdirectory under FIB_CACHE_PATH where build results are stored"""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def is_available() -> bool:
        """Check if TileLang is available in the current environment.

        Returns
        -------
        bool
            True if TileLang is installed, False otherwise.
        """
        return importlib.util.find_spec("tilelang") is not None

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can build the given solution.
        The solution should be TileLang source code.

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is TileLang
        """
        return solution.spec.language == SupportedLanguages.TILELANG

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a TileLang solution into a runnable.

        This method delegates to PythonBuilder.build() and updates the build_type
        in metadata to 'TileLang'.

        Parameters
        ----------
        definition : Definition
            The problem definition.
        solution : Solution
            The TileLang solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the TileLang kernel.
        """
        result = super().build(definition, solution)
        result.metadata.build_type = "tilelang"
        return result
