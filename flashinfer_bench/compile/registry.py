"""Builder registry for dispatching and caching builds."""

from __future__ import annotations

from typing import ClassVar, Dict, List, Type

from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, SupportedLanguages

from .builder import Builder, BuildError
from .builders import PythonBuilder, TorchBuilder, TritonBuilder, TVMFFIBuilder
from .runnable import Runnable

_BUILDER_PRIORITY: List[Type[Builder]] = [TritonBuilder, PythonBuilder, TVMFFIBuilder, TorchBuilder]
"""Builder types in priority order for automatic selection."""


class BuilderRegistry:
    """Central registry for managing and dispatching builders.

    The BuilderRegistry maintains a list of available builders and automatically selects
    the appropriate one for each solution. It also provides caching to avoid redundant
    builds of the same solution.

    This class follows the singleton pattern - use get_instance() to obtain the shared
    registry instance.
    """

    _instance: ClassVar["BuilderRegistry" | None] = None
    """Singleton instance of the BuilderRegistry."""

    _builders: List[Builder]
    """List of available builders in priority order."""

    _cache: Dict[str, Runnable]
    """Cache mapping solution hashes to built runnables."""

    def __init__(self, builders: List[Builder]) -> None:
        """Initialize the registry with a list of builders.

        Parameters
        ----------
        builders : List[Builder]
            List of builder instances to use. Must contain at least one builder.

        Raises
        ------
        ValueError
            If the builders list is empty.
        """
        if len(builders) == 0:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders = list(builders)
        self._cache: Dict[str, Runnable] = {}

    @classmethod
    def get_instance(cls) -> "BuilderRegistry":
        """Get the singleton registry instance.

        On first call, this method initializes the registry by instantiating all
        available builders (those whose is_available() returns True) in priority order.
        Subsequent calls return the same instance.

        The following builders are available (high to low priority):

        - TritonBuilder: Build Triton solutions.
        - PythonBuilder: Build Python solutions.
        - TVMFFIBuilder: Build CUDA/C++ solutions using TVM-FFI backend.
        - TorchBuilder: Build CUDA/C++ solutions using PyTorch extension system.

        Returns
        -------
        BuilderRegistry
            The shared registry instance.
        """
        if cls._instance is None:
            builders = []
            for builder_type in _BUILDER_PRIORITY:
                if builder_type.is_available():
                    builders.append(builder_type())
            cls._instance = BuilderRegistry(builders)
        return cls._instance

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a solution into a runnable, using cache if available.

        This method first checks if the solution has already been built (by comparing
        its hash). If not, it tries each registered builder in priority order until
        one reports it can build the solution. The resulting runnable is cached for
        future use.

        Parameters
        ----------
        definition : Definition
            The problem definition specifying the expected interface.
        solution : Solution
            The solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the built solution.

        Raises
        ------
        BuildError
            If no registered builder can build this solution, or if the build fails.
        """
        hash = solution.hash()
        if hash in self._cache:
            return self._cache[hash]

        for builder in self._builders:
            # Choose the first builder that can build the solution
            if builder.can_build(solution):
                runnable = builder.build(definition, solution)
                self._cache[hash] = runnable
                return runnable
        raise BuildError(f"No registered builder can build solution '{solution.name}'")

    def build_reference(self, definition: Definition) -> Runnable:
        """Build the reference implementation for a definition.

        This is a convenience method that creates a pseudo-solution from the definition's
        reference code and builds it using the standard build() method.

        Parameters
        ----------
        definition : Definition
            The definition containing the reference implementation.

        Returns
        -------
        Runnable
            An executable wrapper around the reference implementation.

        Raises
        ------
        BuildError
            If the reference implementation cannot be built.
        """
        pseudo = Solution(
            name=f"{definition.name}__reference",
            definition=definition.name,
            author="__builtin__",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["cuda"],
                entry_point="main.py::run",
            ),
            sources=[SourceFile(path="main.py", content=definition.reference)],
            description="reference",
        )
        return self.build(definition, pseudo)

    def cleanup(self) -> None:
        """Cleanup the cache and cleanup all built runnables.

        This method calls cleanup() on all cached runnables to release resources,
        then clears the cache. Cleanup errors are caught and ignored to ensure
        all runnables are cleaned up.
        """
        for runnable in self._cache.values():
            try:
                runnable.cleanup()
            except Exception:
                pass
        self._cache.clear()
