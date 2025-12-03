"""Abstract base class for solution builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from flashinfer_bench.data import Definition, Solution
from flashinfer_bench.env import get_fib_cache_path

from .runnable import Runnable
from .utils import create_package_name


class BuildError(RuntimeError):
    """Raised when a builder fails to construct a runnable implementation."""


class Builder(ABC):
    """Abstract base class for building solutions into runnable implementations.

    A Builder transforms a (Definition, Solution) pair into a Runnable object, which is an
    executable implementation of the solution. Different builders handle different programming
    languages (e.g., Python, CUDA, Triton) and build systems.

    Subclasses must implement all its abstract methods. Expectedly, the concrete builder should
    operate in the folder `FIB_CACHE_PATH / builder_specific_subfolder / package_name`, where
    `package_name` is a unique name created from the solution.
    """

    def __init__(self, package_prefix: str, build_dir_name: str) -> None:
        """Initialize the builder.

        Parameters
        ----------
        package_prefix : str
            The prefix to prepend to the package name. This should be unique for each builder type.
        build_dir_name : str
            The name of the build subdirectory of the concrete builder. This should be
            unique for each builder type.
        """
        self._package_prefix = package_prefix
        self._build_dir_name = build_dir_name

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this builder is available in the current environment.

        Override this method in subclasses to check for specific dependencies
        (e.g., CUDA, Triton, TVM). The default implementation returns True.

        Returns
        -------
        bool
            True if the builder can be used, False otherwise.
        """
        ...

    @abstractmethod
    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can handle the given solution.

        Parameters
        ----------
        solution : Solution
            The solution to check.

        Returns
        -------
        bool
            True if this builder can build the solution, False otherwise.
        """
        ...

    @abstractmethod
    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a solution into a runnable implementation.

        This method compiles/loads the solution's source code and returns a Runnable
        object that can be executed with the interface specified by the definition.

        Parameters
        ----------
        definition : Definition
            The problem definition that specifies the expected interface.
        solution : Solution
            The solution implementation to build.

        Returns
        -------
        Runnable
            An executable wrapper around the built implementation.

        Raises
        ------
        BuildError
            If the build fails for any reason (compilation errors, missing dependencies, etc.).
        """
        ...

    def get_package_name_and_build_path(self, solution: Solution) -> Tuple[str, Path]:
        """Get the package name and build path for the solution. The package name is a unique
        identifier for the solution with only alphanumeric characters and underscores. The
        build path is FIB_CACHE_PATH / build_dir_name / package_name.

        Parameters
        ----------
        solution : Solution
            The solution to get the package name and build path for.

        Returns
        -------
        Tuple[str, Path]
            The package name and build path for the solution.
        """
        package_name = create_package_name(solution, self._package_prefix)
        build_path = get_fib_cache_path() / self._build_dir_name / package_name
        return package_name, build_path
