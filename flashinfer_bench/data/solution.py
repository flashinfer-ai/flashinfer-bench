"""Strong-typed data definitions for solution implementations."""

import ast
from enum import Enum
from typing import List, Optional

from pydantic import Field, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString


class SupportedLanguages(str, Enum):
    """Supported programming languages for solution implementations.

    Enumeration of programming languages that can be used to implement
    solutions for computational workloads.
    """

    PYTHON = "python"
    """Python programming language."""
    TRITON = "triton"
    """Triton GPU programming language."""
    CUDA = "cuda"
    """CUDA C++ programming language."""


class SourceFile(BaseModelWithDocstrings):
    """A single source code file in a solution implementation.

    Represents a source code file with its relative path and complete content.
    The file content is validated for syntax correctness based on the file extension.
    """

    path: NonEmptyString
    """The relative path of the file, including its name and extension (e.g., 'src/kernel.cu',
    'main.py'). When compiling the solution, a temporary solution source directory will be
    created, and the file will be placed according to this path."""
    content: NonEmptyString
    """The complete text content of the source file."""

    @model_validator(mode="after")
    def _validate_python_syntax(self) -> "SourceFile":
        """Validate Python syntax for .py files.

        Raises
        ------
        ValueError
            If the file is a Python file and contains invalid syntax.
        """
        if self.path.endswith(".py"):
            try:
                ast.parse(self.content, mode="exec")
            except SyntaxError as e:
                raise ValueError(f"SourceFile content must be valid Python code: {e}") from e

        # TODO(shanli): syntax validation for other languages
        return self


class BuildSpec(BaseModelWithDocstrings):
    """Build specification for a solution implementation.

    Contains all technical specifications required to build and execute
    a solution, including language, hardware targets, dependencies,
    entry point, and build commands.
    """

    language: SupportedLanguages
    """The primary programming language (e.g., 'triton', 'cuda', 'python')."""
    target_hardware: List[str] = Field(min_length=1)
    """List of hardware architectures this solution is compatible with (e.g., 'NVIDIA_H100', 'NVIDIA_B200')."""
    entry_point: NonEmptyString
    """The exact path to the function to be called. Format: '{file_path}::{function_name}' (e.g., 'main.py::run')."""
    dependencies: Optional[List[NonEmptyString]] = Field(default=None)
    """Optional list of required libraries or toolchains (e.g., 'CUDA >= 12.0', 'triton >= 2.2')."""
    build_commands: Optional[List[str]] = Field(default=None)
    """Optional list of shell commands required to build the source code."""

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "BuildSpec":
        """Validate entry_point format.

        Raises
        ------
        ValueError
            If entry_point doesn't follow the required format.
        """
        if "::" not in self.entry_point:
            raise ValueError("spec.entry_point must be '<relative_file.py>::<function_name>'")
        # TODO(shanli): validations against entry file existence and function existence
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete implementation for a given Definition.

    Represents a complete solution that provides a high-performance implementation
    for a computational workload defined by a Definition. Contains all source code,
    build specifications, and metadata required for building, interfacing, and
    benchmarking the implementation.
    """

    name: NonEmptyString
    """A unique, human-readable name for this specific solution (e.g., 'rmsnorm_triton_v1_h100')."""
    definition: NonEmptyString
    """The name of the Definition this implementation solves."""
    author: NonEmptyString
    """The name of the author or agent system that created this solution."""
    spec: BuildSpec
    """Technical specifications for building and executing this solution."""
    sources: List[SourceFile] = Field(min_length=1)
    """Array of source code files representing the complete implementation."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the solution's technique or approach."""

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Solution":
        """Validate that all source file paths are unique.

        Raises
        ------
        ValueError
            If duplicate source file paths are found, or the entry point file is not found in the
            sources.
        """
        seen_paths = set()
        for i, source in enumerate(self.sources):
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::")[0]

        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")
        # TODO(shanli): stronger validation for entry file and function

        return self

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file specified in the build spec.

        Returns
        -------
        Optional[SourceFile]
            The SourceFile object containing the entry point, or None if not found.
        """
        entry_path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    def requires_build(self) -> bool:
        """Check if the solution requires a build step.

        Returns
        -------
        bool
            True if the solution requires building (has build commands or uses CUDA),
            False otherwise.
        """
        return bool(self.spec.build_commands) or self.spec.language == SupportedLanguages.CUDA
