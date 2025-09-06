"""Strong-typed data definitions for solution implementations."""

import ast
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SupportedLanguages(Enum):
    PYTHON = "python"
    TRITON = "triton"
    CUDA = "cuda"


@dataclass
class SourceFile:
    """A single source code file."""

    path: str
    content: str

    def __post_init__(self):
        if not self.path:
            raise ValueError("SourceFile path cannot be empty")
        if not isinstance(self.content, str):
            raise ValueError("SourceFile content must be a string")

        if self.path.endswith(".py"):
            try:
                ast.parse(self.content, mode="exec")
            except SyntaxError as e:
                raise ValueError(f"SourceFile content must be valid Python code: {e}") from e

        # TODO(shanli): syntax validation for other languages


@dataclass
class BuildSpec:
    """Build specification for a solution."""

    language: SupportedLanguages
    target_hardware: List[str]
    entry_point: str
    dependencies: Optional[List[str]] = None
    build_commands: Optional[List[str]] = None

    def __post_init__(self):
        if not isinstance(self.language, SupportedLanguages):
            raise ValueError("language must be of SupportedLanguages type")

        # Validate target_hardware
        if not isinstance(self.target_hardware, list):
            raise ValueError("target_hardware must be a list")
        if not self.target_hardware:
            raise ValueError("target_hardware cannot be empty")
        for hw in self.target_hardware:
            if not isinstance(hw, str) or not hw:
                raise ValueError("Each target hardware must be a non-empty string")

        # Validate entry_point
        if not self.entry_point:
            raise ValueError("entry_point cannot be empty")
        if "::" not in self.entry_point:
            raise ValueError("spec.entry_point must be '<relative_file.py>::<function_name>'")
        # TODO(shanli): validations against entry file existence and function existence

        # Validate dependencies if present
        if self.dependencies is not None:
            if not isinstance(self.dependencies, list):
                raise ValueError("dependencies must be a list")
            for dep in self.dependencies:
                if not isinstance(dep, str) or not dep:
                    raise ValueError("Each dependency must be a non-empty string")
        # TODO(shanli): more structured dependency specification and validation

        # Validate build_commands if present
        if self.build_commands is not None:
            if not isinstance(self.build_commands, list):
                raise ValueError("build_commands must be a list")
            for cmd in self.build_commands:
                if not isinstance(cmd, str):
                    raise ValueError("Each build command must be a string")


@dataclass
class Solution:
    """A concrete implementation for a given Definition."""

    name: str
    definition: str  # Name of the Definition this solves
    author: str
    spec: BuildSpec
    sources: List[SourceFile]
    description: Optional[str] = None

    def __post_init__(self):
        # Basic validation
        if not self.name:
            raise ValueError("Solution name cannot be empty")

        if not self.definition:
            raise ValueError("Solution must reference a definition")

        if not self.author:
            raise ValueError("Solution must have an author")

        if not isinstance(self.spec, BuildSpec):
            raise ValueError("Solution spec must be a BuildSpec")

        if not isinstance(self.sources, list):
            raise ValueError("Solution sources must be a list of source files")

        if not self.sources:
            raise ValueError("Solution must have at least one source file")

        # Validate sources
        entry_file = self.spec.entry_point.split("::")[0]

        seen_paths = set()
        for i, source in enumerate(self.sources):
            if not isinstance(source, SourceFile):
                raise ValueError(f"Source {i} must be a SourceFile")

            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        # TODO(shanli): stronger validation for entry file and function
        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file."""
        path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == path:
                return source

    def requires_build(self) -> bool:
        """Check if the solution requires a build step."""
        return bool(self.spec.build_commands) or self.spec.language == SupportedLanguages.CUDA

    def to_json(self) -> str:
        """Serialize the Solution to JSON string."""
        import json

        def serialize_obj(obj):
            """Recursively serialize dataclass objects and enums."""
            if hasattr(obj, "__dataclass_fields__"):
                # Handle dataclass objects
                result = {}
                for field_name in obj.__dataclass_fields__:
                    field_value = getattr(obj, field_name)
                    result[field_name] = serialize_obj(field_value)
                return result
            elif isinstance(obj, Enum):
                # Handle enum objects
                return obj.value
            elif isinstance(obj, list):
                # Handle lists
                return [serialize_obj(item) for item in obj]
            elif isinstance(obj, dict):
                # Handle dictionaries
                return {key: serialize_obj(value) for key, value in obj.items()}
            else:
                # Handle primitive types
                return obj

        serialized_data = serialize_obj(self)
        return json.dumps(serialized_data, indent=2, ensure_ascii=False)
