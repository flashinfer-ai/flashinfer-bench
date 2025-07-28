from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flashinfer_bench.utils.json_utils import from_json, to_json


@dataclass
class Solution:
    """Describes a solution implementation.

    A Solution provides a concrete, high-performance implementation for a given Definition.
    Each Solution is self-contained, encapsulating source code and all metadata required
    for building, interfacing, and benchmarking.
    """

    name: str
    definition: str
    author: str
    spec: Dict[str, Any]
    sources: List[Dict[str, str]]
    description: Optional[str] = field(default=None)

    def __post_init__(self):
        """Validate the solution after initialization."""
        # Validate required fields
        if not self.name:
            raise ValueError("Solution name cannot be empty")

        if not self.definition:
            raise ValueError("Solution must reference a definition")

        if not self.author:
            raise ValueError("Solution must have an author")

        if not self.spec:
            raise ValueError("Solution must have a spec")

        if not self.sources:
            raise ValueError("Solution must have at least one source file")

        # Validate spec
        self._validate_spec()

        # Validate sources
        self._validate_sources()

    def _validate_spec(self):
        """Validate the spec dictionary."""
        required_fields = ["language", "target_hardware", "entry_point"]

        for field in required_fields:
            if field not in self.spec:
                raise ValueError(f"Solution spec missing required field '{field}'")

        # Validate language
        if not isinstance(self.spec["language"], str) or not self.spec["language"]:
            raise ValueError("spec.language must be a non-empty string")

        # Validate target_hardware
        if not isinstance(self.spec["target_hardware"], list):
            raise ValueError("spec.target_hardware must be a list")

        if not self.spec["target_hardware"]:
            raise ValueError("spec.target_hardware cannot be empty")

        for hw in self.spec["target_hardware"]:
            if not isinstance(hw, str) or not hw:
                raise ValueError("Each target hardware must be a non-empty string")

        # Validate entry_point
        if not isinstance(self.spec["entry_point"], str) or not self.spec["entry_point"]:
            raise ValueError("spec.entry_point must be a non-empty string")

        # Validate optional fields
        if "dependencies" in self.spec:
            if not isinstance(self.spec["dependencies"], list):
                raise ValueError("spec.dependencies must be a list")

            for dep in self.spec["dependencies"]:
                if not isinstance(dep, str) or not dep:
                    raise ValueError("Each dependency must be a non-empty string")

        if "build_steps" in self.spec:
            if not isinstance(self.spec["build_steps"], list):
                raise ValueError("spec.build_steps must be a list")

            for step in self.spec["build_steps"]:
                if not isinstance(step, str):
                    raise ValueError("Each build step must be a string")

    def _validate_sources(self):
        """Validate the sources list."""
        seen_paths = set()

        for i, source in enumerate(self.sources):
            if not isinstance(source, dict):
                raise ValueError(f"Source {i} must be a dictionary")

            if "path" not in source:
                raise ValueError(f"Source {i} missing required field 'path'")

            if "content" not in source:
                raise ValueError(f"Source {i} missing required field 'content'")

            path = source["path"]

            if not isinstance(path, str) or not path:
                raise ValueError(f"Source {i} path must be a non-empty string")

            if not isinstance(source["content"], str):
                raise ValueError(f"Source {i} content must be a string")

            if path in seen_paths:
                raise ValueError(f"Duplicate source path '{path}'")

            seen_paths.add(path)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> "Solution":
        """Create from JSON string."""
        return from_json(json_str, cls)

    def get_main_source(self) -> Optional[Dict[str, str]]:
        """Get the main source file (main.py)."""
        for source in self.sources:
            if source["path"] == "main.py":
                return source

        # If no main.py, return first Python file
        for source in self.sources:
            if source["path"].endswith(".py"):
                return source

        return None

    def get_source_by_path(self, path: str) -> Optional[Dict[str, str]]:
        """Get a source file by its path."""
        for source in self.sources:
            if source["path"] == path:
                return source
        return None

    def is_jit_compiled(self) -> bool:
        """Check if the solution uses a JIT-compiled language."""
        return self.spec["language"].lower() in ["triton", "jax"]

    def requires_build(self) -> bool:
        """Check if the solution requires a build step."""
        return "build_steps" in self.spec and bool(self.spec["build_steps"])
    
    def get_code(self) -> str:
        """Get the source code of the main file."""
        main_source = self.get_main_source()
        if main_source:
            return main_source["content"]
        return ""
    
    def get_author(self) -> str:
        """Get the author of the solution."""
        return self.author if self.author else "Unknown"
