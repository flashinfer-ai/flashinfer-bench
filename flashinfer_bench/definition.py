from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flashinfer_bench.utils.json_utils import from_json, to_json
from flashinfer_bench.utils.validation import (
    validate_axis,
    validate_constraints,
    validate_reference_code,
    validate_tensor,
)


@dataclass
class Definition:
    """Describes a workload definition.

    A Definition provides a formal, machine-readable specification for a
    computational workload found in a model's forward pass.
    """

    name: str
    type: str
    axes: Dict[str, Dict[str, Any]]
    inputs: Dict[str, Dict[str, Any]]
    outputs: Dict[str, Dict[str, Any]]
    reference: str
    tags: Optional[List[str]] = field(default=None)
    description: Optional[str] = field(default=None)
    constraints: Optional[List[str]] = field(default=None)

    def __post_init__(self):
        """Validate the definition after initialization."""
        # Validate required fields
        if not self.name:
            raise ValueError("Definition name cannot be empty")

        if not self.type:
            raise ValueError("Definition type cannot be empty")

        if not self.axes:
            raise ValueError("Definition must have at least one axis")

        if not self.inputs:
            raise ValueError("Definition must have at least one input")

        if not self.outputs:
            raise ValueError("Definition must have at least one output")

        if not self.reference:
            raise ValueError("Definition must have a reference implementation")

        # Validate axes
        for axis_name, axis_def in self.axes.items():
            try:
                validate_axis(axis_def)
            except ValueError as e:
                raise ValueError(f"Invalid axis '{axis_name}': {e}")

        # Validate inputs
        for input_name, input_def in self.inputs.items():
            try:
                validate_tensor(input_def, self.axes)
                # Set default dtype if not specified
                if "dtype" not in input_def:
                    input_def["dtype"] = "float16"
            except ValueError as e:
                raise ValueError(f"Invalid input '{input_name}': {e}")

        # Validate outputs
        for output_name, output_def in self.outputs.items():
            try:
                validate_tensor(output_def, self.axes)
                # Set default dtype if not specified
                if "dtype" not in output_def:
                    output_def["dtype"] = "float16"
            except ValueError as e:
                raise ValueError(f"Invalid output '{output_name}': {e}")

        # Validate reference code
        try:
            validate_reference_code(self.reference)
        except ValueError as e:
            raise ValueError(f"Invalid reference implementation: {e}")

        # Validate tags if present
        if self.tags:
            if not isinstance(self.tags, list):
                raise ValueError("Tags must be a list")
            for tag in self.tags:
                if not isinstance(tag, str):
                    raise ValueError("All tags must be strings")
                if not tag.strip():
                    raise ValueError("Tags cannot be empty strings")

        # Validate constraints if present
        if self.constraints:
            try:
                validate_constraints(self.constraints, self.axes)
            except ValueError as e:
                raise ValueError(f"Invalid constraints: {e}")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> "Definition":
        """Create from JSON string."""
        return from_json(json_str, cls)

    def get_const_axes(self) -> Dict[str, int]:
        """Get all constant axes and their values."""
        return {name: axis["value"] for name, axis in self.axes.items() if axis["type"] == "const"}

    def get_var_axes(self) -> List[str]:
        """Get all variable axis names."""
        return [name for name, axis in self.axes.items() if axis["type"] == "var"]

    def get_input_shapes(self, var_values: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]:
        """Get concrete input shapes given variable axis values."""
        var_values = var_values or {}
        shapes = {}

        for input_name, input_def in self.inputs.items():
            shape = []
            for axis_name in input_def["shape"]:
                if self.axes[axis_name]["type"] == "const":
                    shape.append(self.axes[axis_name]["value"])
                elif axis_name in var_values:
                    shape.append(var_values[axis_name])
                else:
                    raise ValueError(f"Missing value for variable axis '{axis_name}'")
            shapes[input_name] = shape

        return shapes

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[int]]:
        """Get concrete output shapes given variable axis values."""
        var_values = var_values or {}
        shapes = {}

        for output_name, output_def in self.outputs.items():
            shape = []
            for axis_name in output_def["shape"]:
                if self.axes[axis_name]["type"] == "const":
                    shape.append(self.axes[axis_name]["value"])
                elif axis_name in var_values:
                    shape.append(var_values[axis_name])
                else:
                    raise ValueError(f"Missing value for variable axis '{axis_name}'")
            shapes[output_name] = shape

        return shapes
    
    def get_type(self) -> str:
        """Get the type of the definition."""
        return self.type
    
    def get_tags(self) -> List[str]:
        """Get the tags of the definition."""
        return self.tags or []
    
    def has_tag(self, tag: str) -> bool:
        """Check if the definition has a specific tag."""
        return tag in self.get_tags()
    
    def get_tags_by_namespace(self, namespace: str) -> List[str]:
        """Get all tags that belong to a specific namespace (e.g., 'model', 'stage')."""
        prefix = f"{namespace}:"
        return [tag for tag in self.get_tags() if tag.startswith(prefix)]
    
    def get_tag_value(self, namespace: str) -> Optional[str]:
        """Get the value of a namespaced tag (e.g., for 'model:llama-3.1-8b' returns 'llama-3.1-8b')."""
        tags = self.get_tags_by_namespace(namespace)
        if tags:
            # Return the value part after the colon
            return tags[0].split(":", 1)[1] if ":" in tags[0] else None
        return None
