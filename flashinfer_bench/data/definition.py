"""Strong-typed data definitions for workload specifications."""

import ast
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Union


@dataclass
class AxisConst:
    """Constant axis with a fixed value."""

    type: Literal["const"] = "const"
    value: int = 0
    description: Optional[str] = None

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError(f"AxisConst value must be positive, got {self.value}")


@dataclass
class AxisVar:
    """Variable axis that can be specified at runtime."""

    type: Literal["var"] = "var"
    parent: Optional[str] = None
    description: Optional[str] = None


ALLOWED_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
    "float4_e2m1",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
}


@dataclass
class TensorSpec:
    """Specification for a tensor including shape and data type."""

    shape: Optional[List[str]]
    dtype: Literal[
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3",
        "float8_e5m2",
        "float4_e2m1",
        "int64",
        "int32",
        "int16",
        "int8",
        "bool",
    ]
    description: Optional[str] = None

    def __post_init__(self):
        if self.shape is not None and not isinstance(self.shape, list):
            raise ValueError(f"TensorSpec shape must be a list or None, got {type(self.shape)}")

        # Validate dtype is one of the allowed values
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"Invalid dtype '{self.dtype}'. Must be one of {ALLOWED_DTYPES}")


@dataclass
class Definition:
    """Complete definition of a computational workload."""

    name: str
    type: str
    axes: Dict[str, Union[AxisConst, AxisVar]]
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]
    reference: str
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    constraints: Optional[List[str]] = None

    def __post_init__(self):
        # Basic structural validation
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

        # Validate axes are proper types
        for axis_name, axis_def in self.axes.items():
            if not isinstance(axis_def, (AxisConst, AxisVar)):
                raise ValueError(f"Axis '{axis_name}' must be either AxisConst or AxisVar")

        # Validate tensor specs reference valid axes
        for input_name, input_spec in self.inputs.items():
            if not isinstance(input_spec, TensorSpec):
                raise ValueError(f"Input '{input_name}' must be a TensorSpec")
            if input_spec.shape is not None:
                for axis_name in input_spec.shape:
                    if axis_name not in self.axes:
                        raise ValueError(
                            f"Input '{input_name}' references undefined axis '{axis_name}'"
                        )

        for output_name, output_spec in self.outputs.items():
            if not isinstance(output_spec, TensorSpec):
                raise ValueError(f"Output '{output_name}' must be a TensorSpec")
            if output_spec.shape is not None:
                for axis_name in output_spec.shape:
                    if axis_name not in self.axes:
                        raise ValueError(
                            f"Output '{output_name}' references undefined axis '{axis_name}'"
                        )

        # Validate reference code
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python code: {e}") from e
        run_func = None
        for node in mod.body:
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_func = node
                break
        if run_func is None:
            raise ValueError("Reference must define a top-level function named 'run'")
        # TODO(shanli): validate inputs/outputs for matching definition signature

        # Validate tags if present
        if self.tags is not None:
            if not isinstance(self.tags, list):
                raise ValueError("Tags must be a list")
            for tag in self.tags:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("Tags must be non-empty strings")

        # Validate constraints if present
        if self.constraints is not None:
            if not isinstance(self.constraints, list):
                raise ValueError("Constraints must be a list")
            for constraint in self.constraints:
                if not isinstance(constraint, str) or not constraint.strip():
                    raise ValueError("Constraints must be non-empty strings")
                try:
                    ast.parse(constraint, mode="eval")
                except SyntaxError as e:
                    raise ValueError(f"Constraints must be valid Python expressions: {e}") from e

    def get_const_axes(self) -> Dict[str, int]:
        """Get all constant axes and their values."""
        return {name: axis.value for name, axis in self.axes.items() if isinstance(axis, AxisConst)}

    def get_var_axes(self) -> List[str]:
        """Get all variable axis names."""
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def get_var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
        """
        Get the bindings of variable axes to input tensors dimensions.
        Returns:
            Dict[str, Tuple[str, int]]: axis_name -> (input_name, dim_idx)
        """
        bindings: Dict[str, Tuple[str, int]] = {}
        for inp_name, spec in self.inputs.items():
            if spec.shape is None:  # scalar, no shape
                continue
            for dim_idx, axis in enumerate(spec.shape):
                ax_def = self.axes.get(axis)
                if isinstance(ax_def, AxisVar) and axis not in bindings:
                    bindings[axis] = (inp_name, dim_idx)
        return bindings

    def _get_shapes(
        self, tensors: Dict[str, TensorSpec], var_values: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[int]]:
        """Get concrete tensor shapes given variable axis values."""
        var_values = var_values or {}
        shapes = {}

        for tensor_name, tensor_spec in tensors.items():
            if tensor_spec.shape is None:  # scalar, no shape
                continue
            shape = []
            for axis_name in tensor_spec.shape:
                axis = self.axes[axis_name]
                if isinstance(axis, AxisConst):
                    shape.append(axis.value)
                elif isinstance(axis, AxisVar):
                    if axis_name not in var_values:
                        raise ValueError(f"Missing value for variable axis '{axis_name}'")
                    shape.append(var_values[axis_name])
            shapes[tensor_name] = shape

        return shapes

    def get_input_shapes(self, var_values: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]:
        """Get concrete input shapes given variable axis values."""
        return self._get_shapes(self.inputs, var_values)

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[int]]:
        """Get concrete output shapes given variable axis values."""
        return self._get_shapes(self.outputs, var_values)
