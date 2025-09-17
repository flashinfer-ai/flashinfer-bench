"""The definition of kernels in the FlashInfer Trace schema."""

import ast
from enum import Enum
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class AxisConst(BaseModelWithDocstrings):
    """Constant axis with a fixed value.

    A constant axis represents a dimension that has a fixed, compile-time known value.
    This is useful for dimensions that don't vary across different instances of the
    same kernel definition, such as embedding dimensions or hidden layer sizes.
    """

    type: Literal["const"] = "const"
    """The type identifier for constant axes."""
    value: NonNegativeInt
    """The constant integer value of this axis dimension."""
    description: Optional[str] = None
    """An optional human-readable description explaining the purpose of this axis."""


class AxisVar(BaseModel):
    """Variable axis that can be specified at runtime.

    A variable axis represents a dimension whose value is determined at runtime
    based on the actual input data. Its value will be bound to the input tensor
    dimension at runtime.
    """

    type: Literal["var"] = "var"
    """The type identifier for variable axes."""
    description: Optional[str] = None
    """An optional human-readable description explaining the purpose of this axis."""


class DType(str, Enum):
    """Supported data types for tensors.

    Enumeration of all data types that can be used in tensor specifications.
    Includes both floating-point and integer types commonly used in machine
    learning and high-performance computing applications.
    """

    FLOAT32 = "float32"
    """32-bit IEEE 754 floating point."""
    FLOAT16 = "float16"
    """16-bit IEEE 754 half-precision floating point."""
    BFLOAT16 = "bfloat16"
    """16-bit Brain Floating Point format."""
    FLOAT8_E4M3 = "float8_e4m3"
    """8-bit floating point with 4 exponent bits and 3 mantissa bits."""
    FLOAT8_E5M2 = "float8_e5m2"
    """8-bit floating point with 5 exponent bits and 2 mantissa bits."""
    FLOAT4_E2M1 = "float4_e2m1"
    """4-bit floating point with 2 exponent bits and 1 mantissa bit."""
    INT64 = "int64"
    """64-bit signed integer."""
    INT32 = "int32"
    """32-bit signed integer."""
    INT16 = "int16"
    """16-bit signed integer."""
    INT8 = "int8"
    """8-bit signed integer."""
    BOOL = "bool"
    """Boolean type."""


class TensorSpec(BaseModelWithDocstrings):
    """Specification for a tensor including shape and data type, to use as input or output of a
    kernel.

    This includes the symbolic shape (referencing defined axes) and the data type.
    Scalars are represented with a None shape.
    """

    shape: Optional[List[NonEmptyString]]
    """List of axis names defining the tensor shape. None for scalar values."""
    dtype: DType
    """The data type of all elements in this tensor."""
    description: Optional[str] = None
    """An optional human-readable description of this tensor's purpose and usage."""


class Definition(BaseModelWithDocstrings):
    """Complete definition of a computational workload.

    A Definition provides a formal, machine-readable specification for a computational
    workload. It defines the tensor formats, dimension semantics, and computational
    logic through a reference implementation. This serves as the single source of
    truth for kernel development and optimization.
    """

    name: NonEmptyString
    """A unique, human-readable name for the kernel definition."""
    op_type: NonEmptyString
    """The general compute category (e.g., 'gemm', 'gqa_paged', 'mla_ragged')."""
    axes: Dict[NonEmptyString, Union[AxisConst, AxisVar]]
    """Dictionary of symbolic dimensions used in tensor shapes. The axes will be bound to the
    input tensor dimensions at runtime."""
    inputs: Dict[NonEmptyString, TensorSpec]
    """Named input tensors required by this kernel."""
    outputs: Dict[NonEmptyString, TensorSpec]
    """Named output tensors produced by this kernel."""
    reference: NonEmptyString
    """Reference implementation code. It defines the compute logic of the kernel. Must be a valid
    Python code with a 'run' function that takes the input tensors and returns the output tensors.
    """
    tags: Optional[List[NonEmptyString]] = Field(default=None)
    """Optional list of tags for grouping and filtering kernels. It's used in the FlashInfer-Bench
    website."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the kernel's purpose."""
    constraints: Optional[List[NonEmptyString]] = Field(default=None)
    """Optional list of constraint expressions describing relationships between axes."""

    @model_validator(mode="after")
    def _validate_reference_code(self) -> "Definition":
        """Validate that reference contains valid Python code with a 'run' function.

        Raises
        ------
        ValueError
            If the reference code is not valid Python syntax or doesn't contain
            a top-level 'run' function.
        """
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python code: {e}") from e

        # Check for 'run' function
        has_run_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "run" for node in mod.body
        )
        if not has_run_func:
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_constraints_syntax(self) -> "Definition":
        """Validate that constraints are valid Python expressions.

        Raises
        ------
        ValueError
            If any constraint is not a valid Python expression.
        """
        if self.constraints is not None:
            for constraint in self.constraints:
                try:
                    ast.parse(constraint, mode="eval")
                except SyntaxError as e:
                    raise ValueError(f"Constraints must be valid Python expressions: {e}") from e
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> "Definition":
        """Validate that tensor shapes reference defined axes.

        Ensures that all axis names used in input and output tensor shapes
        are properly defined in the axes dictionary.

        Raises
        ------
        ValueError
            If any tensor shape references an undefined axis.
        """
        all_tensors = {**self.inputs, **self.outputs}

        for tensor_name, tensor_spec in all_tensors.items():
            if tensor_spec.shape is not None:
                for axis_name in tensor_spec.shape:
                    if axis_name not in self.axes:
                        tensor_type = "input" if tensor_name in self.inputs else "output"
                        raise ValueError(
                            f'{tensor_type.capitalize()} "{tensor_name}" references undefined '
                            f'axis "{axis_name}"'
                        )
        return self

    def get_const_axes(self) -> Dict[str, int]:
        """Get all constant axes and their values.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping constant axis names to their fixed values.
        """
        return {name: axis.value for name, axis in self.axes.items() if isinstance(axis, AxisConst)}

    def get_var_axes(self) -> List[str]:
        """Get all variable axis names.

        Returns
        -------
        List[str]
            List of all variable axis names defined in this Definition.
        """
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def get_var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
        """Get the bindings of variable axes to input tensor dimensions.

        Determines which input tensor and dimension index corresponds to each
        variable axis. If multiple input tensors share the same axis, the
        binding will be to the first tensor encountered.

        Returns
        -------
        Dict[str, Tuple[str, int]]
            Dictionary mapping axis names to tuples of (input_tensor_name, dimension_index).
            Only includes variable axes that appear in input tensor shapes.
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
        """Get concrete tensor shapes given variable axis values.

        Parameters
        ----------
        tensors : Dict[str, TensorSpec]
            Dictionary of tensor specifications to compute shapes for.
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping tensor names to their concrete shapes as lists of integers.
            Scalar tensors (shape=None) are excluded from the result.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
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
        """Get concrete input shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping input tensor names to their concrete shapes.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        return self._get_shapes(self.inputs, var_values)

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[int]]:
        """Get concrete output shapes given variable axis values.

        Parameters
        ----------
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping output tensor names to their concrete shapes.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        return self._get_shapes(self.outputs, var_values)
