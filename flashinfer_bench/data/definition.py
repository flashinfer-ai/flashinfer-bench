"""The definition of kernels in the FlashInfer Trace schema."""

from __future__ import annotations

import ast
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from flashinfer_bench.data.utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt
from flashinfer_bench.utils import dtype_str_to_torch_dtype

if TYPE_CHECKING:
    import torch


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
    description: Optional[str] = Field(default=None)
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
    FLOAT8_E4M3FN = "float8_e4m3fn"
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


AxisSpec = Union[AxisConst, AxisVar]
"""Union type representing all possible axis specifications."""


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
    """The general compute category (e.g., 'gemm', 'gqa_ragged', 'mla_paged', 'moe')."""
    axes: Dict[NonEmptyString, Union[AxisConst, AxisVar]]
    """Dictionary of symbolic dimensions used in tensor shapes. The axes will be bound to the
    input tensor dimensions at runtime."""
    inputs: Dict[NonEmptyString, TensorSpec]
    """Named input tensors required by this kernel. The order of inputs is preserved as the
    kernel's interface."""
    outputs: Dict[NonEmptyString, TensorSpec]
    """Named output tensors produced by this kernel. The names of the output must not overlap
    with the names of the inputs. The order of outputs is preserved as the kernel's interface."""
    reference: NonEmptyString
    """Reference implementation code. It defines the compute logic of the kernel. Must be a valid
    Python code with a 'run' function that takes the input tensors and returns the output tensors.
    """
    tags: List[NonEmptyString] = Field(default_factory=list)
    """Optional list of tags for grouping and filtering kernels. It's used in the FlashInfer-Bench
    website."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the kernel's purpose."""
    constraints: List[NonEmptyString] = Field(default_factory=list)
    """Optional list of constraint expressions describing relationships between axes."""

    @model_validator(mode="after")
    def _validate_reference_code(self) -> Definition:
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
    def _validate_input_output_names(self) -> Definition:
        """Validate that input and output names are unique and do not overlap.

        Raises
        ------
        ValueError
            If the input or output names are not unique or overlap.
        """
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_constraints_syntax(self) -> Definition:
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
    def _validate_tensor_axis_references(self) -> Definition:
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

    @cached_property
    def const_axes(self) -> Dict[str, int]:
        """Get all constant axes and their values.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping constant axis names to their fixed values.
        """
        return {name: axis.value for name, axis in self.axes.items() if isinstance(axis, AxisConst)}

    @cached_property
    def var_axes(self) -> List[str]:
        """Get all variable axis names.

        Returns
        -------
        List[str]
            List of all variable axis names defined in this Definition.
        """
        return [name for name, axis in self.axes.items() if isinstance(axis, AxisVar)]

    @cached_property
    def var_axes_bindings(self) -> Dict[str, Tuple[str, int]]:
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

    def get_axes_values(self, input_shapes: Iterable[Optional[Tuple[int, ...]]]) -> Dict[str, int]:
        """Get concrete variable axis values from input shapes.

        Parameters
        ----------
        input_shapes : Iterable[Optional[Tuple[int, ...]]]
            Iterable of input tensor shapes.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping variable axis names to their concrete values.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from input_shapes, or a axis occurs in
            multiple input tensors, but the values are not consistent.
        """
        var_axes_values: Dict[str, int] = {}
        for (inp_name, inp_spec), inp_shape in zip(self.inputs.items(), input_shapes):
            if inp_spec.shape is None:  # scalar, no shape
                continue
            if len(inp_spec.shape) != len(inp_shape):
                raise ValueError(
                    f"Input '{inp_name}''s defined dimension is {len(inp_spec.shape)} but the "
                    f"actual dimension is {len(inp_shape)}"
                )
            for axis_name, axis_value in zip(inp_spec.shape, inp_shape):
                if axis_name in self.axes and self.axes[axis_name].type == "var":
                    if axis_name in var_axes_values:
                        if var_axes_values[axis_name] != axis_value:
                            raise ValueError(
                                f"Axis '{axis_name}' has different values for different input "
                                f"tensors: {var_axes_values[axis_name]} and {axis_value}"
                            )
                    else:
                        var_axes_values[axis_name] = axis_value

        if len(var_axes_values) != len(self.var_axes):
            raise ValueError(
                f"Missing values for variable axes: "
                f"{set(self.var_axes) - set(var_axes_values.keys())}"
            )
        return var_axes_values

    def get_axes_values_from_inputs(self, inputs: Iterable[Any]) -> Dict[str, int]:
        """Get concrete variable axis values directly from input values.

        Convenience method that combines extract_shapes and get_var_axes_values.

        Parameters
        ----------
        inputs : Iterable[Any]
            Iterable of input values (tensors or other types).

        Returns
        -------
        Dict[str, int]
            Dictionary mapping variable axis names to their concrete values.
        """
        shapes = [tuple(val.shape) if hasattr(val, "shape") else None for val in inputs]
        return self.get_axes_values(shapes)

    def _get_shapes(
        self, tensors: Iterable[TensorSpec], var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
        """Get concrete tensor shapes given variable axis values.

        Parameters
        ----------
        tensors : List[TensorSpec]
            List of tensor specifications to compute shapes for.
        var_values : Optional[Dict[str, int]], default=None
            Values for variable axes. If None, defaults to empty dictionary.

        Returns
        -------
        List[Optional[Tuple[int, ...]]]
            List of concrete shapes as tuples of integers. None for scalar tensors.

        Raises
        ------
        ValueError
            If a required variable axis value is missing from var_values.
        """
        var_axes_values = var_axes_values or {}
        shapes = []

        for tensor_spec in tensors:
            if tensor_spec.shape is None:  # scalar, no shape
                shapes.append(None)
                continue
            shape = []
            for axis_name in tensor_spec.shape:
                axis = self.axes[axis_name]
                if isinstance(axis, AxisConst):
                    shape.append(axis.value)
                elif isinstance(axis, AxisVar):
                    if axis_name not in var_axes_values:
                        raise ValueError(f"Missing value for variable axis '{axis_name}'")
                    shape.append(var_axes_values[axis_name])
            shapes.append(shape)

        return shapes

    def get_input_shapes(
        self, var_axes_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
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
        return self._get_shapes(self.inputs.values(), var_axes_values)

    def get_output_shapes(
        self, var_values: Optional[Dict[str, int]] = None
    ) -> List[Optional[Tuple[int, ...]]]:
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
        return self._get_shapes(self.outputs.values(), var_values)

    @cached_property
    def torch_input_dtypes(self) -> List[torch.dtype]:
        """Get the torch data types of the input tensors.

        Returns
        -------
        List[torch.dtype]
            List of torch data types of the input tensors.
        """
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.inputs.values()]

    @cached_property
    def torch_output_dtypes(self) -> List[torch.dtype]:
        """Get the torch data types of the output tensors.

        Returns
        -------
        List[torch.dtype]
            List of torch data types of the output tensors.
        """
        return [dtype_str_to_torch_dtype(spec.dtype) for spec in self.outputs.values()]

    def merge_kwargs_to_args(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Merge keyword arguments into positional arguments based on input/output order.

        Parameters
        ----------
        args : Tuple[Any, ...]
            Positional arguments.
        kwargs : Dict[str, Any]
            Keyword arguments to merge.

        Returns
        -------
        Tuple[Any, ...]
            Merged positional arguments in the order: inputs, then outputs.
        """
        if not kwargs:
            return args

        param_names = list(self.inputs.keys()) + list(self.outputs.keys())

        if len(args) > len(param_names):
            raise TypeError(
                f"Too many positional arguments: got {len(args)}, "
                f"expected at most {len(param_names)}"
            )

        # Check for duplicate arguments
        positional_arg_names = set(param_names[: len(args)])
        for name in kwargs:
            if name in positional_arg_names:
                raise TypeError(f"Got multiple values for argument '{name}'")

        result = list(args)
        for i in range(len(args), len(param_names)):
            name = param_names[i]
            if name in kwargs:
                result.append(kwargs[name])
            else:
                break
        return tuple(result)
