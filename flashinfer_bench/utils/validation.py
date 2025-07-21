from typing import Any, Dict, List

ALLOWED_DTYPES = [
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
    "float4_e2m1",
    "int8",
    "int16",
    "int32",
    "bool",
]


def validate_dtype(dtype: str) -> None:
    """Validate that a dtype is allowed."""
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Invalid dtype '{dtype}'. Allowed values: {ALLOWED_DTYPES}")


def validate_axis(axis: Dict[str, Any]) -> None:
    """Validate an axis definition."""
    if "type" not in axis:
        raise ValueError("Axis must have a 'type' field")

    axis_type = axis["type"]

    if axis_type == "const":
        if "value" not in axis:
            raise ValueError("Const axis must have a 'value' field")
        if not isinstance(axis["value"], int) or axis["value"] <= 0:
            raise ValueError("Const axis value must be a positive integer")

    elif axis_type == "var":
        if "parent" in axis and not isinstance(axis["parent"], (str, type(None))):
            raise ValueError("Axis parent must be a string or None")

    else:
        raise ValueError(f"Invalid axis type '{axis_type}'. Must be 'const' or 'var'")


def validate_tensor(tensor: Dict[str, Any], axes: Dict[str, Dict[str, Any]]) -> None:
    """Validate a tensor definition."""
    if "shape" not in tensor:
        raise ValueError("Tensor must have a 'shape' field")

    if not isinstance(tensor["shape"], list):
        raise ValueError("Tensor shape must be a list")

    # Validate shape references valid axes
    for axis_name in tensor["shape"]:
        if axis_name not in axes:
            raise ValueError(f"Tensor shape references undefined axis '{axis_name}'")

    # Validate dtype if present
    if "dtype" in tensor:
        validate_dtype(tensor["dtype"])


def validate_reference_code(code: str) -> None:
    """Validate that reference code is valid Python with a run function."""
    try:
        # Check it's valid Python
        compile(code, "<reference>", "exec")

        # Check for run function
        namespace = {}
        exec(code, namespace)

        if "run" not in namespace:
            raise ValueError("Reference code must contain a 'run' function")

        if not callable(namespace["run"]):
            raise ValueError("'run' must be a callable function")

    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax in reference code: {e}")
    except Exception as e:
        raise ValueError(f"Error in reference code: {e}")


def validate_workload_axes(
    workload_axes: Dict[str, int], definition_axes: Dict[str, Dict[str, Any]]
) -> None:
    """Validate that workload axes match definition variable axes."""
    # Check all var axes are provided
    for axis_name, axis_def in definition_axes.items():
        if axis_def["type"] == "var":
            if axis_name not in workload_axes:
                raise ValueError(f"Workload missing required variable axis '{axis_name}'")
            if not isinstance(workload_axes[axis_name], int) or workload_axes[axis_name] <= 0:
                raise ValueError(f"Workload axis '{axis_name}' must be a positive integer")

    # Check no extra axes
    for axis_name in workload_axes:
        if axis_name not in definition_axes:
            raise ValueError(f"Workload contains unknown axis '{axis_name}'")
        if definition_axes[axis_name]["type"] != "var":
            raise ValueError(f"Workload cannot override const axis '{axis_name}'")


def validate_constraints(constraints: List[str], axes: Dict[str, Dict[str, Any]]) -> None:
    """Validate constraint expressions reference valid axes."""
    for constraint in constraints:
        if not isinstance(constraint, str):
            raise ValueError("Constraints must be strings")
        # Basic validation - could be enhanced with proper parsing
        if not constraint.strip():
            raise ValueError("Constraint cannot be empty")
