"""Generate JSON Schema from function signatures and docstrings."""

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Literal, Union, get_args, get_origin, get_type_hints

import docstring_parser
from pydantic import BaseModel


def _is_pydantic_model(python_type: Any) -> bool:
    """Check if a type is a Pydantic model class."""
    try:
        return isinstance(python_type, type) and issubclass(python_type, BaseModel)
    except TypeError:
        return False


def _python_type_to_json_schema(python_type: Any) -> dict:
    """Convert Python type annotation to JSON Schema type."""
    if python_type is None or python_type is type(None):
        return {"type": "null"}

    # Handle Pydantic models
    if _is_pydantic_model(python_type):
        return python_type.model_json_schema()

    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Literal types -> enum
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Handle Optional[X] which is Union[X, None]
    if origin is Union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # Optional[X] -> just return schema for X
            return _python_type_to_json_schema(non_none_args[0])
        # Union of multiple types - deduplicate by converting to JSON and back
        schemas = [_python_type_to_json_schema(a) for a in non_none_args]
        unique_schemas = []
        seen = set()
        for s in schemas:
            key = str(sorted(s.items()))
            if key not in seen:
                seen.add(key)
                unique_schemas.append(s)
        if len(unique_schemas) == 1:
            return unique_schemas[0]
        return {"anyOf": unique_schemas}

    # Handle List[X]
    if origin is list:
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle Dict[K, V] - only generate additionalProperties if K is str
    if origin is dict:
        if args and len(args) == 2 and args[0] is str:
            return {"type": "object", "additionalProperties": _python_type_to_json_schema(args[1])}
        return {"type": "object"}

    # Handle basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    return type_map.get(python_type, {"type": "string"})


def function_to_schema(func: Callable) -> dict:
    """Generate OpenAI/Anthropic compatible tool schema from a function.

    This function extracts information from:
    - Function name
    - Type hints for parameter types
    - Numpy-style docstring for descriptions and parameter docs

    Parameters
    ----------
    func : Callable
        The function to generate schema for. Must have type hints and
        numpy-style docstring.

    Returns
    -------
    dict
        A tool schema compatible with OpenAI/Anthropic function calling format.

    Examples
    --------
    >>> from flashinfer_bench.agents.schema import function_to_schema
    >>> from flashinfer_bench.agents import flashinfer_bench_run_ncu
    >>> schema = function_to_schema(flashinfer_bench_run_ncu)
    >>> print(schema["name"])
    flashinfer_bench_run_ncu
    """
    # Parse docstring
    doc = docstring_parser.parse(func.__doc__ or "")

    # Get type hints (use empty dict if fails)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Get signature
    sig = inspect.signature(func)

    # Build parameter descriptions from docstring
    param_docs = {p.arg_name: p.description for p in doc.params}

    # Build properties
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # Get type from hints or default to string
        param_type = hints.get(name, str)

        # Build property schema
        prop_schema = _python_type_to_json_schema(param_type)

        # Add description from docstring (normalize whitespace)
        if name in param_docs and param_docs[name]:
            desc = " ".join(param_docs[name].split())
            prop_schema["description"] = desc

        # Add default if present
        if param.default is not inspect.Parameter.empty and param.default is not None:
            prop_schema["default"] = param.default

        properties[name] = prop_schema

        # Track required parameters (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    # Build description from docstring
    description = doc.short_description or ""
    if doc.long_description:
        description += "\n\n" + doc.long_description

    return {
        "name": func.__name__,
        "description": description.strip(),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def get_all_tool_schemas() -> List[dict]:
    """Get schemas for all agent tools in flashinfer_bench.

    Returns
    -------
    List[dict]
        List of tool schemas compatible with OpenAI/Anthropic function calling.

    Examples
    --------
    >>> from flashinfer_bench.agents.schema import get_all_tool_schemas
    >>> schemas = get_all_tool_schemas()
    >>> for s in schemas:
    ...     print(s["name"])
    flashinfer_bench_list_ncu_options
    flashinfer_bench_run_ncu
    flashinfer_bench_run_sanitizer
    """
    from flashinfer_bench.agents.ncu import (
        flashinfer_bench_list_ncu_options,
        flashinfer_bench_run_ncu,
    )
    from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer

    tools = [
        flashinfer_bench_list_ncu_options,
        flashinfer_bench_run_ncu,
        flashinfer_bench_run_sanitizer,
    ]

    return [function_to_schema(func) for func in tools]
