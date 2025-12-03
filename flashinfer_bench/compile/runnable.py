"""Runnable wrapper for compiled solutions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel

from flashinfer_bench.data import Definition
from flashinfer_bench.utils import dtype_str_to_torch_dtype


class RunnableMetadata(BaseModel):
    """Metadata about a runnable implementation.

    This class stores information about how a runnable was built, including the
    builder type, source definition/solution, and additional builder-specific data.
    """

    build_type: Union[Literal["torch", "tvm_ffi", "python", "triton"], str]
    """The type of build that produced this runnable (e.g., 'python', 'torch', 'triton',
    'tvm_ffi')."""
    definition: str
    """Name of the definition that specifies the expected interface."""
    solution: str
    """Name of the solution that was compiled into this runnable."""
    misc: Dict[str, Any]
    """Miscellaneous metadata about the runnable. Contents vary by builder type."""


class Runnable:
    """An executable wrapper around a compiled solution.

    A Runnable encapsulates a callable function along with metadata about how it was built
    and a cleanup function to release resources. It provides a uniform interface for
    executing solutions regardless of the build system or language used.
    """

    metadata: RunnableMetadata
    """Metadata about the build process and source solution."""

    _callable: Callable[..., Any]
    """The underlying callable function."""
    _cleaner: Optional[Callable[[], None]]
    """Optional cleanup function to release build artifacts and resources."""

    def __init__(
        self,
        callable: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor for the Runnable class.

        Parameters
        ----------
        callable : Callable[..., Any]
            The callable that is wrapped by the runnable.
        metadata : RunnableMetadata
            The metadata for the runnable.
        cleaner : Optional[Callable[[], None]]
            The cleaner function for the runnable. It will clean up the build artifacts/resources.
        """
        self._callable = callable
        self.metadata = metadata
        self._cleaner = cleaner

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the runnable with keyword arguments.

        This method calls the underlying compiled function with the provided inputs.
        If the function returns a single-element tuple, it is automatically unpacked
        to a scalar value for convenience.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for the underlying function.

        Returns
        -------
        Any
            The result of the underlying function. Single-element tuples are unpacked
            to scalar values.
        """
        ret = self._callable(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 1:
            return ret[0]
        return ret

    def call_value_returning(self, **kwargs: Any) -> Any:
        """Call a destination-passing style (DPS) function in value-returning style.

        Some solutions use the destination-passing style,
        where output tensors are passed as arguments and the function modifies them in-place::

            function(**input_tensors, **output_tensors) -> None

        This method provides a value-returning interface by automatically allocating output
        tensors based on the definition, calling the DPS function, and returning the outputs::

            result = runnable.call_dps(**input_tensors)  # -> output_tensors

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for input tensors matching the definition's input specification.

        Returns
        -------
        Any
            The output tensor(s). Single outputs are returned as-is, multiple outputs are
            returned as a tuple, and empty outputs return None.

        Raises
        ------
        ValueError
            If the metadata does not contain the full definition object needed for
            output tensor allocation.
        """
        import torch

        if "definition" not in self.metadata.misc or not isinstance(
            self.metadata.misc["definition"], Definition
        ):
            raise ValueError(
                "When calling in destination passing style, metadata.misc must "
                "contain the full definition."
            )
        definition: Definition = self.metadata.misc["definition"]

        # Allocate output tensors first
        var_values = definition.get_var_values(
            {name: list(tensor.shape) for name, tensor in kwargs.items()}
        )
        output_shapes = definition.get_output_shapes(var_values)
        output_tensors: Dict[str, torch.Tensor] = {}

        # Determine device from input tensors
        devices = {v.device for v in kwargs.values() if hasattr(v, "device")}
        if len(devices) > 1:
            raise ValueError("All input tensors must be on the same device")
        device = devices.pop() if devices else "cpu"

        for name, shape in output_shapes.items():
            output_tensors[name] = torch.empty(
                shape, dtype=dtype_str_to_torch_dtype(definition.outputs[name].dtype)
            ).to(device)

        self._callable(**kwargs, **output_tensors)

        results = tuple(output_tensors.values())
        if len(results) == 0:
            return None
        if len(results) == 1:
            return results[0]
        return results

    def cleanup(self) -> None:
        """Clean up build artifacts and release resources.

        This method calls the cleaner function if one was provided during construction.
        It is idempotent: calling it multiple times is safe and has no additional effect
        after the first call.
        """
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
