from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from flashinfer_bench.data import Definition
from flashinfer_bench.utils import dtype_str_to_torch_dtype


class Runnable:
    def __init__(
        self, fn: Callable[..., Any], closer: Optional[Callable[[], None]], meta: Dict[str, Any]
    ) -> None:
        """A runnable callable with a required resource closer.

        closer: must be provided by the builder and be idempotent.
        """
        self._fn = fn
        self._closer: Optional[Callable[[], None]] = closer
        self.meta: Dict[str, Any] = meta

    def __call__(self, **kwargs: Any) -> Any:
        """
        - Accept kwargs only (aligns with Definition.inputs naming)
        - Unpack a single-element tuple to a scalar value
        - No type/shape/count validation; errors surface naturally
        """
        ret = self._fn(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 1:
            return ret[0]
        return ret

    def close(self) -> None:
        """Release build artifacts/resources; must be idempotent."""
        if self._closer:
            try:
                self._closer()
            finally:
                self._closer = None


class TVMFFIRunnable(Runnable):
    def __init__(
        self,
        fn: Callable[..., Any],
        closer: Optional[Callable[[], None]],
        meta: Dict[str, Any],
        definition: Definition,
    ) -> None:
        super().__init__(fn, closer, meta)
        self._definition = definition

    def __call__(self, **kwargs: Any) -> Any:
        import torch

        # Allocate output tensors first

        var_values = self._definition.get_var_values(
            {name: list(tensor.shape) for name, tensor in kwargs.items()}
        )
        output_shapes = self._definition.get_output_shapes(var_values)
        output_tensors: Dict[str, torch.Tensor] = {}

        # Determine device from input tensors
        devices = {v.device for v in kwargs.values() if hasattr(v, "device")}
        if len(devices) > 1:
            raise ValueError("All input tensors must be on the same device")
        device = devices.pop() if devices else "cpu"

        for name, shape in output_shapes.items():
            output_tensors[name] = torch.empty(
                shape, dtype=dtype_str_to_torch_dtype(self._definition.outputs[name].dtype)
            ).to(device)

        self.call_dest(**kwargs, **output_tensors)

        results = list(output_tensors.values())
        if len(results) == 1:
            return results[0]
        return results

    def call_dest(self, **kwargs: Any) -> None:
        """Call the underlying function with destination passing style."""
        self._fn(**kwargs)
