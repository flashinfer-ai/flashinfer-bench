from __future__ import annotations

from typing import Any, Mapping

from flashinfer_bench.data.definition import Definition


class ShapeResolver:
    """Placeholder: precompute shape->axis mappings; infer var axes."""

    def __init__(self, definitions: Mapping[str, Definition]):
        self._definitions = definitions
        #  build input->(dim_idx->axis) mapping table

    def infer_axes(self, def_name: str, runtime_args: dict[str, Any]) -> dict[str, int]:
        raise NotImplementedError
