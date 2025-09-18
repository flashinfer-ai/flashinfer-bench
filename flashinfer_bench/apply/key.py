from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type, Union

from flashinfer_bench.data import AxisVar, Definition, Workload


@dataclass(frozen=True)
class ApplyKey:
    axes: Tuple[Tuple[str, int], ...] = field(default_factory=tuple)
    # Features extracted from input tensors
    feats: Tuple[Tuple[str, Union[int, Union[float, bool]]], ...] = field(default_factory=tuple)

    @property
    def encode(self) -> str:
        return json.dumps(
            {"axes": list(self.axes), "feats": list(self.feats)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_encoded(cls, s: str) -> "ApplyKey":
        d = json.loads(s)
        axes = tuple((k, int(v)) for k, v in d.get("axes", []))
        feats = tuple((k, v) for k, v in d.get("feats", []))
        return cls(axes=axes, feats=feats)

    def __hash__(self) -> int:
        return hash((self.axes, self.feats))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ApplyKey):
            return False
        return self.axes == other.axes and self.feats == other.feats


class ApplyKeyBuilder(ABC):
    def __init__(self, defn: Definition) -> None:
        self.defn = defn
        # axis -> (input, dim_idx)
        self._axis_proj: Dict[str, Tuple[str, int]] = self._collect_var_axis_projections(defn)

    @abstractmethod
    def build_from_runtime(self, runtime_kwargs: Dict[str, Any]) -> ApplyKey:
        """Build a key from runtime kwargs"""
        ...

    @abstractmethod
    def build_from_workload(self, workload: Workload) -> ApplyKey:
        """Build a key from offline workload trace"""
        ...

    @abstractmethod
    def features(self, runtime_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """Lightweight feature extraction"""
        ...

    def _collect_var_axis_projections(self, defn: Definition) -> Dict[str, Tuple[str, int]]:
        """
        Iterate over the shape of inputs, find the first occurrence of each var axis:
          axis_name -> (input_name, dim_idx)
        """
        proj: Dict[str, Tuple[str, int]] = {}
        axis_defs = defn.axes
        inputs = defn.inputs

        for inp_name, spec in inputs.items():
            shape = spec.shape
            if shape is None:  # scalar
                continue
            for dim_idx, axis_name in enumerate(shape):
                axis_def = axis_defs.get(axis_name)
                if axis_def is None:
                    continue
                if isinstance(axis_def, AxisVar) and axis_name not in proj:
                    proj[axis_name] = (inp_name, dim_idx)

        var_axes = [k for k, v in axis_defs.items() if isinstance(v, AxisVar)]
        missing = [ax for ax in var_axes if ax not in proj]
        if missing:
            raise ValueError(f"Cannot locate var axes {missing} from inputs of '{defn.name}'")
        return proj

    def _materialize_axes(self, runtime_kwargs: Dict[str, Any]) -> Dict[str, int]:
        axes: Dict[str, int] = {}
        for axis, (inp, dim_idx) in self._axis_proj.items():
            if inp not in runtime_kwargs:
                raise KeyError(
                    f"Missing runtime input '{inp}' for axis '{axis}' in '{self.defn.name}'"
                )
            val = runtime_kwargs[inp]
            shape = val.shape
            if dim_idx >= len(shape):
                raise ValueError(
                    f"Input '{inp}' rank {len(shape)} < expected dim {dim_idx} for axis '{axis}'"
                )
            axes[axis] = int(shape[dim_idx])
        return axes


# Key Builders


class AxesOnlyKeyBuilder(ApplyKeyBuilder):
    def build_from_runtime(self, runtime_kwargs: Dict[str, Any]) -> ApplyKey:
        axes = self._materialize_axes(runtime_kwargs)
        return ApplyKey(axes=tuple(sorted(axes.items())))

    def build_from_workload(self, workload: Workload) -> ApplyKey:
        axes = workload.axes
        return ApplyKey(axes=tuple(sorted(axes.items())))

    def features(self, runtime_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return ()


# TODO(shanli): add more feature specific keys (e.g. avg_seq_len)
class GEMMKeyBuilder(AxesOnlyKeyBuilder):
    pass


class GQAKeyBuilder(AxesOnlyKeyBuilder):
    pass


class MLAKeyBuilder(AxesOnlyKeyBuilder):
    pass


class ApplyKeyFactory:
    _REGISTRY: Dict[str, Type[ApplyKeyBuilder]] = {}

    @classmethod
    def register(cls, type_name: str, builder_cls: Type[ApplyKeyBuilder]) -> None:
        cls._REGISTRY[type_name] = builder_cls

    @classmethod
    def for_type(cls, type_name: str) -> Type[ApplyKeyBuilder]:
        # Default to AxesOnlyKeyBuilder if not registered
        return cls._REGISTRY.get(type_name, AxesOnlyKeyBuilder)

    @classmethod
    def specialize(cls, defn: Definition) -> ApplyKeyBuilder:
        builder_cls = cls.for_type(defn.op_type)
        return builder_cls(defn)


ApplyKeyFactory.register("gemm", GEMMKeyBuilder)
ApplyKeyFactory.register("gqa", GQAKeyBuilder)
ApplyKeyFactory.register("mla", MLAKeyBuilder)
