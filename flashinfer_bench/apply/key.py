"""Key builders for matching workloads to solutions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict

from flashinfer_bench.data import Definition, Workload


class ApplyKey(BaseModel):
    """Key for matching workloads to solutions in apply runtime.

    This is an immutable (frozen) model that can be used as dict keys or in sets.
    """

    model_config = ConfigDict(frozen=True)

    axes: Tuple[Tuple[str, int], ...] = ()
    """Variable axis values as sorted (name, value) tuples."""
    feats: Tuple[Tuple[str, Union[int, float, bool]], ...] = ()
    """Additional features extracted from input tensors."""


class ApplyKeyBuilder(ABC):
    def __init__(self, definition: Definition) -> None:
        self.definition = definition

    @abstractmethod
    def build_from_args(self, args: Tuple[Any, ...]) -> ApplyKey:
        """Build a key from positional runtime arguments (inputs only)"""
        ...

    @abstractmethod
    def build_from_workload(self, workload: Workload) -> ApplyKey:
        """Build a key from offline workload trace"""
        ...

    @abstractmethod
    def features(self, args: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]:
        """Lightweight feature extraction from input args"""
        ...


# Key Builders


class AxesOnlyKeyBuilder(ApplyKeyBuilder):
    def build_from_args(self, args: Tuple[Any, ...]) -> ApplyKey:
        axes = self.definition.get_axes_values_from_inputs(args)
        return ApplyKey(axes=tuple(sorted(axes.items())))

    def build_from_workload(self, workload: Workload) -> ApplyKey:
        axes = workload.axes
        return ApplyKey(axes=tuple(sorted(axes.items())))

    def features(self, args: Tuple[Any, ...]) -> Tuple[Tuple[str, Any], ...]:
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
    def specialize(cls, definition: Definition) -> ApplyKeyBuilder:
        builder_cls = cls.for_type(definition.op_type)
        return builder_cls(definition)


ApplyKeyFactory.register("gemm", GEMMKeyBuilder)
ApplyKeyFactory.register("gqa", GQAKeyBuilder)
ApplyKeyFactory.register("mla", MLAKeyBuilder)
