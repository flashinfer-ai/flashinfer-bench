from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

from .tracing_policy import (
    BUILTIN_DEDUP_POLICY_FACTORIES,
    BUILTIN_TENSORS_DUMP_FUNCTIONS,
    DedupPolicy,
    DedupPolicyFactory,
    TensorsToDumpFunction,
)

TensorsToDumpLiteral = Literal["dump_all", "dump_none", "dump_int32"]
"""Possible tensors_to_dump literals."""


DedupPolicyLiteral = Literal["keep_all", "keep_first", "keep_first_by_axes"]
"""Possible dedup policy literals."""


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[TensorsToDumpLiteral, List[str], TensorsToDumpFunction]
    """Which inputs to persist. Can be:
    - TensorsToDumpLiteral: string literal for built-in dump functions
    - List[str]: static list of tensor names
    - TensorsToDumpFunction: custom function that selects tensors from runtime arguments
    """

    dedup_policy: Union[DedupPolicyLiteral, DedupPolicyFactory]
    """Deduplication policy factory. Can be a string literal for built-in policies or a factory
    function for custom policies. Can be:
    - DedupPolicyLiteral: string literal for built-in policies
    - DedupPolicyFactory: custom factory function that creates a dedup policy instance
    """

    def __post_init__(self):
        """Convert literal strings to actual functions/factories."""
        # Resolve tensors_to_dump literal
        if isinstance(self.tensors_to_dump, str):
            dump_func = BUILTIN_TENSORS_DUMP_FUNCTIONS.get(self.tensors_to_dump)
            if dump_func is None:
                raise ValueError(
                    f"Unknown tensors_to_dump literal: {self.tensors_to_dump}. "
                    f"Must be one of {list(BUILTIN_TENSORS_DUMP_FUNCTIONS.keys())}"
                )
            self.tensors_to_dump = dump_func

        # Resolve dedup_policy literal
        if isinstance(self.dedup_policy, str):
            factory = BUILTIN_DEDUP_POLICY_FACTORIES.get(self.dedup_policy)
            if factory is None:
                raise ValueError(
                    f"Unknown dedup_policy literal: {self.dedup_policy}. "
                    f"Must be one of {list(BUILTIN_DEDUP_POLICY_FACTORIES.keys())}"
                )
            self.dedup_policy = factory

    def create_dedup_policy(self) -> DedupPolicy:
        """Create a new dedup policy instance.

        Returns
        -------
        DedupPolicy
            A new policy instance with independent state.
        """
        if callable(self.dedup_policy):
            return self.dedup_policy()
        else:
            raise TypeError(
                f"dedup_policy must be callable after __post_init__, got {type(self.dedup_policy)}"
            )

    def get_tensors_to_dump(self, runtime_args: Dict[str, Any]) -> List[str]:
        """Get the tensors to dump from the runtime arguments. The validity of the result is
        checked, so every returned tensor name must exist in the runtime arguments.

        Parameters
        ----------
        runtime_args : Dict[str, Any]
            The runtime arguments to get the tensors to dump from.

        Returns
        -------
        List[str]
            The tensors to dump.

        Raises
        ------
        ValueError
            If tensors_to_dump is not a list of strings or a callable, or the result is not valid.
        """
        if isinstance(self.tensors_to_dump, list):
            result = self.tensors_to_dump
        elif callable(self.tensors_to_dump):
            result = self.tensors_to_dump(runtime_args)
        else:
            raise ValueError("tensors_to_dump must be a list of strings or a callable")

        # Check the validity of the result
        if not isinstance(result, list):
            raise ValueError("tensors_to_dump callable must return a list of strings")
        for name in result:
            if not isinstance(name, str):
                raise ValueError(
                    f"tensors_to_dump callable must return a list of strings, but got "
                    f"{type(name).__name__}"
                )
            if name not in runtime_args:
                raise ValueError(
                    f"tensors_to_dump callable returned {name} which is not in runtime_args"
                )
        return result
