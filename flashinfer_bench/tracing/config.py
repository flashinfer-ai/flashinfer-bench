from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

from .builtin.policies import (
    BUILTIN_DEDUP_POLICIES,
    BUILTIN_INPUT_DUMP_POLICIES,
    InputDumpPolicyFunction,
)
from .dedup_policy import DedupPolicy, DedupPolicyFactory

InputDumpPolicyLiteral = Literal["dump_all", "dump_none", "dump_int32"]
"""Possible input_dump_policy literals."""


DedupPolicyLiteral = Literal["keep_all", "keep_first", "keep_first_by_axes", "keep_none"]
"""Possible dedup policy literals."""


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    input_dump_policy: Union[InputDumpPolicyLiteral, List[str], InputDumpPolicyFunction]
    """Which inputs to persist. Can be:
    - InputDumpPolicyLiteral: string literal for built-in dump functions
    - List[str]: static list of tensor names
    - InputDumpPolicyFunction: custom function that selects tensors from runtime arguments
    """

    filter_policy: Union[DedupPolicyLiteral, DedupPolicyFactory]
    """Deduplication policy factory. Can be a string literal for built-in policies or a factory
    function for custom policies. Can be:
    - DedupPolicyLiteral: string literal for built-in policies
    - DedupPolicyFactory: custom factory function that creates a dedup policy instance
    """

    def __post_init__(self):
        """Convert literal strings to actual functions/factories."""
        # Resolve input_dump_policy literal
        if isinstance(self.input_dump_policy, str):
            dump_func = BUILTIN_INPUT_DUMP_POLICIES.get(self.input_dump_policy)
            if dump_func is None:
                raise ValueError(
                    f"Unknown input_dump_policy literal: {self.input_dump_policy}. "
                    f"Must be one of {list(BUILTIN_INPUT_DUMP_POLICIES.keys())}"
                )
            self.input_dump_policy = dump_func

        # Resolve filter_policy literal
        if isinstance(self.filter_policy, str):
            factory = BUILTIN_DEDUP_POLICIES.get(self.filter_policy)
            if factory is None:
                raise ValueError(
                    f"Unknown filter_policy literal: {self.filter_policy}. "
                    f"Must be one of {list(BUILTIN_DEDUP_POLICIES.keys())}"
                )
            self.filter_policy = factory

    def create_filter_policy(self) -> DedupPolicy:
        """Create a new dedup policy instance.

        Returns
        -------
        DedupPolicy
            A new policy instance with independent state.
        """
        if callable(self.filter_policy):
            return self.filter_policy()
        else:
            raise TypeError(
                f"filter_policy must be callable after __post_init__, got {type(self.filter_policy)}"
            )

    def get_inputs_to_dump(self, runtime_args: Dict[str, Any]) -> List[str]:
        """Get the inputs to dump from the runtime arguments. The validity of the result is
        checked, so every returned input name must exist in the runtime arguments.

        Parameters
        ----------
        runtime_args : Dict[str, Any]
            The runtime arguments to get the inputs to dump from.

        Returns
        -------
        List[str]
            The inputs to dump.

        Raises
        ------
        ValueError
            If input_dump_policy is not a list of strings or a callable, or the result is not valid.
        """
        if isinstance(self.input_dump_policy, list):
            result = self.input_dump_policy
        elif callable(self.input_dump_policy):
            result = self.input_dump_policy(runtime_args)
        else:
            raise ValueError("input_dump_policy must be a list of strings or a callable")

        # Check the validity of the result
        if not isinstance(result, list):
            raise ValueError("input_dump_policy callable must return a list of strings")
        for name in result:
            if not isinstance(name, str):
                raise ValueError(
                    f"input_dump_policy callable must return a list of strings, but got "
                    f"{type(name).__name__}"
                )
            if name not in runtime_args:
                raise ValueError(
                    f"input_dump_policy callable returned {name} which is not in runtime_args"
                )
        return result
