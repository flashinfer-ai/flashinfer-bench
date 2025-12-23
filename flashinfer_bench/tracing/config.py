from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Protocol, Union

from flashinfer_bench.tracing.builtin.policies import (
    BUILTIN_FILTER_POLICIES,
    BUILTIN_INPUT_DUMP_POLICIES,
)
from flashinfer_bench.tracing.workload_entry import WorkloadEntry

InputDumpPolicyLiteral = Literal["dump_all", "dump_none", "dump_int32"]
"""Possible input_dump_policy literals."""


InputDumpPolicyFunction = Callable[Dict[str, Any], List[str]]
"""Function that selects which inputs to dump from input names and values."""


FilterPolicyLiteral = Literal["keep_all", "keep_first", "keep_first_by_axes", "keep_none"]
"""Possible filter policy literals."""


class FilterPolicy(Protocol):
    """Protocol for workload deduplication policy.

    A filter policy maintains internal state and supports both online and offline
    deduplication strategies. Entries are submitted one at a time via submit(),
    and selected entries are retrieved via drain().
    """

    def submit(self, entry: WorkloadEntry) -> None:
        """Submit a workload entry for deduplication consideration.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to submit.
        """
        ...

    def drain(self) -> List[WorkloadEntry]:
        """Drain and return all selected entries.

        Returns
        -------
        List[WorkloadEntry]
            List of entries that passed the deduplication policy.
            After calling this method, the internal buffer is cleared.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of the deduplication policy.

        This method should be called when starting a new deduplication session
        to clear any cached state or statistics.
        """
        ...


FilterPolicyFactory = Callable[[], FilterPolicy]
"""Factory function for filter policy."""


@dataclass
class TracingConfig:
    """Defines how to collect and deduplicate workloads for a definition."""

    input_dump_policy: Union[InputDumpPolicyLiteral, List[str], InputDumpPolicyFunction]
    """Which inputs to persist. Can be:
    - InputDumpPolicyLiteral: string literal for built-in dump functions
    - List[str]: static list of tensor names
    - InputDumpPolicyFunction: custom function that selects tensors from runtime arguments
    """

    filter_policy: Union[FilterPolicyLiteral, FilterPolicyFactory]
    """Deduplication policy factory. Can be a string literal for built-in policies or a factory
    function for custom policies. Can be:
    - FilterPolicyLiteral: string literal for built-in policies
    - FilterPolicyFactory: custom factory function that creates a filter policy instance
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
            factory = BUILTIN_FILTER_POLICIES.get(self.filter_policy)
            if factory is None:
                raise ValueError(
                    f"Unknown filter_policy literal: {self.filter_policy}. "
                    f"Must be one of {list(BUILTIN_FILTER_POLICIES.keys())}"
                )
            self.filter_policy = factory

    def create_filter_policy(self) -> FilterPolicy:
        """Create a new filter policy instance.

        Returns
        -------
        FilterPolicy
            A new policy instance with independent state.
        """
        if callable(self.filter_policy):
            return self.filter_policy()
        else:
            raise TypeError(
                f"filter_policy must be callable after __post_init__, got {type(self.filter_policy)}"
            )

    def get_inputs_to_dump(self, names: List[str], values: List[Any]) -> Dict[str, Any]:
        """Get the inputs to dump from the runtime arguments.

        Parameters
        ----------
        names : List[str]
            Input names in order.
        values : List[Any]
            Input values in order (same length as names).

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping selected input names to their values.

        Raises
        ------
        ValueError
            If input_dump_policy is invalid or returns invalid names.
        """
        name_to_value = dict(zip(names, values))

        if isinstance(self.input_dump_policy, list):
            names_to_dump = self.input_dump_policy
        elif callable(self.input_dump_policy):
            names_to_dump = self.input_dump_policy(name_to_value)
        else:
            raise ValueError("input_dump_policy must be a list of strings or a callable")

        if not isinstance(names_to_dump, list):
            raise ValueError("input_dump_policy callable must return a list of strings")

        result: Dict[str, Any] = {}
        for name in names_to_dump:
            if not isinstance(name, str) or name not in name_to_value:
                raise ValueError(f"input_dump_policy returned invalid input name: {name}")
            result[name] = name_to_value[name]
        return result
