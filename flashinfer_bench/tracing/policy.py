"""Policy protocols and registry for tracing configurations."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Type, Union, overload

from .workload_entry import WorkloadEntry

# ============================================================================
# Policy Protocols
# ============================================================================


class InputDumpPolicy(Protocol):
    """Protocol for input dump policy.

    An input dump policy determines which inputs to persist when tracing workloads.
    """

    def dump(self, inputs: Dict[str, Any]) -> List[str]:
        """Select which inputs to dump.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Dictionary mapping input names to their values.

        Returns
        -------
        List[str]
            List of input names to dump.
        """
        ...


class FilterPolicy(Protocol):
    """Protocol for workload filtering and deduplication policy.

    A filter policy maintains internal state and supports both online and offline
    filtering and deduplication strategies. Entries are submitted one at a time via submit(),
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


# ============================================================================
# Policy Registry
# ============================================================================


class PolicyRegistry:
    """Registry for input dump policies and filter policies.

    All policies are registered by name and can be instantiated with kwargs.
    This enables serialization of TracingConfig to YAML/JSON.
    """

    _input_dump_policies: ClassVar[Dict[str, Type[InputDumpPolicy]]] = {}
    _filter_policies: ClassVar[Dict[str, Type[FilterPolicy]]] = {}

    @overload
    @classmethod
    def register_input_dump_policy(
        cls, name: str, policy_cls: Type[InputDumpPolicy], *, override: bool = False
    ) -> Type[InputDumpPolicy]: ...

    @overload
    @classmethod
    def register_input_dump_policy(
        cls, name: str, *, override: bool = False
    ) -> Callable[[Type[InputDumpPolicy]], Type[InputDumpPolicy]]: ...

    @classmethod
    def register_input_dump_policy(
        cls,
        name: str,
        policy_cls: Optional[Type[InputDumpPolicy]] = None,
        *,
        override: bool = False,
    ) -> Union[Type[InputDumpPolicy], Callable[[Type[InputDumpPolicy]], Type[InputDumpPolicy]]]:
        """Register an input dump policy class.

        Can be used as a decorator or called directly.

        Parameters
        ----------
        name : str
            The name to register the policy under.
        policy_cls : Type[InputDumpPolicy], optional
            The policy class to register. If not provided, returns a decorator.
        override : bool, optional
            If True, allows overriding an existing policy with the same name.
            Default is False.

        Returns
        -------
        Type[InputDumpPolicy] | Callable[[Type[InputDumpPolicy]], Type[InputDumpPolicy]]
            If policy_cls is provided, returns the registered policy class.
            Otherwise, returns a decorator that registers the class.

        Raises
        ------
        ValueError
            If a policy with the same name is already registered and override is False.

        Examples
        --------
        As decorator:
        >>> @PolicyRegistry.register_input_dump_policy("dump_all")
        ... class DumpAllPolicy:
        ...     def dump(self, inputs): ...

        Direct call:
        >>> PolicyRegistry.register_input_dump_policy("dump_all", DumpAllPolicy)
        """

        def decorator(cls_to_register: Type[InputDumpPolicy]) -> Type[InputDumpPolicy]:
            if not override and name in cls._input_dump_policies:
                raise ValueError(f"Input dump policy '{name}' already registered")
            cls._input_dump_policies[name] = cls_to_register
            return cls_to_register

        if policy_cls is not None:
            return decorator(policy_cls)
        return decorator

    @classmethod
    def get_input_dump_policy(cls, name: str) -> Optional[Type[InputDumpPolicy]]:
        """Get an input dump policy class by name.

        Parameters
        ----------
        name : str
            The name of the policy to retrieve.

        Returns
        -------
        Type[InputDumpPolicy] | None
            The policy class if found, otherwise None.
        """
        return cls._input_dump_policies.get(name)

    @classmethod
    def list_input_dump_policies(cls) -> list[str]:
        """List all registered input dump policy names.

        Returns
        -------
        list[str]
            A list of all registered input dump policy names.
        """
        return list(cls._input_dump_policies.keys())

    @overload
    @classmethod
    def register_filter_policy(
        cls, name: str, policy_cls: Type[FilterPolicy], *, override: bool = False
    ) -> Type[FilterPolicy]: ...

    @overload
    @classmethod
    def register_filter_policy(
        cls, name: str, *, override: bool = False
    ) -> Callable[[Type[FilterPolicy]], Type[FilterPolicy]]: ...

    @classmethod
    def register_filter_policy(
        cls, name: str, policy_cls: Optional[Type[FilterPolicy]] = None, *, override: bool = False
    ) -> Union[Type[FilterPolicy], Callable[[Type[FilterPolicy]], Type[FilterPolicy]]]:
        """Register a filter policy class.

        Can be used as a decorator or called directly.

        Parameters
        ----------
        name : str
            The name to register the policy under.
        policy_cls : Type[FilterPolicy], optional
            The policy class to register. If not provided, returns a decorator.
        override : bool, optional
            If True, allows overriding an existing policy with the same name.
            Default is False.

        Returns
        -------
        Type[FilterPolicy] | Callable[[Type[FilterPolicy]], Type[FilterPolicy]]
            If policy_cls is provided, returns the registered policy class.
            Otherwise, returns a decorator that registers the class.

        Raises
        ------
        ValueError
            If a policy with the same name is already registered and override is False.

        Examples
        --------
        As decorator:
        >>> @PolicyRegistry.register_filter_policy("keep_all")
        ... class KeepAllPolicy:
        ...     def submit(self, entry): ...
        ...     def drain(self): ...
        ...     def reset(self): ...

        Direct call:
        >>> PolicyRegistry.register_filter_policy("keep_all", KeepAllPolicy)
        """

        def decorator(cls_to_register: Type[FilterPolicy]) -> Type[FilterPolicy]:
            if not override and name in cls._filter_policies:
                raise ValueError(f"Filter policy '{name}' already registered")
            cls._filter_policies[name] = cls_to_register
            return cls_to_register

        if policy_cls is not None:
            return decorator(policy_cls)
        return decorator

    @classmethod
    def get_filter_policy(cls, name: str) -> Optional[Type[FilterPolicy]]:
        """Get a filter policy class by name.

        Parameters
        ----------
        name : str
            The name of the policy to retrieve.

        Returns
        -------
        Type[FilterPolicy] | None
            The policy class if found, otherwise None.
        """
        return cls._filter_policies.get(name)

    @classmethod
    def list_filter_policies(cls) -> list[str]:
        """List all registered filter policy names.

        Returns
        -------
        list[str]
            A list of all registered filter policy names.
        """
        return list(cls._filter_policies.keys())
