"""Configuration for tracing workloads. All configs are serializable to YAML/JSON."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .policy import FilterPolicy, InputDumpPolicy, PolicyRegistry


class TracingConfig(BaseModel):
    """Defines how to collect and deduplicate workloads for a definition.

    This class is fully serializable to YAML/JSON. All policies are referenced
    by name and instantiated with kwargs when needed.

    See also: :mod:`flashinfer_bench.tracing.presets` for builtin policies and config presets.
    """

    input_dump_policy: Union[str, List[str]] = "dump_none"
    """Policy for selecting which tensor inputs to dump.

    Can be either:
    - A registered policy name (str): The policy's `dump(inputs)` method determines the names of
      the tensors to save.
    - A static list of input names (List[str]): Explicitly specify the names of the tensors to save.

    Builtin policies:
    - "dump_all": Dump all tensor inputs.
    - "dump_none": Dump no tensor inputs (only axes are recorded).
    - "dump_int": Dump only integer and boolean tensor inputs (e.g., indptrs, indices).
    """

    input_dump_policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the input dump policy constructor."""

    filter_policy: str = "keep_all"
    """Policy for filtering/deduplicating workload entries.

    The policy's `submit(entry)` method is called for each entry, and `drain()` returns
    the filtered entries. This allows deduplication based on axes, sequence lengths, etc.

    Builtin policies:
    - "keep_all": Keep all entries without deduplication.
    - "keep_none": Keep no entries (discard all).
    - "keep_first": Keep only the first entry.
    - "keep_first_k": Keep the first k entries. Requires kwarg `k`.
    - "keep_first_k_by_axes": Keep first entry per unique axes combination.
    - "attention": Deduplicate by computed avg_kv_len and avg_q_len from indptr tensors.
    """

    filter_policy_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the filter policy constructor."""

    @model_validator(mode="after")
    def _validate_policies(self) -> "TracingConfig":
        """Validate that policies are registered."""
        if isinstance(self.input_dump_policy, str):
            if PolicyRegistry.get_input_dump_policy(self.input_dump_policy) is None:
                raise ValueError(
                    f"Unknown input_dump_policy: {self.input_dump_policy}. "
                    f"Available: {PolicyRegistry.list_input_dump_policies()}"
                )

        if PolicyRegistry.get_filter_policy(self.filter_policy) is None:
            raise ValueError(
                f"Unknown filter_policy: {self.filter_policy}. "
                f"Available: {PolicyRegistry.list_filter_policies()}"
            )
        return self

    def create_input_dump_policy(self) -> InputDumpPolicy:
        """Create a new input dump policy instance.

        Returns
        -------
        InputDumpPolicy
            A new policy instance.

        Raises
        ------
        ValueError
            If input_dump_policy is a static list (use get_inputs_to_dump instead).
        """
        if isinstance(self.input_dump_policy, list):
            raise ValueError("input_dump_policy is a static list, use get_inputs_to_dump() instead")
        policy_cls = PolicyRegistry.get_input_dump_policy(self.input_dump_policy)
        if policy_cls is None:
            raise ValueError(f"Unknown input_dump_policy: {self.input_dump_policy}")
        return policy_cls(**self.input_dump_policy_kwargs)

    def create_filter_policy(self) -> FilterPolicy:
        """Create a new filter policy instance.

        Returns
        -------
        FilterPolicy
            A new policy instance with independent state.
        """
        policy_cls = PolicyRegistry.get_filter_policy(self.filter_policy)
        if policy_cls is None:
            raise ValueError(f"Unknown filter_policy: {self.filter_policy}")
        return policy_cls(**self.filter_policy_kwargs)

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
            If input_dump_policy returns invalid names.
        """
        name_to_value = dict(zip(names, values))

        if isinstance(self.input_dump_policy, list):
            names_to_dump = self.input_dump_policy
        else:
            policy = self.create_input_dump_policy()
            names_to_dump = policy.dump(name_to_value)

        if not isinstance(names_to_dump, list):
            raise ValueError("input_dump_policy.dump() must return a list of strings")

        result: Dict[str, Any] = {}
        for name in names_to_dump:
            if not isinstance(name, str) or name not in name_to_value:
                raise ValueError(f"input_dump_policy returned invalid input name: {name}")
            result[name] = name_to_value[name]
        return result


class TracingConfigRegistry(BaseModel):
    """Per-definition tracing configuration registry.

    If default is not set, only kernels explicitly registered in per_definition are traced.
    Set default to a TracingConfig to trace all kernels.
    """

    default: Optional[TracingConfig] = None
    """Fallback config for definitions not in per_definition. None means skip."""

    per_definition: Dict[str, TracingConfig] = Field(default_factory=dict)
    """Mapping from definition name to its specific TracingConfig."""

    def get(self, def_name: str) -> Optional[TracingConfig]:
        """Get config for a definition, falling back to default if not registered.

        Parameters
        ----------
        def_name : str
            The definition name to look up.

        Returns
        -------
        Optional[TracingConfig]
            The config for this definition. If the definition don't need to be traced, return None.
        """
        return self.per_definition.get(def_name, self.default)

    def register(
        self, def_name: str, config: TracingConfig, *, override: bool = False
    ) -> "TracingConfigRegistry":
        """Register config for a single definition.

        Parameters
        ----------
        def_name : str
            The definition name.
        config : TracingConfig
            The config to use.
        override : bool
            If the existing config will be overridden. If False, raise an error if
            the definition already exists. Default: False.

        Returns
        -------
        TracingConfigRegistry
            Self, for method chaining.

        Raises
        ------
        ValueError
            If def_name already exists and override is False.
        """
        if not override and def_name in self.per_definition:
            raise ValueError(
                f"Definition '{def_name}' already exists. Use override=True to replace it."
            )
        self.per_definition[def_name] = config
        return self

    def register_many(
        self, configs: Dict[str, TracingConfig], *, override: bool = False
    ) -> "TracingConfigRegistry":
        """Register configs for multiple definitions.

        Parameters
        ----------
        configs : Dict[str, TracingConfig]
            Mapping from definition names to their configs.
        override : bool
            If the existing configs will be overridden. If False, raise an error if
            any key already exists. Default: False.

        Returns
        -------
        TracingConfigRegistry
            Self, for method chaining.

        Raises
        ------
        ValueError
            If any key already exists and override is False.
        """
        if not override:
            conflicts = self.per_definition.keys() & configs.keys()
            if conflicts:
                raise ValueError(
                    "Cannot register. The following definitions already exist: "
                    f"{list(conflicts)}"
                )
        self.per_definition.update(configs)
        return self
