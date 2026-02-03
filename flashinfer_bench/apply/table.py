"""Apply table for mapping workload keys to optimal solutions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Trace, TraceSet
from flashinfer_bench.env import get_fib_cache_path

from .config import ApplyConfigRegistry
from .key import ApplyKey, ApplyKeyFactory


def _apply_table_dir() -> Path:
    """Get the directory for storing apply table cache files.

    Returns
    -------
    Path
        The apply table cache directory path.
    """
    return get_fib_cache_path() / "apply_table"


@dataclass
class ApplyTable:
    """Apply table for mapping workload keys to optimal solutions.

    This class manages a lookup table that maps workload characteristics (ApplyKey)
    to the best performing solution for each kernel definition. It supports caching
    to disk and ahead-of-time compilation of frequently used solutions.
    """

    digest: str
    """Hash digest identifying this table's configuration and data."""
    index: Dict[str, Dict[ApplyKey, str]] = field(default_factory=dict)
    """Mapping from definition name to (key -> solution_name) lookup."""
    def_best: Dict[str, str] = field(default_factory=dict)
    """Mapping from definition name to best overall solution name."""

    @classmethod
    def _load_from_disk(cls, digest: str) -> Optional[Dict[str, Any]]:
        """Load apply table data from cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the cached data.

        Returns
        -------
        Optional[Dict[str, Any]]
            The cached data if exists, None otherwise.
        """
        index_path = _apply_table_dir() / f"{digest}.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)

        return None

    @classmethod
    def _save_to_disk(cls, digest: str, data: Dict[str, Any]) -> None:
        """Save apply table data to cache file.

        Parameters
        ----------
        digest : str
            The digest hash identifying the data to cache.
        data : Dict[str, Any]
            The data to save to cache.
        """
        index_path = _apply_table_dir() / f"{digest}.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_or_build(
        cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry
    ) -> "ApplyTable":
        """Load an existing apply table from cache or build a new one.

        This method first attempts to load a cached apply table based on the
        digest of the trace set and configuration. If no cached version exists,
        it builds a new table from scratch and caches it for future use.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry for building the apply table.

        Returns
        -------
        ApplyTable
            The loaded or newly built apply table.
        """
        digest = cls._digest(trace_set, config_registry)

        # Try to load from cache
        raw = cls._load_from_disk(digest)
        if raw is not None:
            index: Dict[str, Dict[ApplyKey, str]] = {}
            for def_name, items in raw.get("index", {}).items():
                bucket: Dict[ApplyKey, str] = {}
                for key_enc, sol_name in items.items():
                    key = ApplyKey.model_validate_json(key_enc)
                    bucket[key] = sol_name
                index[def_name] = bucket

            def_best: Dict[str, str] = raw.get("def_best", {})

            table = cls(digest=digest, index=index, def_best=def_best)
            cls._prewarm_aot(trace_set, config_registry, table)

            return table

        # Build fresh
        table = cls._build(trace_set, config_registry)
        # Persist minimal index
        to_dump: Dict[str, Any] = {"digest": table.digest, "index": {}, "def_best": {}}
        for def_name, bucket in table.index.items():
            for key, sol_name in bucket.items():
                to_dump["index"].setdefault(def_name, {})[key.model_dump_json()] = sol_name
        # Always compute and persist def_best
        for def_name, sol_name in table.def_best.items():
            to_dump["def_best"][def_name] = sol_name

        cls._save_to_disk(digest, to_dump)
        cls._prewarm_aot(trace_set, config_registry, table)

        return table

    @classmethod
    def _build(cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry) -> "ApplyTable":
        """Build a new apply table from trace set data.

        This method processes all traces in the trace set to build lookup tables
        mapping workload keys to optimal solutions for each kernel definition.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry including error tolerances.

        Returns
        -------
        ApplyTable
            The newly built apply table.
        """
        digest = cls._digest(trace_set, config_registry)

        index: Dict[str, Dict[ApplyKey, str]] = {}
        def_best: Dict[str, str] = {}

        for def_name, definition in trace_set.definitions.items():
            config = config_registry.get(def_name)
            if config is None:
                continue
            per_key, ranked = cls._sweep_def(trace_set, def_name, config.max_atol, config.max_rtol)

            # Build index
            for key, t in per_key.items():
                if not t.solution:
                    continue
                bucket = index.setdefault(def_name, {})
                bucket[key] = t.solution

            # Build def_best
            if ranked:
                def_best[def_name] = ranked[0][0]

        return cls(digest=digest, index=index, def_best=def_best)

    @classmethod
    def _sweep_def(
        cls, trace_set: TraceSet, def_name: str, max_atol: float, max_rtol: float
    ) -> Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]]]:
        """Sweep through traces for a definition to find optimal solutions per key.

        This method processes all traces for a given kernel definition, groups them
        by workload key, and selects the best performing solution for each key.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data.
        def_name : str
            Name of the kernel definition to process.
        max_atol : float
            Maximum absolute error tolerance for filtering traces.
        max_rtol : float
            Maximum relative error tolerance for filtering traces.

        Returns
        -------
        Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]]]
            A tuple containing:
            - Dictionary mapping keys to best traces
            - List of (solution_name, win_count) pairs sorted by wins
        """
        traces = trace_set.filter_traces(def_name, max_atol, max_rtol)
        builder = ApplyKeyFactory.specialize(trace_set.definitions[def_name])

        # Pick the trace with the highest speedup_factor for each key
        per_key: Dict[ApplyKey, Trace] = {}
        for t in traces:
            key = builder.build_from_workload(t.workload)
            prev = per_key.get(key)
            if (
                prev is None
                or t.evaluation.performance.speedup_factor
                > prev.evaluation.performance.speedup_factor
            ):
                per_key[key] = t

        # Count wins per solution
        win_counts: Dict[str, int] = {}
        for t in per_key.values():
            if t.solution:
                win_counts[t.solution] = win_counts.get(t.solution, 0) + 1

        ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
        return per_key, ranked

    @classmethod
    def _prewarm_aot(
        cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry, table: "ApplyTable"
    ) -> None:
        """Perform ahead-of-time compilation of frequently used solutions.

        This method pre-compiles solutions that are used frequently according to
        the AOT ratio configuration. This reduces runtime compilation overhead.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing definitions and solutions.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry containing AOT ratio and miss policy settings.
        table : ApplyTable
            The apply table containing solution mappings.
        """
        reg = BuilderRegistry.get_instance()

        for def_name, bucket in table.index.items():
            if not bucket:
                continue

            config = config_registry.get(def_name)
            if config is None:
                continue
            if not (config.aot_ratio and config.aot_ratio > 0.0):
                continue

            win_counts = Counter(bucket.values())
            ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            cutoff = max(1, int(len(ranked) * config.aot_ratio))

            definition = trace_set.definitions.get(def_name)
            if not definition:
                continue
            for sol_name, _ in ranked[:cutoff]:
                solution = trace_set.get_solution(sol_name)
                if solution:
                    reg.build(definition, solution)

        # Build def_best for definitions with on_miss_policy == "use_def_best"
        for def_name, sol_name in table.def_best.items():
            config = config_registry.get(def_name)
            if config is None:
                continue
            if config.on_miss_policy == "use_def_best":
                definition = trace_set.definitions.get(def_name)
                solution = trace_set.get_solution(sol_name)
                if definition and solution:
                    reg.build(definition, solution)

    @classmethod
    def _digest(cls, trace_set: TraceSet, config_registry: ApplyConfigRegistry) -> str:
        """Compute a hash digest for the trace set and configuration.

        This method creates a deterministic hash that uniquely identifies the
        combination of trace set data and apply configuration. The digest is
        used for caching apply tables.

        Parameters
        ----------
        trace_set : TraceSet
            The trace set containing benchmark data.
        config_registry : ApplyConfigRegistry
            Per-definition configuration registry.

        Returns
        -------
        str
            SHA256 hash digest as a hexadecimal string.
        """
        trace_set_dict = trace_set.to_dict()
        for definition in trace_set_dict["definitions"].values():
            for drop in ("description", "tags", "reference", "constraints"):
                definition.pop(drop, None)
        for sol_list in trace_set_dict["solutions"].values():
            for solution in sol_list:
                spec = solution.get("spec", {}) or {}
                deps = spec.get("dependencies") or []
                spec["dependencies"] = sorted(deps)
                new_sources = []
                for sf in solution.get("sources") or []:
                    new_sources.append(
                        {
                            "path": sf["path"],
                            "sha1": hashlib.sha1(sf["content"].encode("utf-8")).hexdigest(),
                        }
                    )
                solution["sources"] = new_sources
        kept_traces: List[Dict[str, Any]] = []
        for traces in trace_set_dict["traces"].values():
            for trace in traces:
                ev = trace.get("evaluation") or {}
                perf = ev.get("performance") or {}
                corr = ev.get("correctness") or {}
                kept_traces.append(
                    {
                        "definition": trace["definition"],
                        "solution": trace.get("solution", ""),
                        "axes": sorted((trace["workload"] or {}).get("axes", {}).items()),
                        "status": ev.get("status"),
                        "max_abs_error": corr.get("max_absolute_error"),
                        "max_rel_error": corr.get("max_relative_error"),
                        "speedup": perf.get("speedup_factor"),
                    }
                )

        # Build per-definition config dict for digest
        default_cfg = config_registry.default
        per_def_cfg_dict: Dict[str, Dict[str, Any]] = {}
        for def_name in trace_set_dict["definitions"]:
            cfg = config_registry.per_definition.get(def_name)
            if cfg is not None:
                per_def_cfg_dict[def_name] = {"max_atol": cfg.max_atol, "max_rtol": cfg.max_rtol}

        payload = {
            "cfg": {
                "default": (
                    {"max_atol": default_cfg.max_atol, "max_rtol": default_cfg.max_rtol}
                    if default_cfg
                    else None
                ),
                "per_def": per_def_cfg_dict,
            },
            "definitions": trace_set_dict["definitions"],
            "solutions": trace_set_dict["solutions"],
            "traces": sorted(
                kept_traces,
                key=lambda x: (
                    x["definition"],
                    x["solution"],
                    x["axes"],
                    x["status"] or "",
                    x["speedup"] or 0.0,
                ),
            ),
        }
        str_repr = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(str_repr).hexdigest()

    def match_solution(self, def_name: str, key: ApplyKey) -> Optional[str]:
        """Find the optimal solution for a given definition and workload key.

        This method looks up the best solution for a specific kernel definition
        and workload characteristics combination.

        Parameters
        ----------
        def_name : str
            Name of the kernel definition.
        key : ApplyKey
            Workload characteristics key to match against.

        Returns
        -------
        Optional[str]
            The name of the optimal solution, or None if no match is found.
        """
        return self.index.get(def_name, {}).get(key)
