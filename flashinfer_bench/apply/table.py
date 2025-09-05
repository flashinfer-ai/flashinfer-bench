from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flashinfer_bench.compile.registry import get_registry
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.json_codec import dataclass_to_dict
from flashinfer_bench.data.trace import Trace
from flashinfer_bench.data.traceset import TraceSet

from .config import ApplyConfig
from .key import ApplyKey, ApplyKeyFactory


def _cache_root() -> Path:
    base = os.environ.get("FIB_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "flashinfer_bench"


def _apply_table_dir() -> Path:
    return _cache_root() / "apply_table"


@dataclass
class ApplyTable:
    digest: str
    # def_name -> (key -> solution_name)
    index: Dict[str, Dict[ApplyKey, str]] = field(default_factory=dict)
    # def_name -> best solution_name
    def_best: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load_or_build(cls, ts: TraceSet, config: ApplyConfig) -> "ApplyTable":
        digest = cls._digest(ts, config)

        apply_dir = _apply_table_dir()
        apply_dir.mkdir(parents=True, exist_ok=True)
        index_path = apply_dir / f"{digest}.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                raw = json.load(f)

            index: Dict[str, Dict[ApplyKey, str]] = {}
            for def_name, items in raw.get("index", {}).items():
                bucket: Dict[ApplyKey, str] = {}
                for key_enc, sol_name in items.items():
                    key = ApplyKey.from_encoded(key_enc)
                    bucket[key] = sol_name
                index[def_name] = bucket

            def_best: Dict[str, str] = {}
            reg = get_registry()

            for def_name, sol_name in raw["def_best"].items():
                defn = ts.definitions.get(def_name)
                sol = ts.get_solution(sol_name)
                if defn and sol:
                    reg.build(defn, sol)
                    def_best[def_name] = sol_name

            table = cls(digest=digest, index=index, def_best=def_best)

            if config.aot_ratio and config.aot_ratio > 0.0:
                cls._prewarm_aot(ts, config, table)

            return table

        # Build fresh
        table = cls._build(ts, config)
        # Persist minimal index
        to_dump: Dict[str, Any] = {"digest": table.digest, "index": {}, "def_best": {}}
        for def_name, bucket in table.index.items():
            for key, sol_name in bucket.items():
                to_dump["index"].setdefault(def_name, {})[key.encode] = sol_name
        # Always compute and persist def_best
        for def_name, sol_name in table.def_best.items():
            to_dump["def_best"][def_name] = sol_name

        with open(index_path, "w") as f:
            json.dump(to_dump, f)

        if config.aot_ratio and config.aot_ratio > 0.0:
            cls._prewarm_aot(ts, config, table)

        return table

    @classmethod
    def _build(cls, ts: TraceSet, config: ApplyConfig) -> "ApplyTable":
        digest = cls._digest(ts, config)
        reg = get_registry()

        index: Dict[str, Dict[ApplyKey, str]] = {}
        def_best: Dict[str, Runnable] = {}

        for def_name, defn in ts.definitions.items():
            per_key, ranked = cls._sweep_def(ts, def_name, config.max_atol, config.max_rtol)

            # Build index
            for key, t in per_key.items():
                if not t.solution:
                    continue
                bucket = index.setdefault(def_name, {})
                bucket[key] = t.solution

            # Build def_best
            if ranked:
                best_sol_name = ranked[0][0]
                sol = ts.get_solution(best_sol_name)
                if sol:
                    if config.on_miss_policy == "use_def_best":
                        # Only AOT if on_miss_policy is use_def_best
                        reg.build(defn, sol)
                    def_best[def_name] = best_sol_name

        return cls(digest=digest, index=index, def_best=def_best)

    @classmethod
    def _sweep_def(
        cls,
        ts: TraceSet,
        def_name: str,
        max_atol: float,
        max_rtol: float,
    ) -> Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]]]:
        traces = ts.filter_traces(def_name, max_atol, max_rtol)
        builder = ApplyKeyFactory.specialize(ts.definitions[def_name])

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
    def _prewarm_aot(cls, ts: TraceSet, config: ApplyConfig, table: "ApplyTable") -> None:
        if not (config.aot_ratio and config.aot_ratio > 0.0):
            return
        reg = get_registry()

        for def_name, bucket in table.index.items():
            if not bucket:
                continue

            win_counts = Counter(bucket.values())
            ranked = sorted(win_counts.items(), key=lambda kv: kv[1], reverse=True)
            cutoff = max(1, int(len(ranked) * config.aot_ratio))

            defn = ts.definitions.get(def_name)
            if not defn:
                continue
            for sol_name, _ in ranked[:cutoff]:
                sol = ts.get_solution(sol_name)
                if sol:
                    reg.build(defn, sol)

        if config.on_miss_policy == "use_def_best":
            for def_name, sol_name in table.def_best.items():
                defn = ts.definitions.get(def_name)
                sol_name = sol_name.meta.get("solution")
                if sol_name:
                    reg.build(defn, sol)

    @classmethod
    def _digest(cls, ts: TraceSet, config: ApplyConfig) -> str:
        d = dataclass_to_dict(ts)
        for defn in d["definitions"].values():
            for drop in ("description", "tags", "reference", "constraints"):
                defn.pop(drop, None)
        for sol_list in d["solutions"].values():
            for sol in sol_list:
                spec = sol.get("spec", {}) or {}
                deps = spec.get("dependencies") or []
                spec["dependencies"] = sorted(deps)
                new_sources = []
                for sf in sol.get("sources") or []:
                    new_sources.append(
                        {
                            "path": sf["path"],
                            "sha1": hashlib.sha1(sf["content"].encode("utf-8")).hexdigest(),
                        }
                    )
                sol["sources"] = new_sources
        kept_traces = []
        for traces in d["traces"].values():
            for t in traces:
                ev = t.get("evaluation") or {}
                perf = ev.get("performance") or {}
                corr = ev.get("correctness") or {}
                kept_traces.append(
                    {
                        "definition": t["definition"],
                        "solution": t.get("solution", ""),
                        "axes": sorted((t["workload"] or {}).get("axes", {}).items()),
                        "status": ev.get("status"),
                        "max_abs_error": corr.get("max_absolute_error"),
                        "max_rel_error": corr.get("max_relative_error"),
                        "speedup": perf.get("speedup_factor"),
                    }
                )
        payload = {
            "cfg": {"max_atol": config.max_atol, "max_rtol": config.max_rtol},
            "definitions": d["definitions"],
            "solutions": d["solutions"],
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
        return self.index.get(def_name, {}).get(key)
