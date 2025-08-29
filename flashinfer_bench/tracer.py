import atexit
import json
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Set, Tuple, Union

import safetensors.torch
import torch

from flashinfer_bench.definition import Definition

# Global singleton tracer instance
_current_tracer: Optional["Tracer"] = None
_tracer_lock = threading.Lock()


@dataclass
class TracingRule:
    """Defines how to collect and deduplicate workloads for a definition."""

    tensors_to_dump: Union[List[str], Callable[[Dict[str, Any]], List[str]]]
    """Which inputs to persist. List[str] for static selection, Callable for dynamic."""

    dedup_policy: Callable[[List["TraceEntry"]], List["TraceEntry"]]
    """Final in-group deduplication decision. Returns True if duplicate."""

    dedup_keys: Optional[Callable[["TraceEntry"], Hashable]] = None
    """Blocking function for candidate partitioning during dedup."""


@dataclass
class TracingConfig:
    """Global configuration for tracing."""

    out_dir: Path
    """Output directory for workload JSONL files."""

    blob_dir: Path
    """Root directory for safetensors storage."""

    rules: Dict[str, TracingRule]
    """Mapping from definition name to tracing rule."""

    dataset_path: Optional[str] = None
    """Optional dataset path for tracing-only mode (when FIB_DATASET_PATH not set)."""

    strict: bool = True
    """Whether to strictly validate rules at enable-time."""

    def __post_init__(self):
        """Ensure paths are Path objects and create directories."""
        self.out_dir = Path(self.out_dir)
        self.blob_dir = Path(self.blob_dir)

        # Create directories
        (self.out_dir / "traces/workloads").mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TraceEntry:
    """In-memory buffer entry for collected workloads."""

    def_name: str
    axes: Dict[str, int]
    definition_input_names: Set[str]
    picked: Dict[str, torch.Tensor]
    order: int
    cuda_graph_snapshot: Optional[Dict[str, torch.Tensor]] = None


class Tracer:
    """Process-wide singleton tracer for workload collection."""

    def __init__(self, config: TracingConfig):
        """
        Initialize the tracer.

        Args:
            config: Tracing configuration
            verbose: Enable verbose logging
        """
        self.config = config

        # In-memory buffer
        self.entries: List[TraceEntry] = []
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()
        self._flushed = False

        # CUDA Graph support
        self._cuda_graph_entries: List[TraceEntry] = []
        self._in_cuda_graph = False

        # Validate configuration at enable-time
        self._validate_config()

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        print(f"[Tracer] Initialized")
        print(f"  Output dir: {config.out_dir}")
        print(f"  Blob dir: {config.blob_dir}")
        print(f"  Rules: {len(config.rules)} definitions configured")

    def _validate_config(self):
        """Validate configuration at enable-time."""
        from flashinfer_bench.apply import _runtime

        # If dataset_path is provided in config, temporarily set it
        original_root = _runtime.root
        if self.config.dataset_path:
            _runtime.root = self.config.dataset_path

        try:
            # Ensure traceset is loaded
            _runtime._ensure_traceset()

            if not _runtime.traceset:
                if self.config.strict:
                    raise ValueError(
                        "TraceSet not available. Set FIB_DATASET_PATH environment variable "
                        "or provide dataset_path in TracingConfig."
                    )
                else:
                    if self.verbose:
                        print("[Tracer] TraceSet not available, validation skipped")
                    return

            traceset = _runtime.traceset

            # Validate rule keys exist in definitions
            for def_name in self.config.rules:
                if def_name not in traceset.definitions:
                    if self.config.strict:
                        raise ValueError(f"Rule for unknown definition: {def_name}")
                    elif self.verbose:
                        print(f"[Tracer] Warning: Rule for unknown definition: {def_name}")

            # Validate static input names
            for def_name, rule in self.config.rules.items():
                if def_name in traceset.definitions and isinstance(rule.tensors_to_dump, list):
                    definition = traceset.definitions[def_name]
                    for input_name in rule.tensors_to_dump:
                        if input_name not in definition.inputs:
                            if self.config.strict:
                                raise ValueError(
                                    f"Rule for {def_name} references unknown input: {input_name}"
                                )
                            elif self.verbose:
                                print(
                                    f"[Tracer] Warning: Rule for {def_name} references unknown input: {input_name}"
                                )

        finally:
            # Restore original root if we changed it
            if self.config.dataset_path:
                _runtime.root = original_root

    def collect(
        self,
        def_name: str,
        runtime_args: Dict[str, Any],
    ):
        """
        Record a workload.

        Args:
            def_name: Definition name
            runtime_args: Runtime arguments from sig.bind_partial
        """
        rule = self.config.rules.get(def_name)
        if rule is None:
            print(f"[Tracer] Tracing rule not configured for {def_name}, skipping")
            return

        from flashinfer_bench.apply import _runtime

        _runtime._ensure_traceset()

        if def_name not in _runtime.traceset.definitions:
            print(f"[Tracer] Definition {def_name} not found")
            return

        try:
            axes = _runtime.infer_axes(def_name, runtime_args)
        except ValueError as e:
            print(f"[Tracer] Error inferring axes for {def_name}: {e}")
            return

        # Validate runtime arguments
        definition = _runtime.traceset.definitions[def_name]
        definition_input_names = set(definition.inputs.keys())
        runtime_input_names = set(runtime_args.keys())

        missing = sorted(definition_input_names - runtime_input_names)
        unexpected = sorted(runtime_input_names - definition_input_names)

        if missing:
            print(f"[Tracer] Missing inputs for {def_name}: {missing}")
            return

        if unexpected:
            print(f"[Tracer] Unexpected inputs for {def_name}: {unexpected}")
            return

        # At this point, runtime_args exactly matches definition.inputs

        # Validate tensors_to_dump
        if isinstance(rule.tensors_to_dump, list):
            names_to_dump = rule.tensors_to_dump
        else:
            names_to_dump = rule.tensors_to_dump(runtime_args) or []

        if not isinstance(names_to_dump, list) or not all(
            isinstance(name, str) for name in names_to_dump
        ):
            print(f"[Tracer] tensors_to_dump must return List[str], got {names_to_dump}")
            return

        # Names specified to dump but not in definition
        unknown = [n for n in names_to_dump if n not in definition_input_names]
        if unknown:
            if self.verbose:
                print(
                    f"[Tracer] Invalid tensors_to_dump for {def_name}: unknown={unknown} (expected one of {sorted(definition_input_names)})"
                )
            return

        picked: Dict[str, torch.Tensor] = {}

        for name in names_to_dump:
            spec = definition.inputs[name]

            try:
                shape_spec = _materialize_shape(definition, axes, spec.get("shape"))
                dtype_spec = _torch_dtype_from_def(spec.get("dtype"))
            except ValueError as e:
                print(f"[Tracer] Error materializing specs for {name}: {e}")
                return

            val = runtime_args.get(name)

            if val is None:
                print(
                    f"[Tracer] Required input '{name}' is None (missing or optional) for {def_name}"
                )
                return

            # 0-D tensor (scalar)
            if shape_spec == ():
                if isinstance(val, torch.Tensor):
                    if val.ndim != 0:
                        print(
                            f"[Tracer] Input '{name}' expects 0-D tensor, got shape {tuple(val.shape)}"
                        )
                        return
                    if val.dtype != dtype_spec:
                        print(
                            f"[Tracer] Input '{name}' must have dtype {dtype_spec}, got {val.dtype}"
                        )
                        return
                elif isinstance(val, (int, float, bool)):
                    val = torch.tensor(val, dtype=dtype_spec)
                else:
                    print(
                        f"[Tracer] Input '{name}' must be a 0-D tensor or Python scalar, got {type(val).__name__}"
                    )
                    return
            # Non-scalar input
            else:
                if not isinstance(val, torch.Tensor):
                    print(f"[Tracer] Input '{name}' must be a tensor (got {type(val).__name__})")
                    return
                if val.shape != shape_spec:
                    print(f"[Tracer] Input '{name}' must have shape {shape_spec}, got {val.shape}")
                    return
                if val.dtype != dtype_spec:
                    print(f"[Tracer] Input '{name}' must have dtype {dtype_spec}, got {val.dtype}")
                    return

            if not self._in_cuda_graph:
                val = val.detach().cpu().clone()
            picked[name] = val

        entry = TraceEntry(
            def_name=def_name,
            axes=axes,
            definition_input_names=definition_input_names,
            picked=picked,
            order=self.order_counter,
        )

        with self._lock:
            if self._in_cuda_graph:
                # Deferred snapshot
                self._cuda_graph_entries.append(entry)
            else:
                self.entries.append(entry)

            self.order_counter += 1

            if len(self.entries) % 100 == 0:
                print(f"[Tracer] Buffered {len(self.entries)} entries")

    def cuda_graph_scope(self):
        """Context manager for CUDA Graph collection."""

        class CudaGraphScope:
            def __init__(self, tracer):
                self.tracer = tracer

            def __enter__(self):
                self.tracer._in_cuda_graph = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.tracer._in_cuda_graph = False
                self.tracer.snapshot_graph_tensors()

        return CudaGraphScope(self)

    def snapshot_graph_tensors(self):
        """Snapshot tensors from CUDA Graph entries."""
        # Synchronize CUDA before taking snapshots
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for entry in self._cuda_graph_entries:
            # Create CPU snapshots
            snapshot = {}
            for name, tensor in entry.picked.items():
                if isinstance(tensor, torch.Tensor):
                    snapshot[name] = tensor.detach().cpu().clone()
                else:
                    snapshot[name] = tensor
            entry.cuda_graph_snapshot = snapshot
            entry.picked = snapshot
            self.entries.append(entry)

        self._cuda_graph_entries.clear()

    def flush(self) -> Dict[str, Any]:
        """
        Deduplicate and write collected workloads to disk.

        Returns:
            Statistics about the flush operation
        """
        with self._lock:
            if not self.entries:
                return {
                "total_entries": 0,
                "groups": {},
                "representatives": 0,
                "dedup_errors": self.dedup_errors,
                "files_written": 0,
                }
            batch = self.entries
            self.entries = []
        per_def: Dict[str, List[TraceEntry]] = {}
        for e in batch:
            per_def.setdefault(e.def_name, []).append(e)

        files_written: Set[str] = set()
        per_def_stats: Dict[str, Dict[str, int]] = {}
        total_reps = 0
        dedup_errors = 0

        for def_name, entries in per_def.items():
            rule = self.config.rules.get(def_name)
            if rule is None:
                continue

            # Bucketing by dedup_keys
            buckets: Dict[Hashable, List[TraceEntry]] = {}
            if rule.dedup_keys:
                for e in entries:
                    try:
                        key = rule.dedup_keys(e)
                    except Exception as err:
                        print(f"[Tracer] dedup_keys error for {def_name}:{e} because of {err}")
                        key = ("__err__")
                        dedup_errors += 1
                    if key is None:
                        print(f"[Tracer] dedup_keys returned None for {def_name}:{e}")
                        key = ("__none__")
                    buckets.setdefault(key, []).append(e)
            else:
                # All in the same bucket
                buckets[("__all__",)].extend(entries)

            # Inside each bucket, run dedup_policy to pick representatives
            reps: List[TraceEntry] = []
            for bucket_entries in buckets.values():
                try:
                    reps_in_bucket = rule.dedup_policy(bucket_entries)
                except Exception as err:
                    print(f"[Tracer] dedup_policy error for {def_name} because of {err}, keeping all entries")
                    reps_in_bucket = bucket_entries
                    dedup_errors += 1
                reps.extend(reps_in_bucket)

            output_path = self.config.out_dir / "workloads" / f"{def_name}.workload.jsonl"
            self._write_representatives(def_name, reps, output_path)
            files_written.add(def_name)

            st = per_def_stats.setdefault(def_name, {"total_entries": 0, "buckets": 0, "representatives": 0})
            st["total_entries"] += len(entries)
            st["buckets"] += len(buckets)
            st["representatives"] += len(reps)
            total_reps += len(reps)

        return {
            "total_entries": len(batch),
            "groups": per_def_stats,
            "representatives": total_reps,
            "dedup_errors": dedup_errors,
            "files_written": len(files_written),
        }

    def _write_representatives(self, def_name: str, reps: List[TraceEntry], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            for entry in reps:
                workload_uuid = str(uuid.uuid4())
                input_specs: Dict[str, Any] = {}
                if entry.picked:
                    safepath = self._save_tensors(def_name, workload_uuid, entry.picked)
                    for name in entry.picked:
                        input_specs[name] = {
                            "type": "safetensors",
                            "path": str(safepath.relative_to(self.config.blob_dir)),
                            "tensor_key": name,
                        }
                # backfill random for non-dumped inputs
                for name in entry.definition_input_names:
                    if name not in input_specs:
                        input_specs[name] = {"type": "random"}

                record = {
                    "definition": def_name,
                    "solution": "",
                    "workload": {
                        "uuid": workload_uuid,
                        "axes": entry.axes,
                        "inputs": input_specs,
                    },
                    "evaluation": {},
                }
                json.dump(record, f)
                f.write("\n")

    def _save_tensors(self, def_name: str, workload_uuid: str, tensors: Dict[str, torch.Tensor]) -> Path:
        def_dir = self.config.blob_dir / def_name
        def_dir.mkdir(parents=True, exist_ok=True)
        path = def_dir / f"{def_name}_{workload_uuid}.safetensors"
        cpu_tensors = {k: (v.cpu() if v.is_cuda else v) for k, v in tensors.items()}
        safetensors.torch.save_file(cpu_tensors, path)
        return path

    def _cleanup(self):
        """Cleanup handler for atexit."""
        try:
            self.flush()
        except:
            pass

    def _signal_handler(self, signum, frame):
        """Signal handler for SIGTERM/SIGINT."""
        try:
            self.flush()
        except:
            pass


# ============================================================================
# Public API
# ============================================================================


def enable_tracing(config: TracingConfig) -> Tracer:
    """
    Enable tracing with the given configuration.

    Creates or replaces the process-wide singleton tracer.
    If replacing, flushes the previous instance first.

    Args:
        config: Tracing configuration

    Returns:
        The new tracer instance
    """
    global _current_tracer

    with _tracer_lock:
        # Flush previous tracer if exists
        if _current_tracer is not None:
            try:
                _current_tracer.flush()
            except:
                pass

        # Create new tracer
        _current_tracer = Tracer(config)

        # TODO: Register @apply reporting hook here
        # This would integrate with the apply decorator

        return _current_tracer


def get_tracer() -> Optional[Tracer]:
    """Get the current tracer instance."""
    return _current_tracer


def disable_tracing():
    """Disable tracing and flush any pending data."""
    global _current_tracer

    with _tracer_lock:
        if _current_tracer is not None:
            try:
                _current_tracer.flush()
            except:
                pass
            _current_tracer = None


# ============================================================================
# Utility Functions
# ============================================================================


def _torch_dtype_from_def(def_dtype: str):
    if not def_dtype:
        raise ValueError("dtype is None or empty")
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }
    dtype = table.get(def_dtype.lower(), None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{def_dtype}'")
    return dtype


def _axis_value(definition, axes: Dict[str, Any], axis_name: str) -> int:
    if axis_name in axes:
        return int(axes[axis_name])
    axis_spec = definition.axes.get(axis_name)
    if axis_spec is None:
        raise ValueError(f"Unknown axis '{axis_name}' in shape")
    if axis_spec.get("type") == "const":
        val = int(axis_spec["value"])
        if val is None:
            raise ValueError(f"Const axis '{axis_name}' missing value/size")
        return val
    if axis_spec.get("type") == "var":
        raise ValueError(f"Axis '{axis_name}' is var but missing in axes")
    raise ValueError(f"Unsupported axis type for '{axis_name}': {axis_spec.get('type')}")


def _materialize_shape(definition: Definition, axes: Dict[str, Any], shape_spec) -> Tuple[int, ...]:
    if not shape_spec:
        raise ValueError(f"Input shape specification is None")
    if shape_spec == []:
        return ()
    dims: List[int] = []
    for dim in shape_spec:
        if isinstance(dim, str):
            dims.append(_axis_value(definition, axes, dim))
        else:
            raise ValueError(f"Unsupported shape {dim}")
    return tuple(dims)
