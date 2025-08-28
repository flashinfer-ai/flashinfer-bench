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

    dedup_policy: Callable[
        [str, Dict[str, Any], Dict[str, torch.Tensor], Dict[str, torch.Tensor]], bool
    ]
    """Final in-group deduplication decision. Returns True if duplicate."""

    dedup_keys: Optional[Callable[[str, Dict[str, Any], Dict[str, torch.Tensor]], Hashable]] = None
    """Optional blocking function for candidate partitioning during dedup."""


@dataclass
class TracingConfig:
    """Global configuration for tracing."""

    out_dir: Path
    """Output directory for workload JSONL files."""

    blob_dir: Path
    """Root directory for safetensors storage."""

    rules: Dict[str, TracingRule]
    """Mapping from definition name to tracing rule."""

    default: Optional[TracingRule] = None
    """Fallback rule for definitions not in rules."""

    dataset_path: Optional[str] = None
    """Optional dataset path for tracing-only mode (when FIB_DATASET_PATH not set)."""

    strict: bool = True
    """Whether to strictly validate rules at enable-time."""

    def __post_init__(self):
        """Ensure paths are Path objects and create directories."""
        self.out_dir = Path(self.out_dir)
        self.blob_dir = Path(self.blob_dir)

        # Create directories
        (self.out_dir / "workloads").mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TraceEntry:
    """In-memory buffer entry for collected workloads."""

    def_name: str
    axes: Dict[str, Any]
    definition_input_names: Set[str]
    picked: Dict[str, torch.Tensor]
    order: int
    cuda_graph_snapshot: Optional[Dict[str, torch.Tensor]] = None


class Tracer:
    """Process-wide singleton tracer for workload collection."""

    def __init__(self, config: TracingConfig, verbose: bool = False):
        """
        Initialize the tracer.

        Args:
            config: Tracing configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose

        # In-memory buffer
        self.entries: List[TraceEntry] = []
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()
        self._flushed = False

        # CUDA Graph support
        self._cuda_graph_entries: List[TraceEntry] = []
        self._in_cuda_graph = False

        # Statistics
        self.dedup_errors = 0

        # Validate configuration at enable-time
        self._validate_config()

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        if verbose:
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
        rule = self.config.rules.get(def_name, self.config.default)
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
                picked[name] = val.detach().cpu().clone()

            entry = TraceEntry(
                def_name=def_name,
                axes=axes,
                definition_input_names=definition_input_names,
                picked=picked,
                order=self.order_counter,
            )

            if self._in_cuda_graph:
                # Deferred snapshot
                self._cuda_graph_entries.append(entry)
            else:
                self.entries.append(entry)

            self.order_counter += 1

            if self.verbose and len(self.entries) % 100 == 0:
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
        if self._flushed or not self.entries:
            if self.verbose:
                print("[Tracer] No entries to flush or already flushed")
            return {
                "total_entries": 0,
                "groups": {},
                "representatives": 0,
                "dedup_errors": 0,
                "files_written": 0,
            }

        # Group entries
        groups = self._group_entries(self.entries)

        # Deduplicate and write
        total_representatives = 0
        files_written = set()
        group_stats = {}

        for (def_name, axes_key), group_entries in groups.items():
            # Run deduplication
            representatives = self._deduplicate_group(def_name, group_entries)
            total_representatives += len(representatives)

            # Write to disk
            output_path = self.config.out_dir / "workloads" / f"{def_name}.workload.jsonl"
            self._write_representatives(def_name, representatives, output_path)
            files_written.add(def_name)

            # Track stats
            if def_name not in group_stats:
                group_stats[def_name] = {"total_entries": 0, "axis_groups": 0, "representatives": 0}

            group_stats[def_name]["total_entries"] += len(group_entries)
            group_stats[def_name]["axis_groups"] += 1
            group_stats[def_name]["representatives"] += len(representatives)

            if self.verbose:
                print(
                    f"[Tracer] {def_name}: {len(group_entries)} entries → {len(representatives)} representatives"
                )

        # Mark as flushed and clear entries
        total_entries = len(self.entries)
        self.entries.clear()
        self._flushed = True

        stats = {
            "total_entries": total_entries,
            "groups": group_stats,
            "representatives": total_representatives,
            "dedup_errors": self.dedup_errors,
            "files_written": len(files_written),
        }

        if self.verbose:
            print(f"[Tracer] Flush complete:")
            print(f"  Total entries: {total_entries}")
            print(f"  Definitions: {len(group_stats)}")
            print(f"  Representatives: {total_representatives}")
            print(f"  Dedup errors: {self.dedup_errors}")
            print(f"  Files written: {len(files_written)}")

        # Reset for next collection cycle
        self._flushed = False
        self.dedup_errors = 0  # Reset dedup error counter

        return stats

    def _group_entries(
        self, entries: List[TraceEntry]
    ) -> Dict[Tuple[str, Tuple], List[TraceEntry]]:
        """Group entries by (def_name, axes)."""
        groups = {}
        for entry in entries:
            axes_key = tuple(sorted(entry.axes.items()))
            group_key = (entry.def_name, axes_key)

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(entry)

        return groups

    def _deduplicate_group(self, def_name: str, entries: List[TraceEntry]) -> List[TraceEntry]:
        """Deduplicate entries within a group using deterministic greedy algorithm."""
        rule = self.config.rules.get(def_name, self.config.defaults)
        if rule is None:
            return entries

        # Sort by order for deterministic processing
        sorted_entries = sorted(entries, key=lambda e: e.order)

        # Optional blocking with dedup_keys
        if rule.dedup_keys:
            blocks = {}
            for entry in sorted_entries:
                try:
                    key = rule.dedup_keys(def_name, entry.axes, entry.picked)
                    if key not in blocks:
                        blocks[key] = []
                    blocks[key].append(entry)
                except Exception as e:
                    if self.verbose:
                        print(f"[Tracer] dedup_keys error for {def_name}: {e}")
                    # Treat as unique block
                    blocks[id(entry)] = [entry]

            # Deduplicate within each block
            representatives = []
            for block_entries in blocks.values():
                block_reps = self._deduplicate_block(def_name, block_entries, rule.dedup_policy)
                representatives.extend(block_reps)

            return sorted(representatives, key=lambda e: e.order)
        else:
            # No blocking, deduplicate entire group
            return self._deduplicate_block(def_name, sorted_entries, rule.dedup_policy)

    def _deduplicate_block(
        self, def_name: str, entries: List[TraceEntry], dedup_policy: Callable
    ) -> List[TraceEntry]:
        """Deduplicate entries within a block using the dedup policy."""
        representatives = []

        for candidate in entries:
            is_duplicate = False

            for rep in representatives:
                try:
                    if dedup_policy(def_name, candidate.axes, candidate.picked, rep.picked):
                        is_duplicate = True
                        break
                except Exception as e:
                    # On exception, treat as non-duplicate
                    self.dedup_errors += 1
                    if self.verbose:
                        print(f"[Tracer] dedup_policy error for {def_name}: {e}")

            if not is_duplicate:
                representatives.append(candidate)

        return representatives

    def _write_representatives(
        self, def_name: str, representatives: List[TraceEntry], output_path: Path
    ):
        """Write representative entries to JSONL and save tensors."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a") as f:
            for entry in representatives:
                workload_uuid = str(uuid.uuid4())

                # Save tensors to safetensors
                input_specs = {}
                if entry.picked:
                    safetensor_path = self._save_tensors(def_name, workload_uuid, entry.picked)

                    for input_name in entry.picked:
                        input_specs[input_name] = {
                            "type": "safetensors",
                            "path": str(safetensor_path.relative_to(self.config.blob_dir)),
                            "tensor_key": input_name,
                        }

                # Uniform random backfill for non-dumped inputs
                for input_name in entry.definition_input_names:
                    if input_name not in input_specs:
                        input_specs[input_name] = {"type": "random"}

                # Create workload entry
                workload = {
                    "definition": def_name,
                    "solution": "",
                    "workload": {"uuid": workload_uuid, "axes": entry.axes, "inputs": input_specs},
                    "evaluation": {},
                }

                # Write as single line
                json.dump(workload, f)
                f.write("\n")

    def _save_tensors(
        self, def_name: str, workload_uuid: str, tensors: Dict[str, torch.Tensor]
    ) -> Path:
        """Save tensors to safetensors file."""
        # Create directory for definition
        def_dir = self.config.blob_dir / def_name
        def_dir.mkdir(parents=True, exist_ok=True)

        # Save to safetensors
        safetensor_path = def_dir / f"{workload_uuid}.safetensors"

        # Convert tensors to CPU if needed
        cpu_tensors = {}
        for name, tensor in tensors.items():
            cpu_tensors[name] = tensor.cpu() if tensor.is_cuda else tensor

        safetensors.torch.save_file(cpu_tensors, safetensor_path)

        return safetensor_path

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
