import atexit
import json
import signal
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import safetensors.torch
import torch

from flashinfer_bench.data.definition import Definition
from flashinfer_bench.logging import get_logger

from .types import TraceEntry, TracingRule

# Global singleton tracer instance
_current_tracer: Optional["Tracer"] = None
_tracer_lock = threading.Lock()


class Tracer:
    """Process-wide singleton tracer for workload collection."""

    def __init__(
        self,
        rules: Dict[str, TracingRule],
        out_dir: Optional[Path] = None,
        blob_dir: Optional[Path] = None,
    ):
        """
        Initialize the tracer.

        Args:
            rules: A set of tracing rules
            out_dir: Output directory for traces. Default is FIB_DATASET_PATH/traces/workloads
            blob_dir: Blob directory for safetensors. Default is FIB_DATASET_PATH/blob/workloads
        """
        self.rules = rules
        self.out_dir = out_dir
        self.blob_dir = blob_dir

        # In-memory buffer
        self.entries: List[TraceEntry] = []
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()
        self._flushed = False

        # CUDA Graph support
        self._cuda_graph_entries: List[TraceEntry] = []
        self._in_cuda_graph = False

        self._logger = get_logger("Tracer")

        # Validate configuration at enable-time
        self._validate()

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _validate(self):
        """Validate tracer configuration at enable-time."""
        from flashinfer_bench.apply.runtime import get_runtime

        rt = get_runtime()

        if not rt.traceset:
            raise ValueError("Dataset not available. Set FIB_DATASET_PATH environment variable.")

        if self.out_dir is None and rt.root is not None:
            self.out_dir = Path(rt.root) / "traces" / "workloads"
        if self.blob_dir is None and rt.root is not None:
            self.blob_dir = Path(rt.root) / "blob" / "workloads"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

        # Validate rule keys exist in definitions
        for def_name in self.rules:
            if rt.traceset and def_name not in rt.traceset.definitions:
                self._logger.warning(f"Rule found for unknown definition: {def_name}")

        self._logger.info("Tracer Initialized")
        self._logger.info(f"  Output dir: {self.out_dir} / Blob dir: {self.blob_dir}")
        self._logger.info(f"  Rules: {len(self.rules)} definitions configured")

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
        self._logger.info(f"Tracing '{def_name}'")
        rule = self.rules.get(def_name)
        if rule is None:
            self._logger.error(f"Tracing rule not configured for {def_name}, skipping")
            return

        from flashinfer_bench.apply.runtime import get_runtime

        rt = get_runtime()
        if not rt.traceset or def_name not in rt.traceset.definitions:
            self._logger.error(f"Definition {def_name} not found")
            return

        try:
            axes = rt.infer_axes(def_name, runtime_args)
        except ValueError as e:
            self._logger.error(f"Error inferring axes for {def_name}: {e}")
            return

        # Validate runtime arguments
        definition = rt.traceset.definitions[def_name]
        definition_input_names = set(definition.inputs.keys())
        runtime_input_names = set(runtime_args.keys())

        missing = sorted(definition_input_names - runtime_input_names)
        unexpected = sorted(runtime_input_names - definition_input_names)

        if missing:
            self._logger.error(f"Missing inputs for {def_name}: {missing}")
            return

        if unexpected:
            self._logger.error(f"Unexpected inputs for {def_name}: {unexpected}")
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
            self._logger.error(f"tensors_to_dump must return List[str], got {names_to_dump}")
            return

        # Names specified to dump but not in definition
        unknown = [n for n in names_to_dump if n not in definition_input_names]
        if unknown:
            self._logger.error(
                f"Invalid tensors_to_dump for {def_name}: unknown={unknown} (expected one of {sorted(definition_input_names)})"
            )
            return

        picked: Dict[str, torch.Tensor] = {}

        for name in names_to_dump:
            spec = definition.inputs[name]

            try:
                shape_spec = _materialize_shape(definition, axes, spec.get("shape"))
                dtype_spec = _torch_dtype_from_def(spec.get("dtype"))
            except ValueError as e:
                self._logger.error(f"Error materializing specs for {name}: {e}")
                return

            val = runtime_args.get(name)

            if val is None:
                self._logger.error(
                    f"Required input '{name}' is None (missing or optional) for {def_name}"
                )
                return

            # 0-D tensor (scalar)
            if shape_spec == ():
                if isinstance(val, torch.Tensor):
                    if val.ndim != 0:
                        self._logger.error(
                            f"Input '{name}' expects 0-D tensor, got shape {tuple(val.shape)}"
                        )
                        return
                    if val.dtype != dtype_spec:
                        self._logger.error(
                            f"Input '{name}' must have dtype {dtype_spec}, got {val.dtype}"
                        )
                        return
                elif isinstance(val, (int, float, bool)):
                    val = torch.tensor(val, dtype=dtype_spec)
                else:
                    self._logger.error(
                        f"Input '{name}' must be a 0-D tensor or Python scalar, got {type(val).__name__}"
                    )
                    return
            # Non-scalar input
            else:
                if not isinstance(val, torch.Tensor):
                    self._logger.error(
                        f"Input '{name}' must be a tensor (got {type(val).__name__})"
                    )
                    return
                if val.shape != shape_spec:
                    self._logger.error(
                        f"Input '{name}' must have shape {shape_spec}, got {val.shape}"
                    )
                    return
                if val.dtype != dtype_spec:
                    self._logger.error(
                        f"Input '{name}' must have dtype {dtype_spec}, got {val.dtype}"
                    )
                    return

            if not self._in_cuda_graph:
                val = val.detach().cpu().clone()
            picked[name] = val

        with self._lock:
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

            if len(self.entries) % 100 == 0:
                self._logger.info(f"[Tracer] Buffered {len(self.entries)} entries")

    # TODO(shanli): fix cuda graph tracing
    def cuda_graph_scope(self):
        """Context manager for CUDA Graph collection."""

        class CudaGraphScope:
            def __init__(self, tracer):
                self.tracer = tracer

            def __enter__(self):
                with self.tracer._lock:
                    self.tracer._in_cuda_graph = True
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                with self.tracer._lock:
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
        if self._in_cuda_graph:
            raise RuntimeError("Cannot flush during CUDA Graph replay")
        with self._lock:
            if not self.entries:
                return {
                    "total_entries": 0,
                    "groups": {},
                    "representatives": 0,
                    "dedup_errors": 0,
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
            rule = self.rules.get(def_name)
            if rule is None:
                continue

            # Bucketing by dedup_keys
            buckets: Dict[Hashable, List[TraceEntry]] = {}
            if rule.dedup_keys:
                for e in entries:
                    try:
                        key = rule.dedup_keys(e)
                    except Exception as err:
                        self._logger.warning(
                            f"dedup_keys error for {def_name}:{e} because of {err}"
                        )
                        key = "__err__"
                        dedup_errors += 1
                    if key is None:
                        self._logger.warning(f"dedup_keys returned None for {def_name}:{e}")
                        key = "__none__"
                    buckets.setdefault(key, []).append(e)
            else:
                # All in the same bucket
                buckets.setdefault("__all__", []).extend(entries)

            # Inside each bucket, run dedup_policy to pick representatives
            reps: List[TraceEntry] = []
            for bucket_entries in buckets.values():
                try:
                    reps_in_bucket = rule.dedup_policy(bucket_entries)
                except Exception as err:
                    self._logger.warning(
                        f"dedup_policy error for {def_name} because of {err}, keeping all entries"
                    )
                    reps_in_bucket = bucket_entries
                    dedup_errors += 1
                reps.extend(reps_in_bucket)

            output_path = self.out_dir / f"{def_name}.workload.jsonl"
            self._write_representatives(def_name, reps, output_path)
            files_written.add(def_name)

            st = per_def_stats.setdefault(
                def_name, {"total_entries": 0, "buckets": 0, "representatives": 0}
            )
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
                    try:
                        safepath = self._save_tensors(def_name, workload_uuid, entry.picked)
                    except Exception as e:
                        self._logger.error(f"Failed to save tensors for {def_name}: {e}")
                        continue
                    for name in entry.picked:
                        input_specs[name] = {
                            "type": "safetensors",
                            "path": str(safepath.relative_to(self.blob_dir)),
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

    def _save_tensors(
        self, def_name: str, workload_uuid: str, tensors: Dict[str, torch.Tensor]
    ) -> Path:
        def_dir = self.blob_dir / def_name
        def_dir.mkdir(parents=True, exist_ok=True)
        path = def_dir / f"{def_name}_{workload_uuid}.safetensors"
        cpu_tensors = {k: (v.cpu() if v.is_cuda else v) for k, v in tensors.items()}
        safetensors.torch.save_file(cpu_tensors, path)
        return path

    def _cleanup(self):
        """Cleanup handler for atexit."""
        try:
            self.flush()
        except Exception as e:
            self._logger.error(f"Flush failed: {e}")

    def _signal_handler(self, signum, frame):
        """Signal handler for SIGTERM/SIGINT."""
        try:
            self.flush()
        except Exception as e:
            self._logger.error(f"Flush failed: {e}")


# ============================================================================
# Public API
# ============================================================================


def enable_tracing(
    rules: Optional[Dict[str, TracingRule]] = None,
    out_dir: Optional[Path] = None,
    blob_dir: Optional[Path] = None,
) -> Tracer:
    """
    Enable tracing with the given tracing rule set.

    Creates or replaces the process-wide singleton tracer.
    If replacing, flushes the previous instance first.

    Args:
        rules: A set of tracing rules. Default is `tracing_rules.fib_full_tracing`
        out_dir: Output directory for traces. Default is FIB_DATASET_PATH/traces/workloads
        blob_dir: Blob directory for safetensors. Default is FIB_DATASET_PATH/blob/workloads

    Returns:
        The new tracer instance
    """
    global _current_tracer

    with _tracer_lock:
        # Flush previous tracer if exists
        if _current_tracer is not None:
            try:
                _current_tracer.flush()
            except Exception as e:
                _current_tracer._logger.error(f"Cannot flush existing tracer: {e}, overriding")
                _current_tracer = None

        # If no rules are specified, we do full tracing with preset rules.
        if rules is None:
            from .rule import fib_full_tracing

            rules = fib_full_tracing

        _current_tracer = Tracer(rules, out_dir=out_dir, blob_dir=blob_dir)

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
            except Exception as e:
                _current_tracer._logger.error(f"Flush failed: {e}")
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
        val = axis_spec.get("value")
        if val is None:
            raise ValueError(f"Const axis '{axis_name}' missing value/size")
        return int(val)
    if axis_spec.get("type") == "var":
        raise ValueError(f"Axis '{axis_name}' is var but missing in axes")
    raise ValueError(f"Unsupported axis type for '{axis_name}': {axis_spec.get('type')}")


def _materialize_shape(definition: Definition, axes: Dict[str, Any], shape_spec) -> Tuple[int, ...]:
    if not shape_spec:
        raise ValueError("Input shape specification is None")
    if shape_spec == []:
        return ()
    dims: List[int] = []
    for dim in shape_spec:
        if isinstance(dim, str):
            dims.append(_axis_value(definition, axes, dim))
        else:
            raise ValueError(f"Unsupported shape {dim}")
    return tuple(dims)
