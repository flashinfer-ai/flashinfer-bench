import atexit
import signal
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from flashinfer_bench.data import (
    Definition,
    InputSpec,
    RandomInput,
    SafetensorsInput,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.env import get_fib_dataset_path, get_fib_enable_tracing
from flashinfer_bench.logging import get_logger
from flashinfer_bench.utils import dtype_str_to_python_dtype, dtype_str_to_torch_dtype

from .builtin_tracing_config import FULL_TRACING_CONFIGS
from .tracing_config import TracingConfig
from .tracing_policy import DedupPolicy, WorkloadEntry

logger = get_logger("TracingRuntime")


class TracingRuntime:
    """Process-wide singleton tracer for workload collection."""

    def __init__(
        self,
        trace_set: TraceSet,
        tracing_configs: Optional[Dict[str, TracingConfig]] = None,
        prev_tracing_runtime: Optional["TracingRuntime"] = None,
    ):
        """
        Initialize the tracing runtime.

        Parameters
        ----------
        dataset_path : Optional[str]
            Path to the dataset/traceset directory. Default is the environment variable
            FIB_DATASET_PATH if it exists, or `~/.cache/flashinfer_bench/dataset`.
        tracing_configs : Optional[Dict[str, TracingConfig]]
            A set of tracing configs. Default is `fib.tracing.builtin_config.fib_full_tracing`.
        prev_tracing_runtime : Optional[TracingRuntime]
            The previous tracing runtime. Will be used in the __exit__ method.
        """
        self._trace_set = trace_set

        tracing_configs_non_null = tracing_configs if tracing_configs else FULL_TRACING_CONFIGS
        self._tracing_configs = tracing_configs_non_null

        self._prev_runtime = prev_tracing_runtime

        # Global order counter
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()

        # CUDA Graph support
        self._cuda_graph_entries: List[WorkloadEntry] = []
        self._in_cuda_graph = False

        # Init the var axes table. It maps def name to a list of all its variable axes, described
        # by tuple (input_name, dim_idx, axis_name).
        self._var_axes_table = self._init_var_axes_table()

        # Create independent dedup policy instances for each definition
        # This ensures state isolation between definitions and runtime instances
        self._dedup_policies: Dict[str, DedupPolicy] = {}
        for def_name, config in self._tracing_configs.items():
            self._dedup_policies[def_name] = config.create_filter_policy()

        # Validate config keys exist in definitions
        for def_name in self._tracing_configs:
            if def_name not in self._trace_set.definitions:
                logger.warning(f"Tracing config found for unknown definition: {def_name}")

        # Register cleanup handlers
        atexit.register(self._exit_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("TracingRuntime Initialized")
        logger.info(f"  TraceSet root path: {self._trace_set.root}")
        logger.info(f"  Tracing configs: {len(self._tracing_configs)} definitions configured")

    def collect(self, def_name: str, runtime_args: Dict[str, Any]):
        """
        Record a workload for later serialization to disk.

        Parameters
        ----------
        def_name : str
            Name of the workload definition to trace.
        runtime_args : Dict[str, Any]
            Runtime arguments containing tensor inputs and other parameters.

        Notes
        -----
        This method validates the runtime arguments against the definition,
        materializes tensor shapes, and stores selected tensors according to
        the tracing configuration. Entries are buffered in memory until flush()
        is called.

        The method will log errors and return early if:
        - The definition is not found or not configured for tracing
        - Runtime arguments don't match the definition inputs
        - Tensor validation fails (wrong shape, dtype, etc.)
        - Axis inference fails
        """
        logger.info(f"Tracing '{def_name}'")
        tracing_config = self._tracing_configs.get(def_name)
        if tracing_config is None:
            logger.error(f"Tracing config not configured for {def_name}, skipping")
            return

        if def_name not in self._trace_set.definitions:
            logger.error(f"Definition {def_name} not found")
            return

        try:
            axes = self._infer_axes(def_name, runtime_args)
        except ValueError as e:
            logger.error(f"Error inferring axes for {def_name}: {e}")
            return

        # Validate runtime arguments
        definition = self._trace_set.definitions[def_name]
        definition_input_names = set(definition.inputs.keys())
        runtime_input_names = set(runtime_args.keys())

        missing = sorted(definition_input_names - runtime_input_names)
        unexpected = sorted(runtime_input_names - definition_input_names)

        if len(missing) > 0:
            logger.error(f"Missing inputs for {def_name}: {missing}")
            return

        if len(unexpected) > 0:
            logger.error(f"Unexpected inputs for {def_name}: {unexpected}")
            return

        # At this point, runtime_args exactly matches definition.inputs
        # Validate inputs_to_dump
        input_names_to_dump = tracing_config.get_inputs_to_dump(runtime_args)

        inputs_to_dump: Dict[str, torch.Tensor] = {}

        for name in input_names_to_dump:
            converted = self._convert_arg_to_tensor(definition, axes, name, runtime_args[name])
            if converted is None:
                return
            inputs_to_dump[name] = converted

        entry = WorkloadEntry(
            def_name=def_name, axes=axes, inputs_to_dump=inputs_to_dump, order=self.order_counter
        )

        with self._lock:
            if self._in_cuda_graph:
                # Deferred snapshot for CUDA Graph replay
                self._cuda_graph_entries.append(entry)
            else:
                # Submit entry directly to dedup policy for online filtering
                filter_policy = self._dedup_policies.get(def_name)
                if filter_policy is not None:
                    filter_policy.submit(entry)

            self.order_counter += 1

    def _convert_arg_to_tensor(
        self, definition: Definition, axes: Dict[str, int], name: str, val: Any
    ) -> Optional[torch.Tensor]:
        """Convert a runtime argument to a tensor for further dumping. If the conversion fails,
        log an error and return None.

        Parameters
        ----------
        definition : Definition
            The workload definition containing axis specifications.
        axes : Dict[str, int]
            Runtime axis values provided during tracing.
        name : str
            Name of the argument to convert.
        val : Any
            The runtime argument to convert.

        Returns
        -------
        Optional[torch.Tensor]
            The converted tensor. None if conversion fails.
        """
        spec = definition.inputs[name]

        try:
            shape_tuple = _materialize_shape(definition, axes, spec.shape)
        except ValueError as e:
            logger.error(f"Error materializing specs for {name}: {e}")
            return

        torch_dtype = dtype_str_to_torch_dtype(spec.dtype)

        if val is None:
            logger.error(f'Tensor name "{name}" to dump is not found for {definition.name}')
            return

        # Scalar input
        if shape_tuple is None:
            python_dtype = dtype_str_to_python_dtype(spec.dtype)
            if not isinstance(val, python_dtype):
                logger.error(
                    f'Input "{name}" must be Python scalar of type {python_dtype.__name__},'
                    f" but got {type(val).__name__}"
                )
                return

            val = torch.tensor(val, dtype=torch_dtype)

        # Tensor input
        else:
            if not isinstance(val, torch.Tensor):
                logger.error(f'Input "{name}" must be a tensor (got {type(val).__name__})')
                return
            if val.shape != shape_tuple:
                logger.error(f'Input "{name}" must have shape {shape_tuple}, got {val.shape}')
                return
            if val.dtype != torch_dtype:
                logger.error(f'Input "{name}" must have dtype {torch_dtype}, got {val.dtype}')
                return

        if not self._in_cuda_graph:
            val = val.detach().cpu().clone()

        return val

    def _init_var_axes_table(self) -> Dict[str, List[Tuple[str, int, str]]]:
        result: Dict[str, List[Tuple[str, int, str]]] = {}
        for definition in self._trace_set.definitions.values():
            axes = []
            for input_name, input_spec in definition.inputs.items():
                if input_spec.shape is None:
                    continue
                for dim_idx, axis_name in enumerate(input_spec.shape):
                    if definition.axes[axis_name].type == "var":
                        axes.append((input_name, dim_idx, axis_name))
            result[definition.name] = axes
        return result

    def _infer_axes(self, def_name: str, runtime_args: Dict[str, Any]) -> Dict[str, int]:
        """
        Use input shape in definition + runtime tensor shapes to determine
        concrete values for every variable axis.

        Parameters
        ----------
        def_name : str
            Name of the definition to infer axes for.
        runtime_args : Dict[str, Any]
            Runtime arguments containing tensor inputs and other parameters.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping axis names to their concrete integer values.

        Raises
        ------
        ValueError
            If var_axes_table is not initialized for the definition, or if the input is missing,
            or if the axis has different values for different inputs.
        """
        var_axes = self._var_axes_table.get(def_name)
        if var_axes is None:
            raise ValueError(f"var_axes_table is not initialized for definition {def_name}")

        axes = {}
        for input_name, dim_idx, axis_name in var_axes:
            tensor = runtime_args.get(input_name)
            if tensor is None:
                raise ValueError(f'Missing input "{input_name}" for definition "{def_name}"')
            if dim_idx >= len(tensor.shape):
                raise ValueError(
                    f'Input "{input_name}" rank {len(tensor.shape)} < dim {dim_idx} of axis '
                    f'"{axis_name}"'
                )
            if axis_name in axes:
                if axes[axis_name] != int(tensor.shape[dim_idx]):
                    raise ValueError(
                        f"Axis {axis_name} has different values for different inputs: "
                        f"{axes[axis_name]} and {int(tensor.shape[dim_idx])}"
                    )
            else:
                axes[axis_name] = int(tensor.shape[dim_idx])
        return axes

    def _snapshot_graph_tensors(self):
        """Snapshot tensors from CUDA Graph entries to CPU memory.

        Notes
        -----
        This method is called automatically when exiting cuda_graph_scope().
        It synchronizes CUDA execution, creates CPU copies of all deferred
        tensors from CUDA Graph entries, and submits them to dedup policies.
        The deferred entries buffer is cleared after processing.
        """
        # Synchronize CUDA before taking snapshots
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for entry in self._cuda_graph_entries:
            # Create CPU snapshots
            snapshot = {}
            for name, tensor in entry.inputs_to_dump.items():
                if isinstance(tensor, torch.Tensor):
                    snapshot[name] = tensor.detach().cpu().clone()
                else:
                    snapshot[name] = tensor
            entry.cuda_graph_snapshot = snapshot
            entry.inputs_to_dump = snapshot

            # Submit to dedup policy
            filter_policy = self._dedup_policies.get(entry.def_name)
            if filter_policy is not None:
                filter_policy.submit(entry)

        self._cuda_graph_entries.clear()

    def _convert_workload_entry_to_trace(self, entry: WorkloadEntry) -> Optional[Trace]:
        """Convert a workload entry to a trace.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to convert.

        Returns
        -------
        Optional[Trace]
            The converted trace. None if conversion fails, such as system error occuring during
            saving tensors.
        """
        workload_uuid = str(uuid.uuid4())
        inputs: Dict[str, InputSpec] = {}

        # Dump picked input tensors
        if len(entry.inputs_to_dump) > 0:
            try:
                save_path = self._trace_set.add_workload_blob_tensor(
                    entry.def_name, workload_uuid, entry.inputs_to_dump
                )
            except Exception as e:
                logger.error(f"Failed to save tensors for {entry.def_name}: {e}")
                return None
            for name in entry.inputs_to_dump:
                inputs[name] = SafetensorsInput(path=save_path, tensor_key=name)

        # Set random for non-picked input tensors
        definition = self._trace_set.definitions[entry.def_name]
        for name in definition.inputs.keys():
            if name not in inputs:
                inputs[name] = RandomInput()

        # Return the trace
        return Trace(
            definition=entry.def_name,
            workload=Workload(axes=entry.axes, inputs=inputs, uuid=workload_uuid),
            solution=None,
            evaluation=None,
        )

    def flush(self):
        """
        Drain selected entries from dedup policies and write to disk.
        """
        # Stats
        num_selected_entries = 0
        num_dump_errors = 0

        if self._in_cuda_graph:
            raise RuntimeError("Cannot flush during CUDA Graph replay")

        # Drain entries from each dedup policy and convert to traces
        for filter_policy in self._dedup_policies.values():
            # Drain selected entries from policy
            selected_entries = filter_policy.drain()

            if len(selected_entries) == 0:
                continue

            traces_to_dump: List[Trace] = []
            for entry in selected_entries:
                trace = self._convert_workload_entry_to_trace(entry)
                if trace is None:
                    num_dump_errors += 1
                else:
                    traces_to_dump.append(trace)

            self._trace_set.add_workload_traces(traces_to_dump)
            num_selected_entries += len(selected_entries)

        # Log stats
        logger.info(
            f"Flush done. {num_selected_entries} entries selected, {num_dump_errors} dump errors"
        )

    def _exit_handler(self):
        """Exit handler. Stores all buffered trace entries to disk."""
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Flush failed at exit: {e}")

    def _signal_handler(self, signum, frame):
        """Signal handler for SIGTERM/SIGINT. Stores all buffered trace entries to disk.

        Parameters
        ----------
        signum : int
            Signal number that triggered the handler.
        frame : FrameType
            Current stack frame when the signal was received.
        """
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Flush failed at signal handler: {e}")

    def __enter__(self) -> "TracingRuntime":
        """Context manager entry point.

        Returns
        -------
        TracingRuntime
            Returns self to enable context manager usage.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit point. Flushes buffered traces and restores previous runtime.

        Parameters
        ----------
        exc_type : Optional[Type[BaseException]]
            Exception type if an exception occurred, None otherwise.
        exc_val : Optional[BaseException]
            Exception instance if an exception occurred, None otherwise.
        exc_tb : Optional[TracebackType]
            Traceback object if an exception occurred, None otherwise.

        Returns
        -------
        bool
            Always returns False to propagate any exceptions that occurred.
        """
        self.flush()
        set_tracing_runtime(self._prev_runtime)
        return False


def _init_tracing_runtime_from_env() -> Optional["TracingRuntime"]:
    """Initialize the global tracing runtime instance from environment variables."""
    fib_enable_tracing = get_fib_enable_tracing()
    if not fib_enable_tracing:
        return None
    fib_dataset_path = get_fib_dataset_path()
    trace_set = TraceSet.from_path(fib_dataset_path)
    tracing_configs = FULL_TRACING_CONFIGS
    return TracingRuntime(trace_set, tracing_configs, None)


# Global singleton tracing runtime instance
_global_tracing_runtime: Optional["TracingRuntime"] = _init_tracing_runtime_from_env()


def get_tracing_runtime() -> Optional["TracingRuntime"]:
    """Get the global tracing runtime instance.

    Returns
    -------
    TracingRuntime
        The global runtime instance.
    """
    return _global_tracing_runtime


def set_tracing_runtime(rt: Optional["TracingRuntime"]) -> None:
    """Set the global tracing runtime instance.

    Parameters
    ----------
    rt : Optional[TracingRuntime]
        The runtime instance to set, or None to clear the global runtime.
    """
    global _global_tracing_runtime
    _global_tracing_runtime = rt


def _get_axis_value(definition: Definition, axes: Dict[str, int], axis_name: str) -> int:
    """Get the integer value for a named axis from runtime axes or definition.

    Parameters
    ----------
    definition : Definition
        The workload definition containing axis specifications.
    axes : Dict[str, int]
        Runtime axis values provided during tracing.
    axis_name : str
        Name of the axis to resolve.

    Returns
    -------
    int
        The resolved integer value for the axis.

    Raises
    ------
    ValueError
        If the axis is unknown, has unsupported type, or is missing required values.
    """
    axis_spec = definition.axes.get(axis_name)
    if axis_spec is None:
        raise ValueError(f'Unknown axis "{axis_name}" in shape')

    if axis_spec.type == "const":
        return axis_spec.value
    elif axis_spec.type == "var":
        if axis_name in axes:
            return int(axes[axis_name])
        else:
            raise ValueError(f'Axis "{axis_name}" is a variable axis but missing in axes')
    raise ValueError(f'Unsupported axis type for "{axis_name}": {axis_spec.type}')


def _materialize_shape(
    definition: Definition, axes: Dict[str, int], shape: Optional[List[str]]
) -> Optional[Tuple[int, ...]]:
    """Convert a shape specification with named axes to concrete integer dimensions.

    Parameters
    ----------
    definition : Definition
        The workload definition containing axis specifications.
    axes : Dict[str, int]
        Runtime axis values provided during tracing.
    shape : Optional[List[str]]
        The symbolized tensor shape from the definition. None for scalar.

    Returns
    -------
    Optional[Tuple[int, ...]]
        Tuple of concrete integer dimensions representing the materialized shape.
        Returns None for scalar.

    Raises
    ------
    ValueError
        If shape specification is None, contains unsupported dimensions,
        or axis resolution fails.
    """
    if shape is None:
        return None
    return tuple(_get_axis_value(definition, axes, dim) for dim in shape)


# TODO: Fix cuda graph support
class CudaGraphTracingRuntime:
    """Context manager for tracing operations within CUDA graphs.

    This class provides a context manager that temporarily enables CUDA graph mode
    in the associated TracingRuntime. When entering the context, it sets the runtime
    to CUDA graph mode, and when exiting, it disables the mode and snapshots any
    graph tensors for tracing purposes.
    """

    def __init__(self, tracing_runtime: "TracingRuntime"):
        """Initialize the CUDA graph tracing runtime context manager.

        Parameters
        ----------
        tracing_runtime : TracingRuntime
            The TracingRuntime instance to manage CUDA graph state for.
        """
        self.tracing_runtime = tracing_runtime

    def __enter__(self) -> "CudaGraphTracingRuntime":
        """Enter the CUDA graph tracing context.

        Sets the associated TracingRuntime to CUDA graph mode under lock protection.

        Returns
        -------
        CudaGraphTracingRuntime
            Returns self to enable context manager usage.
        """
        with self.tracing_runtime._lock:
            self.tracing_runtime._in_cuda_graph = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the CUDA graph tracing context.

        Disables CUDA graph mode and snapshots graph tensors for tracing.

        Parameters
        ----------
        exc_type : Optional[Type[BaseException]]
            Exception type if an exception occurred, None otherwise.
        exc_val : Optional[BaseException]
            Exception instance if an exception occurred, None otherwise.
        exc_tb : Optional[TracebackType]
            Traceback object if an exception occurred, None otherwise.
        """
        with self.tracing_runtime._lock:
            self.tracing_runtime._in_cuda_graph = False
            self.tracing_runtime._snapshot_graph_tensors()
        return False
