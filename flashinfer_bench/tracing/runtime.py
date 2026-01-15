"""Runtime system for collecting and managing workload traces."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch

from flashinfer_bench.data import (
    InputSpec,
    RandomInput,
    SafetensorsInput,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.env import get_fib_dataset_path, get_fib_enable_tracing
from flashinfer_bench.utils import dtype_str_to_torch_dtype

from .config import TracingConfig, TracingConfigRegistry
from .policy import FilterPolicy
from .presets import get_full_configs
from .workload_entry import WorkloadEntry

logger = logging.getLogger(__name__)


class TracingRuntime:
    """Process-wide singleton tracer for workload collection."""

    _stack: ClassVar[List[TracingRuntime]] = []
    """Global runtime stack."""
    _cleanup_registered: ClassVar[bool] = False
    """Whether the cleanup handlers have been registered."""
    _env_initialized: ClassVar[bool] = False
    """Whether initialization from environment variables has been attempted."""

    @classmethod
    def get_instance(cls) -> Optional[TracingRuntime]:
        """Get the current global TracingRuntime instance (top of stack).

        Lazily initializes from environment variable FIB_ENABLE_TRACING on first call.

        Should be called in the main thread.

        Returns
        -------
        Optional[TracingRuntime]
            The global runtime instance (top of stack), or None if stack is empty.
        """
        if len(cls._stack) == 0 and not cls._env_initialized:
            # Try to initialize from env on first access
            cls._env_initialized = True
            env_rt = cls._create_from_env()
            if env_rt is not None:
                env_rt.start()

        if len(cls._stack) > 0:
            return cls._stack[-1]
        return None

    def __init__(
        self,
        trace_set: TraceSet,
        tracing_config: Union[
            TracingConfig, Dict[str, TracingConfig], TracingConfigRegistry, None
        ] = None,
    ):
        """Initialize the tracing runtime.

        Parameters
        ----------
        trace_set : TraceSet
            A TraceSet object.
        tracing_config : Union[TracingConfig, Dict[str, TracingConfig], TracingConfigRegistry, \
                None], optional
            Configuration for the tracing runtime. Can be:

            - TracingConfig: A single config used as the default for all definitions.
            - Dict[str, TracingConfig]: A dict mapping the names of the kernels to trace and their
            configs.
            - TracingConfigRegistry: A registry with per-definition configs.
            - None: Uses the full configs preset.
        """
        self._trace_set = trace_set

        if tracing_config is None:
            self._config_registry = get_full_configs()
        elif isinstance(tracing_config, TracingConfig):
            self._config_registry = TracingConfigRegistry(default=tracing_config)
        elif isinstance(tracing_config, dict):
            self._config_registry = TracingConfigRegistry(per_definition=tracing_config)
        else:
            self._config_registry = tracing_config

        # Global order counter
        self.order_counter = 0

        # Thread safety
        self._lock = threading.Lock()

        # CUDA Graph support
        self._cuda_graph_entries: List[WorkloadEntry] = []
        self._in_cuda_graph = False

        # Create independent filter policy instances for each definition
        # This ensures state isolation between definitions and runtime instances
        self._filter_policies: Dict[str, FilterPolicy] = {}

        # Step 1: Create filter policies for per_definition configs
        for def_name, config in self._config_registry.per_definition.items():
            self._filter_policies[def_name] = config.create_filter_policy()

        # Step 2: If default config exists, create filter policies for remaining definitions
        if self._config_registry.default is not None:
            for def_name in self._trace_set.definitions:
                if def_name not in self._filter_policies:
                    self._filter_policies[def_name] = (
                        self._config_registry.default.create_filter_policy()
                    )

        # Validate per_definition keys exist in definitions
        for def_name in self._config_registry.per_definition:
            if def_name not in self._trace_set.definitions:
                logger.warning(f"Tracing config found for unknown definition: {def_name}")

        from flashinfer_bench.integration.flashinfer import install_flashinfer_integrations

        install_flashinfer_integrations()

    def start(self):
        """Activate this runtime instance. Should be called in the
        main thread.

        If this runtime is already the active instance, this method has no effect.
        Multiple runtimes can be nested; internally they are managed via a stack.
        """
        # The current runtime is already activated, do nothing.
        if len(TracingRuntime._stack) > 0 and TracingRuntime._stack[-1] is self:
            return
        TracingRuntime._register_cleanup()
        TracingRuntime._stack.append(self)

        logger.info("TracingRuntime started")
        logger.info(f"  TraceSet root path: {self._trace_set.root}")
        default_status = "enabled" if self._config_registry.default is not None else "disabled"
        logger.info(
            f"  Tracing configs: default={default_status}, "
            f"{len(self._config_registry.per_definition)} definitions with custom config"
        )

    def stop(self):
        """Deactivate this runtime instance and flush buffered traces. Should be called
        in the main thread.

        The runtime must be currently active. After stopping, the previously active
        runtime (if any) is restored.

        Raises
        ------
        RuntimeError
            If this runtime is not the currently active instance.
        """
        if not TracingRuntime._stack or TracingRuntime._stack[-1] is not self:
            raise RuntimeError(
                "Cannot stop a TracingRuntime that is not the current active instance. "
                "Runtimes must be stopped in LIFO order."
            )

        self.flush()
        TracingRuntime._stack.pop()
        logger.info("TracingRuntime stopped")

    def collect(self, def_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        """Record a workload for later serialization to disk.

        When an error occurs, it will print an error message and return to avoid
        crashing the runtime.

        Only input arguments are traced; output arguments (for destination-passing style) are
        ignored. The calling convention is determined by comparing the number of arguments
        to the definition's inputs and outputs.

        Parameters
        ----------
        def_name : str
            Name of the workload definition to trace.
        args : Tuple[Any, ...]
            Positional arguments. For value-returning style, only inputs.
            For destination-passing style, inputs followed by pre-allocated outputs.
            Only the input portion is traced.
        kwargs : Dict[str, Any]
            Keyword arguments to merge into positional arguments.

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

        tracing_config = self._config_registry.get(def_name)
        if tracing_config is None:
            logger.debug(f"Tracing config not configured for {def_name}, skipping")
            return

        definition = self._trace_set.definitions.get(def_name, None)
        if definition is None:
            logger.error(f"Definition {def_name} not found")
            return

        # Merge kwargs into args based on definition's input/output order
        if kwargs:
            args = definition.merge_kwargs_to_args(args, kwargs)

        # Determine calling convention and extract only inputs
        num_inputs = len(definition.inputs)
        num_outputs = len(definition.outputs)

        if len(args) != num_inputs and len(args) != num_inputs + num_outputs:
            logger.error(
                f"Invalid number of arguments for {def_name}: expected {num_inputs} "
                f"(value-returning) or {num_inputs + num_outputs} (destination-passing), "
                f"got {len(args)}"
            )
            return

        input_names = list(definition.inputs.keys())
        input_values = list(args[:num_inputs])

        # Infer axes
        try:
            axes = definition.get_axes_values_from_inputs(input_values)
        except ValueError as e:
            logger.error(f"Error getting axis values for {def_name}: {e}")
            return

        # Get inputs to dump
        inputs_to_dump = tracing_config.get_inputs_to_dump(input_names, input_values)
        # Convert to tensors
        inputs_to_dump_tensors: Dict[str, torch.Tensor] = {}
        for name, val in inputs_to_dump.items():
            try:
                inputs_to_dump_tensors[name] = self._convert_arg_to_tensor(
                    val, definition.inputs[name].dtype
                )
            except ValueError as e:
                logger.error(f"Error converting argument '{name}' to tensor for {def_name}: {e}")
                return

        # Construct workload entry
        entry = WorkloadEntry(
            def_name=def_name,
            axes=axes,
            inputs_to_dump=inputs_to_dump_tensors,
            order=self.order_counter,
        )

        with self._lock:
            if self._in_cuda_graph:
                # Deferred snapshot for CUDA Graph replay
                self._cuda_graph_entries.append(entry)
            else:
                # Submit entry directly to filter policy for online filtering
                filter_policy = self._filter_policies.get(def_name)
                if filter_policy is not None:
                    filter_policy.submit(entry)

            self.order_counter += 1

    def _convert_arg_to_tensor(
        self, val: Union[int, float, bool, list, tuple, torch.Tensor], dtype: str
    ) -> Optional[torch.Tensor]:
        """Convert a runtime argument to a tensor for further dumping.

        If the conversion fails, log an error and return None.

        Parameters
        ----------
        val : Union[int, float, bool, list, tuple, torch.Tensor]
            The runtime argument to convert.
        dtype : str
            The expected dtype string.

        Returns
        -------
        Optional[torch.Tensor]
            The converted tensor. None if conversion fails.
        """
        if isinstance(val, (int, float, bool, list, tuple)):
            val = torch.tensor(val, dtype=dtype_str_to_torch_dtype(dtype))
            return val
        elif isinstance(val, torch.Tensor):
            return val.detach().cpu().clone()
        else:
            raise ValueError(f"Unsupported argument type: {type(val)}")

    def _snapshot_graph_tensors(self):
        """Snapshot tensors from CUDA Graph entries to CPU memory.

        Notes
        -----
        This method is called automatically when exiting cuda_graph_scope().
        It synchronizes CUDA execution, creates CPU copies of all deferred
        tensors from CUDA Graph entries, and submits them to filter policies.
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

            # Submit to filter policy
            filter_policy = self._filter_policies.get(entry.def_name)
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
        """Drain selected entries from filter policies and write to disk."""
        # Stats
        num_selected_entries = 0
        num_dump_errors = 0

        if self._in_cuda_graph:
            raise RuntimeError("Cannot flush during CUDA Graph replay")

        # Drain entries from each filter policy and convert to traces
        for filter_policy in self._filter_policies.values():
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

    def __enter__(self) -> TracingRuntime:
        """Context manager entry point.

        Returns
        -------
        TracingRuntime
            Returns self to enable context manager usage.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit point. Flushes buffered traces and removes from stack.

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
        self.stop()
        return False

    @classmethod
    def _create_from_env(cls) -> Optional[TracingRuntime]:
        """Initialize the global runtime from environment variables if configured."""
        fib_enable_tracing = get_fib_enable_tracing()
        if not fib_enable_tracing:
            return None
        fib_dataset_path = get_fib_dataset_path()
        trace_set = TraceSet.from_path(fib_dataset_path)
        return cls(trace_set, None)

    @classmethod
    def _register_cleanup(cls) -> None:
        """Register atexit and signal handlers for cleanup. Only registers once."""
        if cls._cleanup_registered:
            return
        cls._cleanup_registered = True
        atexit.register(cls._flush_all)
        cls._orig_sigterm = signal.signal(signal.SIGTERM, cls._signal_cleanup)
        cls._orig_sigint = signal.signal(signal.SIGINT, cls._signal_cleanup)

    @classmethod
    def _flush_all(cls) -> None:
        """Flush all runtimes in the stack."""
        for runtime in list(cls._stack):
            try:
                runtime.flush()
            except Exception as e:
                logger.error(f"Flush failed during cleanup: {e}")

    @classmethod
    def _signal_cleanup(cls, signum: int, frame) -> None:
        """Cleanup handler for signals. Flushes runtimes and chains to original handler."""
        cls._flush_all()

        # Restore and call original handler
        orig_handler = cls._orig_sigterm if signum == signal.SIGTERM else cls._orig_sigint
        signal.signal(signum, orig_handler)

        if callable(orig_handler):
            orig_handler(signum, frame)
        elif orig_handler == signal.SIG_DFL:
            os.kill(os.getpid(), signum)
        # If SIG_IGN, do nothing


# TODO: Fix cuda graph support
class CudaGraphTracingRuntime:
    """Context manager for tracing operations within CUDA graphs.

    This class provides a context manager that temporarily enables CUDA graph mode
    in the associated TracingRuntime. When entering the context, it sets the runtime
    to CUDA graph mode, and when exiting, it disables the mode and snapshots any
    graph tensors for tracing purposes.
    """

    def __init__(self, tracing_runtime: TracingRuntime):
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
