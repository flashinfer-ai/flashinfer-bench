"""Program executor for the low-level GPU server."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

import torch
import tvm_ffi
from tvm_ffi.registry import list_global_func_names

from flashinfer_bench.serve.low_level.cache import CacheEntry, ShmUtils
from flashinfer_bench.serve.low_level.errors import ExecutionFailedError, InvalidProgramError
from flashinfer_bench.serve.low_level.schema import (
    BytesReturnValue,
    CallInstruction,
    ExecuteResult,
    Instruction,
    JsonReturnValue,
    Program,
    RegisterRef,
    ReturnInstruction,
    ReturnValue,
    TensorReturnValue,
    UploadFfiModuleInstruction,
    UploadPythonModuleInstruction,
    UploadTensorInstruction,
)
from flashinfer_bench.utils import dtype_str_to_torch_dtype

_TORCH_DTYPE_TO_NAME: Dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.float4_e2m1fn_x2: "float4_e2m1",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    register_file: Dict[int, Any] = field(default_factory=dict)
    """Request-local register file keyed by register index."""
    registered_function_names: set[str] = field(default_factory=set)
    """Global function names registered during the current request."""
    loaded_shared_modules: list[Any] = field(default_factory=list)
    """Uploaded FFI module handles kept alive for the current request."""
    loaded_python_modules: list[Any] = field(default_factory=list)
    """Imported Python module objects kept alive for the current request."""
    loaded_python_module_names: list[str] = field(default_factory=list)
    """Temporary module names inserted into `sys.modules` for uploaded Python modules."""
    blobs: Dict[str, CacheEntry] = field(default_factory=dict)
    """Uploaded request blobs addressable by blob hash."""

    def get_register(self, register_index: int) -> Any:
        """Read a value from the request register file.

        Parameters
        ----------
        register_index
            Register index to read.

        Returns
        -------
        Any
            Value currently stored in the register.
        """
        if register_index not in self.register_file:
            raise InvalidProgramError(f"Register r{register_index} is not defined")
        return self.register_file[register_index]

    def set_register(self, register_index: int, value: Any) -> None:
        """Store a value in the request register file.

        Parameters
        ----------
        register_index
            Register index to write.
        value
            Value to store in the register.

        Returns
        -------
        None
            This method updates the request context in place.
        """
        self.register_file[register_index] = value

    def is_loaded_shared_module(self, value: Any) -> bool:
        """Check whether a value is a request-local uploaded FFI module handle.

        Parameters
        ----------
        value
            Candidate value to test.

        Returns
        -------
        bool
            Whether the value matches one uploaded FFI module handle by identity.
        """
        return any(value is shared_module for shared_module in self.loaded_shared_modules)

    def get_blob_buffer(self, blob_hash: str) -> memoryview:
        """Return a memoryview over one uploaded blob.

        Parameters
        ----------
        blob_hash
            Blob hash referenced by the program instruction.

        Returns
        -------
        memoryview
            Memory-mapped view spanning the logical blob bytes.
        """
        if blob_hash not in self.blobs:
            raise InvalidProgramError(f"Missing blob: {blob_hash}")
        blob = self.blobs[blob_hash]
        if blob.size == 0:
            return memoryview(b"")
        try:
            return ShmUtils.read(blob.path)[: blob.size]
        except FileNotFoundError as error:
            raise InvalidProgramError(f"Missing blob file: {blob_hash}") from error

    def read_blob_bytes(self, blob_hash: str) -> bytes:
        """Copy one uploaded blob into an immutable bytes object.

        Parameters
        ----------
        blob_hash
            Blob hash referenced by the program instruction.

        Returns
        -------
        bytes
            Immutable copy of the requested blob payload.
        """
        if blob_hash not in self.blobs:
            raise InvalidProgramError(f"Missing blob: {blob_hash}")
        blob = self.blobs[blob_hash]
        if blob.size == 0:
            return b""
        blob_buffer = self.get_blob_buffer(blob_hash)
        try:
            return bytes(blob_buffer)
        finally:
            blob_buffer.release()

    def cleanup(self) -> None:
        """Release request-local state after execution finishes.

        Returns
        -------
        None
            This method releases request-owned resources in place.
        """
        self.register_file.clear()
        for function_name in sorted(self.registered_function_names):
            try:
                tvm_ffi.remove_global_func(function_name)
            except Exception:
                pass
        self.loaded_shared_modules.clear()
        for module_name in self.loaded_python_module_names:
            sys.modules.pop(module_name, None)
        self.loaded_python_modules.clear()
        self.loaded_python_module_names.clear()
        self.blobs.clear()
        self.registered_function_names.clear()


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------


def execute_program(
    program: Program, blobs: Dict[str, CacheEntry], request_id: str | None = None
) -> Tuple[ExecuteResult, Dict[str, bytes]]:
    """Execute a validated program against request blobs.

    Parameters
    ----------
    program
        Validated low-level program to execute.
    blobs
        Uploaded request blobs stored in the server cache and keyed by blob hash.

    Returns
    -------
    tuple[ExecuteResult, dict[str, bytes]]
        Structured result payload plus binary multipart parts.
    """
    context = RequestContext(blobs=blobs)
    returned_values: Dict[str, Any] = {}
    execute_program_start_time = perf_counter()
    instruction_loop_ms: float | None = None
    serialize_returns_ms: float | None = None
    elapsed_ms: float | None = None
    cleanup_ms: float | None = None
    try:
        if request_id is not None:
            print(f"TRACE request_id={request_id} span=executor.total phase=begin", flush=True)
        with tempfile.TemporaryDirectory(prefix="fib_low_level_worker_") as temporary_directory:
            tmp_dir = Path(temporary_directory)
            start_time = perf_counter()
            instruction_loop_start_time = perf_counter()
            if request_id is not None:
                print(
                    f"TRACE request_id={request_id} span=executor.instruction_loop phase=begin",
                    flush=True,
                )
            for instruction_index, instruction in enumerate(program.instructions):
                _execute_instruction(
                    context, tmp_dir, instruction, instruction_index, returned_values
                )
            instruction_loop_ms = (perf_counter() - instruction_loop_start_time) * 1000.0
            if request_id is not None:
                print(
                    f"TRACE request_id={request_id} span=executor.instruction_loop phase=end "
                    f"duration_ms={instruction_loop_ms:.3f}",
                    flush=True,
                )
            serialize_returns_start_time = perf_counter()
            if request_id is not None:
                print(
                    f"TRACE request_id={request_id} span=executor.serialize_returns phase=begin",
                    flush=True,
                )
            returns, binary_parts = _serialize_returns(returned_values)
            serialize_returns_ms = (perf_counter() - serialize_returns_start_time) * 1000.0
            if request_id is not None:
                print(
                    f"TRACE request_id={request_id} span=executor.serialize_returns phase=end "
                    f"duration_ms={serialize_returns_ms:.3f}",
                    flush=True,
                )
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            result = ExecuteResult(returns=returns, elapsed_ms=elapsed_ms)
            return result, binary_parts
    finally:
        cleanup_start_time = perf_counter()
        if request_id is not None:
            print(f"TRACE request_id={request_id} span=executor.cleanup phase=begin", flush=True)
        context.cleanup()
        cleanup_ms = (perf_counter() - cleanup_start_time) * 1000.0
        if request_id is not None:
            print(
                f"TRACE request_id={request_id} span=executor.cleanup phase=end "
                f"duration_ms={cleanup_ms:.3f}",
                flush=True,
            )
        print(
            "EXECUTOR_TIMING "
            f"request_id={request_id} "
            f"total_execute_program_ms={(perf_counter() - execute_program_start_time) * 1000.0:.3f} "
            f"elapsed_ms={elapsed_ms} "
            f"instruction_loop_ms={instruction_loop_ms} "
            f"serialize_returns_ms={serialize_returns_ms} "
            f"cleanup_ms={cleanup_ms}",
            flush=True,
        )
        if request_id is not None:
            print(
                f"TRACE request_id={request_id} span=executor.total phase=end "
                f"duration_ms={(perf_counter() - execute_program_start_time) * 1000.0:.3f}",
                flush=True,
            )


def _execute_instruction(
    context: RequestContext,
    tmp_dir: Path,
    instruction: Instruction,
    instruction_index: int,
    returned_values: Dict[str, Any],
) -> None:
    """Dispatch one typed instruction to the matching execution helper.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        Typed instruction to execute.
    instruction_index
        Zero-based instruction index in the program.
    returned_values
        Mutable map populated by `return` instructions.

    Returns
    -------
    None
        This function mutates the request context and return map in place.
    """
    if isinstance(instruction, UploadPythonModuleInstruction):
        _execute_upload_python_module(context, tmp_dir, instruction, instruction_index)
    elif isinstance(instruction, UploadFfiModuleInstruction):
        _execute_upload_ffi_module(context, tmp_dir, instruction, instruction_index)
    elif isinstance(instruction, UploadTensorInstruction):
        _execute_upload_tensor(context, instruction)
    elif isinstance(instruction, CallInstruction):
        _execute_call(context, instruction, instruction_index)
    elif isinstance(instruction, ReturnInstruction):
        _execute_return(context, instruction, returned_values)


def _execute_upload_python_module(
    context: RequestContext,
    tmp_dir: Path,
    instruction: UploadPythonModuleInstruction,
    instruction_index: int,
) -> None:
    """Import a Python module blob and track its registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        Python module upload instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    try:
        blob_bytes = context.read_blob_bytes(instruction.blob)
        _load_python_module_blob(context, tmp_dir, instruction.blob, blob_bytes)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _execute_upload_ffi_module(
    context: RequestContext,
    tmp_dir: Path,
    instruction: UploadFfiModuleInstruction,
    instruction_index: int,
) -> None:
    """Load a shared library blob and store the resulting module handle.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    instruction
        FFI module upload instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    try:
        blob_bytes = context.read_blob_bytes(instruction.blob)
        loaded_module = _load_shared_library_blob(context, tmp_dir, instruction.blob, blob_bytes)
        context.set_register(instruction.dst, loaded_module)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _load_python_module_blob(
    context: RequestContext, tmp_dir: Path, blob_hash: str, blob_bytes: bytes
) -> None:
    """Import a Python module blob and record newly registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    blob_hash
        Blob hash used to derive the temporary file name.
    blob_bytes
        Python module source bytes.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    before_function_names = set(list_global_func_names())
    module_name = f"flashinfer_bench_uploaded_{uuid.uuid4().hex}"
    source_path = tmp_dir / f"{blob_hash}.py"
    source_path.write_bytes(blob_bytes)
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise InvalidProgramError(f"Failed to build import spec for blob: {blob_hash}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        after_function_names = set(list_global_func_names())
        context.registered_function_names.update(after_function_names - before_function_names)
        sys.modules.pop(module_name, None)
        raise

    after_function_names = set(list_global_func_names())
    context.registered_function_names.update(after_function_names - before_function_names)
    context.loaded_python_modules.append(module)
    context.loaded_python_module_names.append(module_name)


def _load_shared_library_blob(
    context: RequestContext, tmp_dir: Path, blob_hash: str, blob_bytes: bytes
):
    """Load a shared library blob and record newly registered global functions.

    Parameters
    ----------
    context
        Request-local execution state.
    tmp_dir
        Temporary directory used for request-local module files.
    blob_hash
        Blob hash used to derive the temporary file name.
    blob_bytes
        Shared-library bytes.

    Returns
    -------
    Any
        Loaded FFI module handle returned by `tvm_ffi.load_module`.
    """
    if not blob_bytes.startswith(b"\x7fELF"):
        raise InvalidProgramError(
            f"upload_ffi_module expects an ELF shared library blob: {blob_hash}"
        )

    before_function_names = set(list_global_func_names())
    shared_library_path = tmp_dir / f"{blob_hash}.so"
    shared_library_path.write_bytes(blob_bytes)
    try:
        loaded_module = tvm_ffi.load_module(str(shared_library_path), keep_module_alive=False)
    except Exception:
        after_function_names = set(list_global_func_names())
        context.registered_function_names.update(after_function_names - before_function_names)
        raise

    after_function_names = set(list_global_func_names())
    context.registered_function_names.update(after_function_names - before_function_names)
    context.loaded_shared_modules.append(loaded_module)
    return loaded_module


def _execute_upload_tensor(context: RequestContext, instruction: UploadTensorInstruction) -> None:
    """Materialize a tensor blob on the worker GPU and store it in a register.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Tensor upload instruction to execute.

    Returns
    -------
    None
        This function mutates the request context in place.
    """
    blob_buffer = context.get_blob_buffer(instruction.blob)
    torch_dtype = dtype_str_to_torch_dtype(instruction.dtype)
    expected_nbytes = math.prod(instruction.shape) * torch_dtype.itemsize
    try:
        if len(blob_buffer) != expected_nbytes:
            raise InvalidProgramError(
                f"upload_tensor blob size mismatch: expected {expected_nbytes} bytes, got {len(blob_buffer)}"
            )

        tensor = torch.frombuffer(blob_buffer, dtype=torch_dtype)
        tensor = tensor.view(instruction.shape)
        tensor = tensor.to("cuda:0")
        context.set_register(instruction.dst, tensor)
    finally:
        blob_buffer.release()


def _execute_call(
    context: RequestContext, instruction: CallInstruction, instruction_index: int
) -> None:
    """Invoke a function and optionally write the return value to a register.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Function call instruction to execute.
    instruction_index
        Zero-based instruction index in the program.

    Returns
    -------
    None
        This function mutates the request context in place when `dst` is provided.
    """
    try:
        resolved_args = [_resolve_operand(context, item) for item in instruction.args]
        function = _resolve_function(context, instruction.func, instruction.module)
        if function is None:
            raise ExecutionFailedError(
                message=f"Unknown function: {instruction.func}", instruction_index=instruction_index
            )
        result = function(*resolved_args)
        if instruction.dst is not None:
            context.set_register(instruction.dst, result)
    except (InvalidProgramError, ExecutionFailedError):
        raise
    except Exception as error:
        raise ExecutionFailedError(
            message=str(error), instruction_index=instruction_index
        ) from error


def _resolve_function(context: RequestContext, function_name: str, module_ref: RegisterRef | None):
    """Resolve a callable either from global scope or from a module register.

    Parameters
    ----------
    context
        Request-local execution state.
    function_name
        Function name to resolve.
    module_ref
        Optional register reference pointing to an uploaded FFI module handle.

    Returns
    -------
    Any
        Resolved callable object, or `None` when the global function is missing.
    """
    if module_ref is None:
        return tvm_ffi.get_global_func(function_name, allow_missing=True)

    module_value = context.get_register(module_ref.r)
    if context.is_loaded_shared_module(module_value):
        return module_value.get_function(function_name)
    raise InvalidProgramError("call.module must resolve to an uploaded FFI module handle")


def _execute_return(
    context: RequestContext, instruction: ReturnInstruction, returned_values: Dict[str, Any]
) -> None:
    """Expose a register value in the final named return map.

    Parameters
    ----------
    context
        Request-local execution state.
    instruction
        Return instruction to execute.
    returned_values
        Mutable map populated with named return values.

    Returns
    -------
    None
        This function mutates the return map in place.
    """
    returned_values[instruction.key] = context.get_register(instruction.reg)


def _resolve_operand(context: RequestContext, operand: Any) -> Any:
    """Resolve a literal operand or a register reference.

    Parameters
    ----------
    context
        Request-local execution state.
    operand
        Literal value or register reference appearing in the instruction.

    Returns
    -------
    Any
        Resolved Python value ready to pass into the target function.
    """
    if isinstance(operand, RegisterRef):
        return context.get_register(operand.r)
    return operand


def _serialize_returns(
    returned_values: Dict[str, Any],
) -> Tuple[Dict[str, ReturnValue], Dict[str, bytes]]:
    """Convert returned register values into the HTTP result payload.

    Parameters
    ----------
    returned_values
        Named values exported by `return` instructions.

    Returns
    -------
    tuple[dict[str, ReturnValue], dict[str, bytes]]
        Serialized return metadata plus binary multipart parts.
    """
    returns: Dict[str, ReturnValue] = {}
    binary_parts: Dict[str, bytes] = {}

    for key, value in returned_values.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            part_name = f"return:{key}"
            binary_parts[part_name] = bytes(tensor.untyped_storage())
            returns[key] = TensorReturnValue(
                dtype=_TORCH_DTYPE_TO_NAME[tensor.dtype], shape=list(tensor.shape), blob=part_name
            )
        elif isinstance(value, (bytes, bytearray)):
            part_name = f"return:{key}"
            binary_parts[part_name] = bytes(value)
            returns[key] = BytesReturnValue(blob=part_name)
        else:
            returns[key] = JsonReturnValue(value=value)

    return returns, binary_parts
