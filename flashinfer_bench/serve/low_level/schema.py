"""Schemas for the low-level GPU server."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from flashinfer_bench.serve.low_level.cache import CacheEntry

# ---------------------------------------------------------------------------
# Program Schema
# ---------------------------------------------------------------------------

SUPPORTED_DTYPES = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
]


class RegisterRef(BaseModel):
    """Reference a value previously stored in the register file."""

    r: int
    """Register index to read from."""


Operand = Union[RegisterRef, int, float, str, bool, list, None]


class UploadPythonModuleInstruction(BaseModel):
    """Import a Python module blob into the worker process."""

    op: Literal["upload_python_module"]
    """Instruction opcode."""
    blob: str
    """Blob key referencing the uploaded Python module source."""


class UploadFfiModuleInstruction(BaseModel):
    """Load a TVM FFI shared library blob and store its module handle."""

    op: Literal["upload_ffi_module"]
    """Instruction opcode."""
    blob: str
    """Blob key referencing the uploaded shared library bytes."""
    dst: int
    """Destination register for the loaded FFI module handle."""


class UploadTensorInstruction(BaseModel):
    """Load a tensor blob onto the worker GPU and store it in a register."""

    op: Literal["upload_tensor"]
    """Instruction opcode."""
    dst: int
    """Destination register for the uploaded tensor."""
    blob: str
    """Blob key referencing the raw tensor bytes."""
    shape: list[int]
    """Tensor shape in row-major order."""
    dtype: SUPPORTED_DTYPES
    """Tensor dtype name."""


class CallInstruction(BaseModel):
    """Call a function with literal operands or register references."""

    op: Literal["call"]
    """Instruction opcode."""
    func: str
    """Function name to invoke."""
    args: list[Operand]
    """Function arguments in call order."""
    dst: int | None = None
    """Destination register for the return value."""
    module: RegisterRef | None = None
    """Optional register reference to a module handle used for function lookup."""


class ReturnInstruction(BaseModel):
    """Expose a register value in the final response payload."""

    op: Literal["return"]
    """Instruction opcode."""
    reg: int
    """Register index to export."""
    key: str
    """Response key used in the returns map."""


Instruction = Union[
    UploadPythonModuleInstruction,
    UploadFfiModuleInstruction,
    UploadTensorInstruction,
    CallInstruction,
    ReturnInstruction,
]


class Program(BaseModel):
    """Executable low-level program independent of HTTP request metadata."""

    instructions: list[Instruction]
    """Ordered instruction sequence to execute."""


# ---------------------------------------------------------------------------
# HTTP Request Schema
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    """JSON payload stored in the ``program`` multipart field for ``/execute``."""

    instructions: list[Instruction]
    """Ordered instruction sequence to execute."""
    timeout_seconds: float | None = None
    """Optional per-request timeout override in seconds."""

    @property
    def program(self) -> Program:
        """Return the execution program without HTTP-only request metadata."""
        return Program(instructions=self.instructions)


class TensorReturnValue(BaseModel):
    """Tensor return value described by metadata plus a binary blob part."""

    type: Literal["tensor"] = "tensor"
    """Return value kind."""
    dtype: SUPPORTED_DTYPES
    """Tensor dtype name."""
    shape: list[int]
    """Tensor shape in row-major order."""
    blob: str
    """Multipart part name containing the raw tensor bytes."""


class BytesReturnValue(BaseModel):
    """Opaque bytes return value stored in a multipart blob part."""

    type: Literal["bytes"] = "bytes"
    """Return value kind."""
    blob: str
    """Multipart part name containing the raw bytes."""


class JsonReturnValue(BaseModel):
    """JSON-serializable return value embedded directly in the result payload."""

    type: Literal["json"] = "json"
    """Return value kind."""
    value: Any
    """JSON-serializable return value."""


ReturnValue = Union[TensorReturnValue, BytesReturnValue, JsonReturnValue]


class ExecuteResult(BaseModel):
    """Successful execution result returned in the ``result`` multipart field."""

    status: Literal["ok"] = "ok"
    """Execution status."""
    returns: dict[str, ReturnValue]
    """Named return values exported by ``return`` instructions."""
    elapsed_ms: float
    """Wall-clock execution time in milliseconds."""
    stdout: str | None = None
    """Captured standard output from the worker process."""
    stderr: str | None = None
    """Captured standard error from the worker process."""


class ErrorResult(BaseModel):
    """Structured execution error returned as JSON when the request fails."""

    status: Literal["error"] = "error"
    """Execution status."""
    error: str
    """Stable machine-readable error code."""
    message: str | None = None
    """Human-readable error summary."""
    instruction_index: int | None = None
    """Zero-based instruction index associated with the failure, when available."""
    timeout_seconds: float | None = None
    """Effective timeout that was exceeded, when the error is a timeout."""
    stdout: str | None = None
    """Captured standard output before the failure."""
    stderr: str | None = None
    """Captured standard error before the failure."""


class ServerExecuteRequest(BaseModel):
    """Decoded server-side execute request independent of HTTP transport."""

    execute_request: ExecuteRequest
    """Validated execute request payload."""
    blob_entries: dict[str, CacheEntry]
    """Cached blob entries referenced by the request."""


class ServerExecuteResponseKind(str, Enum):
    """Semantic result kind returned by the server runtime."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class ServerExecuteResponse(BaseModel):
    """Server runtime result independent of HTTP response formatting."""

    kind: ServerExecuteResponseKind
    """Semantic result kind returned by the server runtime."""
    payload: ExecuteResult | ErrorResult
    """Structured JSON payload returned to the HTTP layer."""
    binary_parts: dict[str, bytes] = Field(default_factory=dict)
    """Binary multipart payloads returned on successful execution."""


class WorkerStatus(BaseModel):
    """Health summary for a single GPU worker."""

    gpu_id: int
    """Physical GPU index assigned to the worker."""
    status: Literal["idle", "busy", "restarting"]
    """Current worker runtime status."""
    uptime_seconds: int
    """Worker uptime since the last process start."""


class HealthResponse(BaseModel):
    """Response payload for the ``/health`` endpoint."""

    status: Literal["ok"] = "ok"
    """Health check status."""
    gpu_count: int
    """Number of configured GPU workers."""
    pending_tasks: int
    """Number of queued requests waiting for a worker."""
    workers: list[WorkerStatus]
    """Per-worker runtime status entries."""


# ---------------------------------------------------------------------------
# Server-Worker IPC Schema
# ---------------------------------------------------------------------------


class WorkerExecuteRequest(BaseModel):
    """IPC request sent from the scheduler process to a worker process."""

    request_id: str
    """Unique request identifier used to match the worker response."""
    request_transport_started_at: float | None = None
    """Monotonic timestamp recorded before the parent enqueues the request."""
    program: Program
    """Typed execution program."""
    blobs: dict[str, CacheEntry]
    """Uploaded request blobs stored in the server cache."""


class WorkerExecuteSuccessResponse(BaseModel):
    """IPC response emitted by a worker after successful execution."""

    request_id: str
    """Unique request identifier used to match the original request."""
    response_transport_started_at: float | None = None
    """Monotonic timestamp recorded before the child enqueues the response."""
    ok: Literal[True] = True
    """Success discriminator for the worker response."""
    result: ExecuteResult
    """Structured execution result."""
    binary_parts: dict[str, bytes]
    """Named binary multipart payloads returned by execution."""


class WorkerExecuteErrorResponse(BaseModel):
    """IPC response emitted by a worker after execution failure."""

    request_id: str
    """Unique request identifier used to match the original request."""
    response_transport_started_at: float | None = None
    """Monotonic timestamp recorded before the child enqueues the response."""
    ok: Literal[False] = False
    """Failure discriminator for the worker response."""
    error: ErrorResult
    """Structured execution error."""
