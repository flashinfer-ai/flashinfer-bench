"""Worker process management for the low-level GPU server."""

from __future__ import annotations

import contextlib
import ctypes
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import threading
import time
from typing import Literal

from flashinfer_bench.serve.low_level.cache import CacheEntry
from flashinfer_bench.serve.low_level.errors import (
    ExecutionFailedError,
    InvalidProgramError,
    TimeoutError,
)
from flashinfer_bench.serve.low_level.executor import execute_program
from flashinfer_bench.serve.low_level.schema import (
    ErrorResult,
    Program,
    WorkerExecuteErrorResponse,
    WorkerExecuteRequest,
    WorkerExecuteSuccessResponse,
)

WorkerRuntimeStatus = Literal["idle", "busy", "restarting"]


def _flush_process_output() -> None:
    """Flush Python and libc buffered output before swapping file descriptors.

    Returns
    -------
    None
        This function flushes process output streams in place.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        libc = ctypes.CDLL(None)
        libc.fflush(None)
    except Exception:
        pass


@contextlib.contextmanager
def _capture_process_output():
    """Capture process-level stdout and stderr into temporary files.

    Yields
    ------
    tuple
        Temporary files capturing stdout and stderr for the current request.
    """
    stdout_file = tempfile.TemporaryFile(mode="w+b")
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        _flush_process_output()
        os.dup2(stdout_file.fileno(), 1)
        os.dup2(stderr_file.fileno(), 2)
        yield stdout_file, stderr_file
    finally:
        _flush_process_output()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _read_captured_output(captured_file) -> str | None:
    """Decode one captured output file into UTF-8 text.

    Parameters
    ----------
    captured_file
        Temporary file object holding captured process output.

    Returns
    -------
    str | None
        Decoded UTF-8 text, or `None` when the file is empty.
    """
    if captured_file is None:
        return None
    captured_file.seek(0)
    output = captured_file.read()
    if not output:
        return None
    return output.decode("utf-8", errors="replace")


def _worker_main(device: str, request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    """Run the worker loop bound to a single visible CUDA device.

    Parameters
    ----------
    device
        CUDA device string assigned to this worker.
    request_queue
        Queue used to receive execution requests from the scheduler.
    response_queue
        Queue used to send typed execution responses back to the scheduler.

    Returns
    -------
    None
        This function runs the worker loop until shutdown.
    """
    visible_device = device.split(":", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

    while True:
        message = request_queue.get()
        if message is None:
            return

        if not isinstance(message, WorkerExecuteRequest):
            message = WorkerExecuteRequest.model_validate(message)

        request_id = message.request_id
        program = message.program
        blobs = message.blobs
        if message.request_transport_started_at is not None:
            print(
                f"TRACE request_id={request_id} span=worker_ipc.request_transport phase=end "
                f"duration_ms={(time.perf_counter() - message.request_transport_started_at) * 1000.0:.3f}",
                flush=True,
            )
        stdout_file = None
        stderr_file = None
        execute_program_ms: float | None = None
        read_captured_output_ms: float | None = None
        response_queue_put_ms: float | None = None
        worker_status = "ok"
        request_start_time = time.perf_counter()

        try:
            print(f"TRACE request_id={request_id} span=worker_main.total phase=begin", flush=True)
            with _capture_process_output() as (stdout_file, stderr_file):
                execute_program_start_time = time.perf_counter()
                print(
                    f"TRACE request_id={request_id} span=worker_main.execute_program phase=begin",
                    flush=True,
                )
                result, binary_parts = execute_program(program, blobs, request_id=request_id)
                execute_program_ms = (time.perf_counter() - execute_program_start_time) * 1000.0
                print(
                    f"TRACE request_id={request_id} span=worker_main.execute_program phase=end "
                    f"duration_ms={execute_program_ms:.3f}",
                    flush=True,
                )
            read_captured_output_start_time = time.perf_counter()
            print(
                f"TRACE request_id={request_id} span=worker_main.read_captured_output phase=begin",
                flush=True,
            )
            stdout = _read_captured_output(stdout_file)
            stderr = _read_captured_output(stderr_file)
            read_captured_output_ms = (
                time.perf_counter() - read_captured_output_start_time
            ) * 1000.0
            print(
                f"TRACE request_id={request_id} span=worker_main.read_captured_output phase=end "
                f"duration_ms={read_captured_output_ms:.3f}",
                flush=True,
            )
            result = result.model_copy(update={"stdout": stdout, "stderr": stderr})
            response_queue_put_start_time = time.perf_counter()
            response_transport_started_at = time.perf_counter()
            print(
                f"TRACE request_id={request_id} span=worker_ipc.response_transport phase=begin",
                flush=True,
            )
            print(
                f"TRACE request_id={request_id} span=worker_main.response_queue_put phase=begin",
                flush=True,
            )
            response_queue.put(
                WorkerExecuteSuccessResponse(
                    request_id=request_id,
                    response_transport_started_at=response_transport_started_at,
                    result=result,
                    binary_parts=binary_parts,
                )
            )
            response_queue_put_ms = (time.perf_counter() - response_queue_put_start_time) * 1000.0
            print(
                f"TRACE request_id={request_id} span=worker_main.response_queue_put phase=end "
                f"duration_ms={response_queue_put_ms:.3f}",
                flush=True,
            )
        except InvalidProgramError as error:
            worker_status = "invalid_program"
            read_captured_output_start_time = time.perf_counter()
            stdout = _read_captured_output(stdout_file)
            stderr = _read_captured_output(stderr_file)
            read_captured_output_ms = (
                time.perf_counter() - read_captured_output_start_time
            ) * 1000.0
            response_queue_put_start_time = time.perf_counter()
            response_transport_started_at = time.perf_counter()
            print(
                f"TRACE request_id={request_id} span=worker_ipc.response_transport phase=begin",
                flush=True,
            )
            response_queue.put(
                WorkerExecuteErrorResponse(
                    request_id=request_id,
                    response_transport_started_at=response_transport_started_at,
                    error=ErrorResult(
                        error="invalid_program", message=error.message, stdout=stdout, stderr=stderr
                    ),
                )
            )
            response_queue_put_ms = (time.perf_counter() - response_queue_put_start_time) * 1000.0
        except ExecutionFailedError as error:
            worker_status = "execution_failed"
            read_captured_output_start_time = time.perf_counter()
            stdout = _read_captured_output(stdout_file)
            stderr = _read_captured_output(stderr_file)
            read_captured_output_ms = (
                time.perf_counter() - read_captured_output_start_time
            ) * 1000.0
            response_queue_put_start_time = time.perf_counter()
            response_transport_started_at = time.perf_counter()
            print(
                f"TRACE request_id={request_id} span=worker_ipc.response_transport phase=begin",
                flush=True,
            )
            response_queue.put(
                WorkerExecuteErrorResponse(
                    request_id=request_id,
                    response_transport_started_at=response_transport_started_at,
                    error=ErrorResult(
                        error="execution_failed",
                        message=error.message,
                        instruction_index=error.instruction_index,
                        stdout=stdout,
                        stderr=stderr,
                    ),
                )
            )
            response_queue_put_ms = (time.perf_counter() - response_queue_put_start_time) * 1000.0
        except Exception as error:
            worker_status = "unexpected_error"
            read_captured_output_start_time = time.perf_counter()
            stdout = _read_captured_output(stdout_file)
            stderr = _read_captured_output(stderr_file)
            read_captured_output_ms = (
                time.perf_counter() - read_captured_output_start_time
            ) * 1000.0
            response_queue_put_start_time = time.perf_counter()
            response_transport_started_at = time.perf_counter()
            print(
                f"TRACE request_id={request_id} span=worker_ipc.response_transport phase=begin",
                flush=True,
            )
            response_queue.put(
                WorkerExecuteErrorResponse(
                    request_id=request_id,
                    response_transport_started_at=response_transport_started_at,
                    error=ErrorResult(
                        error="execution_failed", message=str(error), stdout=stdout, stderr=stderr
                    ),
                )
            )
            response_queue_put_ms = (time.perf_counter() - response_queue_put_start_time) * 1000.0
        finally:
            print(
                "WORKER_MAIN_TIMING "
                f"request_id={request_id} "
                f"status={worker_status} "
                f"total_request_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f} "
                f"execute_program_ms={execute_program_ms} "
                f"read_captured_output_ms={read_captured_output_ms} "
                f"response_queue_put_ms={response_queue_put_ms}",
                flush=True,
            )
            print(
                f"TRACE request_id={request_id} span=worker_main.total phase=end "
                f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
                flush=True,
            )
            if stdout_file is not None:
                stdout_file.close()
            if stderr_file is not None:
                stderr_file.close()


class LowLevelWorkerProcess:
    """One dedicated worker process bound to one GPU."""

    def __init__(self, device: str):
        """Initialize one worker process controller.

        Parameters
        ----------
        device
            CUDA device string assigned to this worker.

        Returns
        -------
        None
            This constructor initializes internal state and starts the worker.
        """
        self.device = device
        self.gpu_id = int(device.split(":", 1)[1])
        self._context = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._context.Queue()
        self._response_queue: mp.Queue = self._context.Queue()
        self._process: mp.Process | None = None
        self._status: WorkerRuntimeStatus = "idle"
        self._busy = False
        self._started_at = 0.0
        self._lock = threading.Lock()
        self.start()

    @property
    def status(self) -> WorkerRuntimeStatus:
        """Return the public runtime status for this worker.

        Returns
        -------
        WorkerRuntimeStatus
            Public worker status exposed through the health endpoint.
        """
        if self._status == "restarting":
            return "restarting"
        return "busy" if self._busy else "idle"

    @property
    def queue_length(self) -> int:
        """Return the number of active requests tracked by this worker.

        Returns
        -------
        int
            Number of currently active requests tracked by this worker.
        """
        return 1 if self._busy else 0

    @property
    def uptime_seconds(self) -> int:
        """Return the worker uptime in seconds since the last process start.

        Returns
        -------
        int
            Worker uptime in whole seconds.
        """
        if self._started_at == 0:
            return 0
        return max(0, int(time.time() - self._started_at))

    def is_alive(self) -> bool:
        """Check whether the underlying worker process is still alive.

        Returns
        -------
        bool
            Whether the worker process is currently alive.
        """
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Spawn the worker process and reset runtime state.

        Returns
        -------
        None
            This method starts the worker process in place.
        """
        self._process = self._context.Process(
            target=_worker_main,
            args=(self.device, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()
        self._started_at = time.time()
        self._status = "idle"
        self._busy = False

    def close(self) -> None:
        """Shut down the worker process and release queue resources.

        Returns
        -------
        None
            This method releases worker-owned process and queue resources in place.
        """
        if self._process is None:
            return
        try:
            self._request_queue.put(None)
        except Exception:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2.0)
        self._process = None
        for queue_object in (self._request_queue, self._response_queue):
            try:
                queue_object.cancel_join_thread()
            except Exception:
                pass
            try:
                queue_object.close()
            except Exception:
                pass

    def restart(self) -> None:
        """Restart the worker process after a fatal error or timeout.

        Returns
        -------
        None
            This method replaces the underlying worker process in place.
        """
        self._status = "restarting"
        self.close()
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self.start()

    def execute(
        self,
        request_id: str,
        program: Program,
        blobs: dict[str, CacheEntry],
        timeout_seconds: float,
    ) -> WorkerExecuteSuccessResponse | WorkerExecuteErrorResponse:
        """Execute one request on this worker and wait for the typed IPC response.

        Parameters
        ----------
        program
            Validated low-level program to execute.
        blobs
            Uploaded request blobs stored in the server cache and keyed by blob hash.
        timeout_seconds
            Effective timeout in seconds for the request.

        Returns
        -------
        WorkerExecuteSuccessResponse | WorkerExecuteErrorResponse
            Typed IPC response returned by the worker process.
        """
        request_transport_started_at = time.perf_counter()
        request_message = WorkerExecuteRequest(
            request_id=request_id,
            request_transport_started_at=request_transport_started_at,
            program=program,
            blobs=blobs,
        )
        request_start_time = time.perf_counter()
        request_queue_put_ms: float | None = None
        response_wait_ms: float | None = None
        status = "ok"
        with self._lock:
            self._busy = True
            try:
                request_queue_put_start_time = time.perf_counter()
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.total phase=begin", flush=True
                )
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.request_queue_put phase=begin",
                    flush=True,
                )
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.request_transport phase=begin",
                    flush=True,
                )
                self._request_queue.put(request_message)
                request_queue_put_ms = (time.perf_counter() - request_queue_put_start_time) * 1000.0
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.request_queue_put phase=end "
                    f"duration_ms={request_queue_put_ms:.3f}",
                    flush=True,
                )
                response_wait_start_time = time.perf_counter()
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.response_wait phase=begin",
                    flush=True,
                )
                response = self._response_queue.get(timeout=timeout_seconds)
                response_wait_ms = (time.perf_counter() - response_wait_start_time) * 1000.0
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.response_wait phase=end "
                    f"duration_ms={response_wait_ms:.3f}",
                    flush=True,
                )
                if response.response_transport_started_at is not None:
                    print(
                        f"TRACE request_id={request_id} span=worker_ipc.response_transport phase=end "
                        f"duration_ms={(time.perf_counter() - response.response_transport_started_at) * 1000.0:.3f}",
                        flush=True,
                    )
                if response.request_id != request_id:
                    raise RuntimeError("Mismatched worker response")
                return response
            except queue.Empty:
                status = "timeout"
                self.restart()
                raise TimeoutError(timeout_seconds)
            finally:
                print(
                    "WORKER_IPC_TIMING "
                    f"request_id={request_id} "
                    f"status={status} "
                    f"total_execute_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f} "
                    f"request_queue_put_ms={request_queue_put_ms} "
                    f"response_wait_ms={response_wait_ms} ",
                    flush=True,
                )
                print(
                    f"TRACE request_id={request_id} span=worker_ipc.total phase=end "
                    f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
                    flush=True,
                )
                self._busy = False
