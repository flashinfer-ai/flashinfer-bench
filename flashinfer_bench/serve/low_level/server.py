"""Server runtime and scheduling for the low-level GPU server."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from flashinfer_bench.serve.low_level.cache import CacheEntry, CacheManager
from flashinfer_bench.serve.low_level.errors import TimeoutError
from flashinfer_bench.serve.low_level.schema import (
    ErrorResult,
    HealthResponse,
    Program,
    ServerExecuteRequest,
    ServerExecuteResponse,
    ServerExecuteResponseKind,
    WorkerExecuteErrorResponse,
    WorkerExecuteSuccessResponse,
    WorkerStatus,
)
from flashinfer_bench.serve.low_level.worker import LowLevelWorkerProcess


@dataclass
class _PendingTask:
    """One server-side queued request waiting to run on a worker."""

    request_id: str
    program: Program
    timeout_seconds: float
    blob_entries: dict[str, CacheEntry]
    future: asyncio.Future[ServerExecuteResponse]
    enqueued_at: float
    started_at: float | None = None


class LowLevelServer:
    """Own the low-level server runtime, scheduling, and decoded request execution."""

    def __init__(self, devices: list[str], default_timeout_seconds: int = 30) -> None:
        """Initialize one low-level server instance.

        Parameters
        ----------
        devices
            CUDA device strings assigned to worker processes.
        default_timeout_seconds
            Default request timeout in seconds.

        Returns
        -------
        None
            This constructor initializes the cache manager and worker scheduling state.
        """
        self.cache_manager = CacheManager()
        self._default_timeout_seconds = default_timeout_seconds
        self._workers = [LowLevelWorkerProcess(device=device) for device in devices]
        self._pending_queue: asyncio.Queue[_PendingTask] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start one queue consumer coroutine per worker."""
        if self._worker_tasks:
            return
        self._worker_tasks = [
            asyncio.create_task(
                self._worker_loop(worker), name=f"low-level-worker-loop-{worker.gpu_id}"
            )
            for worker in self._workers
        ]

    async def shutdown(self) -> None:
        """Cancel queue consumers, drain pending tasks, and close worker processes."""
        if self._worker_tasks:
            for worker_task in self._worker_tasks:
                worker_task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
        self._drain_pending_queue()
        for worker in self._workers:
            worker.close()

    async def execute_request(
        self, server_execute_request: ServerExecuteRequest, trace_id: str
    ) -> ServerExecuteResponse:
        """Execute one decoded low-level program request.

        Parameters
        ----------
        server_execute_request
            Decoded execute request plus cached blob entries referenced by it.

        Returns
        -------
        ServerExecuteResponse
            Semantic execution result independent of HTTP response formatting.
        """
        request_start_time = time.perf_counter()
        print(f"TRACE request_id={trace_id} span=server.execute_request phase=begin", flush=True)
        execute_request = server_execute_request.execute_request
        blob_entries = server_execute_request.blob_entries
        effective_timeout = float(execute_request.timeout_seconds or self._default_timeout_seconds)
        loop = asyncio.get_running_loop()
        pending_task = _PendingTask(
            request_id=trace_id,
            program=execute_request.program,
            timeout_seconds=effective_timeout,
            blob_entries=blob_entries,
            future=loop.create_future(),
            enqueued_at=time.monotonic(),
        )
        queued = False
        try:
            queue_put_start_time = time.perf_counter()
            print(f"TRACE request_id={trace_id} span=server.queue_put phase=begin", flush=True)
            await self._pending_queue.put(pending_task)
            queued = True
            print(
                f"TRACE request_id={trace_id} span=server.queue_put phase=end "
                f"duration_ms={(time.perf_counter() - queue_put_start_time) * 1000.0:.3f}",
                flush=True,
            )
            print(f"TRACE request_id={trace_id} span=server.queue_wait phase=begin", flush=True)
            future_wait_start_time = time.perf_counter()
            print(f"TRACE request_id={trace_id} span=server.future_wait phase=begin", flush=True)
            response = await asyncio.shield(pending_task.future)
            print(
                f"TRACE request_id={trace_id} span=server.future_wait phase=end "
                f"duration_ms={(time.perf_counter() - future_wait_start_time) * 1000.0:.3f}",
                flush=True,
            )
            return response
        finally:
            if not queued:
                release_start_time = time.perf_counter()
                print(
                    f"TRACE request_id={trace_id} span=server.release_blobs phase=begin", flush=True
                )
                self._release_blob_entries(blob_entries)
                print(
                    f"TRACE request_id={trace_id} span=server.release_blobs phase=end "
                    f"duration_ms={(time.perf_counter() - release_start_time) * 1000.0:.3f}",
                    flush=True,
                )
            print(
                f"TRACE request_id={trace_id} span=server.execute_request phase=end "
                f"duration_ms={(time.perf_counter() - request_start_time) * 1000.0:.3f}",
                flush=True,
            )

    async def _worker_loop(self, worker: LowLevelWorkerProcess) -> None:
        """Continuously pull queued tasks and execute them on one worker."""
        while True:
            pending_task = await self._pending_queue.get()
            pending_task.started_at = time.monotonic()
            queue_wait_seconds = pending_task.started_at - pending_task.enqueued_at
            print(
                f"TRACE request_id={pending_task.request_id} span=server.queue_wait phase=end "
                f"duration_ms={queue_wait_seconds * 1000.0:.3f}",
                flush=True,
            )
            remaining_timeout = pending_task.timeout_seconds - queue_wait_seconds
            worker_loop_start_time = time.perf_counter()
            print(
                f"TRACE request_id={pending_task.request_id} span=server.worker_loop phase=begin",
                flush=True,
            )
            worker_execute_start_time = time.perf_counter()
            try:
                if remaining_timeout <= 0:
                    server_response = ServerExecuteResponse(
                        kind=ServerExecuteResponseKind.TIMEOUT,
                        payload=ErrorResult(
                            error="timeout", timeout_seconds=pending_task.timeout_seconds
                        ),
                    )
                else:
                    print(
                        f"TRACE request_id={pending_task.request_id} span=server.worker_execute phase=begin",
                        flush=True,
                    )
                    response = await asyncio.to_thread(
                        worker.execute,
                        request_id=pending_task.request_id,
                        program=pending_task.program,
                        blobs=pending_task.blob_entries,
                        timeout_seconds=float(remaining_timeout),
                    )
                    print(
                        f"TRACE request_id={pending_task.request_id} span=server.worker_execute phase=end "
                        f"duration_ms={(time.perf_counter() - worker_execute_start_time) * 1000.0:.3f}",
                        flush=True,
                    )
                    if isinstance(response, WorkerExecuteErrorResponse):
                        server_response = ServerExecuteResponse(
                            kind=ServerExecuteResponseKind.ERROR, payload=response.error
                        )
                    else:
                        success_response: WorkerExecuteSuccessResponse = response
                        server_response = ServerExecuteResponse(
                            kind=ServerExecuteResponseKind.OK,
                            payload=success_response.result,
                            binary_parts=success_response.binary_parts,
                        )
            except TimeoutError as error:
                server_response = ServerExecuteResponse(
                    kind=ServerExecuteResponseKind.TIMEOUT,
                    payload=ErrorResult(error="timeout", timeout_seconds=error.timeout_seconds),
                )
            except asyncio.CancelledError:
                if not pending_task.future.done():
                    pending_task.future.cancel()
                raise
            except Exception as error:
                server_response = ServerExecuteResponse(
                    kind=ServerExecuteResponseKind.ERROR,
                    payload=ErrorResult(error="execution_failed", message=str(error)),
                )
            finally:
                print(
                    "SCHEDULER_TIMING "
                    f"request_id={pending_task.request_id} "
                    f"selected_gpu_id={worker.gpu_id} "
                    f"queue_wait_ms={queue_wait_seconds * 1000.0:.3f} "
                    f"worker_execute_ms={(time.perf_counter() - worker_execute_start_time) * 1000.0:.3f}",
                    flush=True,
                )
                if not pending_task.future.done():
                    pending_task.future.set_result(server_response)
                release_start_time = time.perf_counter()
                print(
                    f"TRACE request_id={pending_task.request_id} span=server.release_blobs phase=begin",
                    flush=True,
                )
                self._release_blob_entries(pending_task.blob_entries)
                print(
                    f"TRACE request_id={pending_task.request_id} span=server.release_blobs phase=end "
                    f"duration_ms={(time.perf_counter() - release_start_time) * 1000.0:.3f}",
                    flush=True,
                )
                print(
                    f"TRACE request_id={pending_task.request_id} span=server.worker_loop phase=end "
                    f"duration_ms={(time.perf_counter() - worker_loop_start_time) * 1000.0:.3f}",
                    flush=True,
                )
                self._pending_queue.task_done()

    def _drain_pending_queue(self) -> None:
        """Cancel and release every task still waiting in the global queue."""
        while True:
            try:
                pending_task = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending_task.future.done():
                pending_task.future.cancel()
            self._release_blob_entries(pending_task.blob_entries)
            self._pending_queue.task_done()

    def _release_blob_entries(self, blob_entries: dict[str, CacheEntry]) -> None:
        """Release acquired cache entries after the queued task fully finishes."""
        for acquired_entry in blob_entries.values():
            self.cache_manager.release(acquired_entry)

    def health_response(self) -> HealthResponse:
        """Return the current server health response payload.

        Returns
        -------
        HealthResponse
            Structured health summary for all managed workers.
        """
        workers = [
            WorkerStatus(
                gpu_id=worker.gpu_id, status=worker.status, uptime_seconds=worker.uptime_seconds
            )
            for worker in self._workers
        ]
        return HealthResponse(
            gpu_count=len(workers), pending_tasks=self._pending_queue.qsize(), workers=workers
        )


__all__ = ["LowLevelServer"]
